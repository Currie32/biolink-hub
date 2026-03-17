"""Pairwise relationship classifier using batched Haiku calls.

Given a set of entities, systematically evaluates co-occurring entity pairs
for relationships. Uses a focused micro-prompt per batch of pairs.

Tiers:
  1. Same sentence — always classify
  2. Adjacent sentences — classify if biologically plausible pair types
  3. Abstract-wide thematic — title/first-sentence entities × primary disease/gene
"""

import json
import logging
import os
import re
import time

from ..schema import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    RelationshipContext,
)

logger = logging.getLogger(__name__)

RELATIONSHIP_TYPES = [
    "activates", "inhibits", "upregulates", "downregulates",
    "associated_with", "causes", "treats", "increases_risk",
    "decreases_risk", "binds", "regulates", "interacts_with",
    "NO_RELATIONSHIP",
]

# Biologically plausible pair types
PLAUSIBLE_PAIRS = {
    ("CHEMICAL", "DISEASE"), ("CHEMICAL", "GENE"), ("CHEMICAL", "CHEMICAL"),
    ("GENE", "DISEASE"), ("GENE", "GENE"), ("VARIANT", "DISEASE"),
    ("VARIANT", "GENE"), ("CHEMICAL", "VARIANT"),
}


def _is_plausible(type_a: str, type_b: str) -> bool:
    a, b = type_a.upper(), type_b.upper()
    return (a, b) in PLAUSIBLE_PAIRS or (b, a) in PLAUSIBLE_PAIRS


# ---------------------------------------------------------------------------
# Sentence utilities
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _find_entity_in_sentences(
    entity_text: str,
    sentences: list[str],
) -> set[int]:
    """Return indices of sentences containing this entity."""
    et = entity_text.lower()
    return {i for i, s in enumerate(sentences) if et in s.lower()}


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _generate_pairs(
    entities: list[ExtractedEntity],
    sentences: list[str],
) -> list[dict]:
    """Generate entity pairs to classify, organized by tier.

    Returns list of dicts: {entity_a, entity_b, sentence_context, tier}
    """
    # Map entities to sentence indices
    ent_sents: dict[str, set[int]] = {}
    for e in entities:
        ent_sents[e.text] = _find_entity_in_sentences(e.text, sentences)

    # Entity lookup by text
    ent_by_text: dict[str, ExtractedEntity] = {}
    for e in entities:
        ent_by_text[e.text] = e

    pairs = []
    seen = set()

    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            if e1.text == e2.text:
                continue
            if not _is_plausible(e1.type, e2.type):
                continue

            pair_key = (min(e1.text, e2.text), max(e1.text, e2.text))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            sents1 = ent_sents.get(e1.text, set())
            sents2 = ent_sents.get(e2.text, set())

            # Tier 1: Same sentence
            shared = sents1 & sents2
            if shared:
                sent_idx = min(shared)
                pairs.append({
                    "entity_a": e1,
                    "entity_b": e2,
                    "context": sentences[sent_idx],
                    "tier": 1,
                })
                continue

            # Tier 2: Adjacent sentences
            adjacent = False
            for s1 in sents1:
                for s2 in sents2:
                    if abs(s1 - s2) == 1:
                        # Combine both sentences as context
                        lo, hi = min(s1, s2), max(s1, s2)
                        ctx = " ".join(sentences[lo:hi + 1])
                        pairs.append({
                            "entity_a": e1,
                            "entity_b": e2,
                            "context": ctx,
                            "tier": 2,
                        })
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                continue

            # Tier 3: Thematic — only if one entity is in first sentence
            if 0 in sents1 or 0 in sents2:
                # Use first sentence + sentence containing the other entity
                other_sents = sents2 if 0 in sents1 else sents1
                if other_sents:
                    other_idx = min(other_sents)
                    ctx_parts = [sentences[0]]
                    if other_idx != 0:
                        ctx_parts.append(sentences[other_idx])
                    pairs.append({
                        "entity_a": e1,
                        "entity_b": e2,
                        "context": " ".join(ctx_parts),
                        "tier": 3,
                    })

    return pairs


# ---------------------------------------------------------------------------
# Batch prompt construction
# ---------------------------------------------------------------------------

BATCH_SYSTEM = """You are a biomedical relationship classifier. For each entity pair and its sentence context, determine if there is an explicit relationship.

Relationship types: {rel_types}

Rules:
- Only classify if the text STATES or CLEARLY IMPLIES a direct relationship.
- Co-occurrence in the same sentence is NOT enough — there must be a verb or phrase connecting them.
- Output NO_RELATIONSHIP if uncertain.
- For direction: "positive" (activates/causes/increases), "negative" (inhibits/treats/decreases), "neutral" (associated/binds).
""".format(rel_types=", ".join(RELATIONSHIP_TYPES))

BATCH_USER_TEMPLATE = """Classify each entity pair. Return a JSON array with one object per pair.

{pair_descriptions}

Return ONLY a JSON array:
[{{"pair": 1, "type": "associated_with", "direction": "neutral", "confidence": 0.8}}, ...]

Use "NO_RELATIONSHIP" with confidence 0.0 if no relationship exists."""


def _build_batch_prompt(pairs_batch: list[dict]) -> list[dict]:
    """Build a prompt for a batch of pairs."""
    descriptions = []
    for idx, p in enumerate(pairs_batch, 1):
        ea = p["entity_a"]
        eb = p["entity_b"]
        descriptions.append(
            f"Pair {idx}: \"{ea.text}\" ({ea.type}) <-> \"{eb.text}\" ({eb.type})\n"
            f"  Context: \"{p['context'][:400]}\""
        )

    return [
        {"role": "system", "content": BATCH_SYSTEM},
        {"role": "user", "content": BATCH_USER_TEMPLATE.format(
            pair_descriptions="\n\n".join(descriptions)
        )},
    ]


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_haiku(messages: list[dict], api_key: str) -> str | None:
    """Call Haiku API and return response text."""
    try:
        import httpx
    except ImportError:
        return None

    model = os.environ.get("BIOEXTRACT_PAIRWISE_MODEL", "claude-haiku-4-5-20251001")

    try:
        with httpx.Client(timeout=30) as client:
            for attempt in range(3):
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 2048,
                        "system": messages[0]["content"],
                        "messages": [messages[1]],
                    },
                )
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                break

            if resp.status_code != 200:
                logger.warning("Pairwise API error %d", resp.status_code)
                return None
            return resp.json()["content"][0]["text"]
    except Exception as e:
        logger.warning("Pairwise API call failed: %s", e)
        return None


def _parse_batch_response(
    response_text: str,
    pairs_batch: list[dict],
) -> list[ExtractedRelationship]:
    """Parse the batch classification response."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct parse first
    results = None
    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        # Haiku sometimes wraps JSON in prose — try to extract the array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                results = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if results is None:
        logger.warning("Failed to parse pairwise response: %s", text[:200])
        return []

    if not isinstance(results, list):
        return []

    relationships = []
    for item in results:
        rel_type = item.get("type", "NO_RELATIONSHIP")
        if rel_type == "NO_RELATIONSHIP":
            continue

        pair_idx = item.get("pair", 0) - 1  # 1-indexed
        if pair_idx < 0 or pair_idx >= len(pairs_batch):
            continue

        p = pairs_batch[pair_idx]
        confidence = float(item.get("confidence", 0.6))
        direction = item.get("direction", "neutral")

        # Apply tier-based confidence discount
        tier_discount = {1: 1.0, 2: 0.9, 3: 0.75}.get(p["tier"], 0.7)
        confidence *= tier_discount

        relationships.append(ExtractedRelationship(
            subject=p["entity_a"].text,
            predicate=f"pairwise_{rel_type}",
            object=p["entity_b"].text,
            type=rel_type,
            direction=direction,
            negated=False,
            context=RelationshipContext(),
            confidence=confidence,
        ))

    return relationships


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_pairs(
    text: str,
    entities: list[ExtractedEntity],
    batch_size: int = 8,
    exclude_pairs: set[tuple[str, str]] | None = None,
    tier_1_only: bool = False,
) -> list[ExtractedRelationship]:
    """Classify relationships between entity pairs using batched Haiku calls.

    Args:
        text: The abstract text.
        entities: Merged entities from other models.
        batch_size: Number of pairs per API call.
        exclude_pairs: Set of (min_text, max_text) pairs to skip (already found).
        tier_1_only: If True, only classify same-sentence pairs (tier 1).

    Returns:
        List of classified relationships with confidence scores.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY for pairwise classifier")
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Deduplicate entities by text
    seen_texts = set()
    unique_entities = []
    for e in entities:
        if e.text.lower() not in seen_texts:
            seen_texts.add(e.text.lower())
            unique_entities.append(e)

    pairs = _generate_pairs(unique_entities, sentences)
    if not pairs:
        return []

    # Filter: tier 1 only if requested
    if tier_1_only:
        pairs = [p for p in pairs if p["tier"] == 1]

    # Filter: exclude already-found pairs
    if exclude_pairs:
        filtered = []
        for p in pairs:
            a = p["entity_a"].text.lower().strip()
            b = p["entity_b"].text.lower().strip()
            pk = (min(a, b), max(a, b))
            if pk not in exclude_pairs:
                filtered.append(p)
        pairs = filtered

    if not pairs:
        return []

    tier_counts = {}
    for p in pairs:
        tier_counts[p["tier"]] = tier_counts.get(p["tier"], 0) + 1
    logger.info(
        "Pairwise: %d pairs (T1=%d, T2=%d, T3=%d)",
        len(pairs),
        tier_counts.get(1, 0),
        tier_counts.get(2, 0),
        tier_counts.get(3, 0),
    )

    # Batch and classify
    all_rels = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        messages = _build_batch_prompt(batch)
        response = _call_haiku(messages, api_key)
        if response:
            rels = _parse_batch_response(response, batch)
            all_rels.extend(rels)

    return all_rels
