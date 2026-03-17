"""Ensemble extraction — Sonnet primary + targeted improvements.

Architecture:
  1. Sonnet full extraction (primary — all entities and relationships)
  2. Haiku extraction (entity gap-fill for grounded entities Sonnet missed,
     relationship confirmation)
  3. Pattern matcher (relationship confirmation via verb evidence)
  4. Pairwise classifier (targeted gap-fill for variant-involving same-sentence
     pairs that Sonnet missed)

Strategy: Sonnet is the baseline. Other models can:
  - CONFIRM Sonnet's outputs (boost confidence)
  - GAP-FILL entities (only if dictionary-grounded)
  - GAP-FILL relationships (only variant pairs in same sentence, with guards)
  - Never contradict or override Sonnet
"""

import logging
import os
import time

from ..schema import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    RelationshipContext,
)
from ..dictionaries.lookup import DictionaryLookup
from ..normalize import EntityNormalizer
from .teacher_prompt import build_extraction_prompt, parse_teacher_output
from .teacher_prompt_conservative import (
    build_conservative_prompt,
    parse_conservative_output,
)
from .pattern_matcher import extract_with_patterns
from .pairwise_classifier import classify_pairs

logger = logging.getLogger(__name__)


# Entity types that are allowed for gap-fill from Haiku
# (must also be dictionary-grounded to be added)
_GAPFILL_ENTITY_TYPES = {"GENE", "CHEMICAL", "DISEASE", "VARIANT"}


# ---------------------------------------------------------------------------
# Individual model runners
# ---------------------------------------------------------------------------

def _call_claude(
    messages: list[dict],
    api_key: str,
    model: str,
    max_tokens: int = 8192,
) -> str | None:
    """Generic Claude API call with retry."""
    try:
        import httpx
    except ImportError:
        return None

    try:
        with httpx.Client(timeout=60) as client:
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
                        "max_tokens": max_tokens,
                        "system": messages[0]["content"],
                        "messages": [messages[1]],
                    },
                )
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"  (rate limited, waiting {wait}s)", end=" ", flush=True)
                    time.sleep(wait)
                    continue
                break

            if resp.status_code != 200:
                logger.warning("API error %d: %s", resp.status_code, resp.text[:200])
                return None
            return resp.json()["content"][0]["text"]
    except Exception as e:
        logger.warning("API call failed: %s", e)
        return None


def _run_sonnet_extraction(
    text: str,
    api_key: str,
    dictionary: DictionaryLookup | None = None,
) -> ExtractionResult | None:
    """Model 1: Sonnet full extraction (primary)."""
    known_entities = None
    if dictionary and dictionary.is_available():
        from ..extract import BioExtractor
        extractor = BioExtractor.__new__(BioExtractor)
        extractor._dictionary = dictionary
        known_entities = extractor._get_dictionary_hints(text)

    messages = build_extraction_prompt(text, known_entities)
    model = os.environ.get("BIOEXTRACT_MODEL", "claude-sonnet-4-6")
    response = _call_claude(messages, api_key, model)

    if not response:
        return None

    parsed = parse_teacher_output(response)
    if "parse_error" in parsed:
        return None

    entities = []
    for e in parsed.get("entities", []):
        entities.append(ExtractedEntity(
            text=e["text"],
            type=e["type"],
            start=e.get("start", 0),
            end=e.get("end", 0),
            confidence=0.85,
        ))

    relationships = []
    for r in parsed.get("relationships", []):
        ctx = r.get("context", {})
        relationships.append(ExtractedRelationship(
            subject=r["subject"],
            predicate=r.get("predicate", ""),
            object=r["object"],
            type=r["type"],
            direction=r.get("direction", "neutral"),
            negated=r.get("negated", False),
            context=RelationshipContext(
                organism=ctx.get("organism"),
                cell_type=ctx.get("cell_type"),
                experiment_type=ctx.get("experiment_type"),
            ),
            confidence=0.85,
        ))

    return ExtractionResult(
        text=text,
        entities=entities,
        relationships=relationships,
        extraction_method="sonnet_full",
    )


def _run_haiku_extraction(text: str, api_key: str) -> ExtractionResult | None:
    """Model 2: Haiku conservative extraction."""
    messages = build_conservative_prompt(text)
    model = os.environ.get("BIOEXTRACT_HAIKU_MODEL", "claude-haiku-4-5-20251001")
    response = _call_claude(messages, api_key, model, max_tokens=4096)

    if not response:
        return None

    parsed = parse_conservative_output(response)
    if "parse_error" in parsed:
        return None

    entities = []
    for e in parsed.get("entities", []):
        entities.append(ExtractedEntity(
            text=e["text"],
            type=e["type"],
            start=e.get("start", 0),
            end=e.get("end", 0),
            confidence=0.70,
        ))

    relationships = []
    for r in parsed.get("relationships", []):
        relationships.append(ExtractedRelationship(
            subject=r["subject"],
            predicate=r.get("predicate", ""),
            object=r["object"],
            type=r["type"],
            direction=r.get("direction", "neutral"),
            negated=r.get("negated", False),
            context=RelationshipContext(),
            confidence=0.70,
        ))

    return ExtractionResult(
        text=text,
        entities=entities,
        relationships=relationships,
        extraction_method="haiku_conservative",
    )


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _merge_entities(
    sonnet: ExtractionResult | None,
    haiku: ExtractionResult | None,
    normalizer: EntityNormalizer | None,
    dictionary: DictionaryLookup | None,
    text: str,
) -> list[ExtractedEntity]:
    """Merge entities: Sonnet primary + Haiku gap-fill for grounded entities.

    Rules:
    - All Sonnet entities kept. Haiku confirmation boosts confidence.
    - Haiku-only entities added ONLY if:
      1. They are a gap-fillable type (GENE, CHEMICAL, DISEASE, VARIANT)
      2. They can be grounded to a dictionary canonical_id after normalization
      3. No existing Sonnet entity already covers the same canonical_id
    """
    if sonnet is None or len(sonnet.entities) == 0:
        # Sonnet failed — use all Haiku entities as primary (normalize them)
        if haiku and haiku.entities:
            entities = list(haiku.entities)
            if normalizer and dictionary and dictionary.is_available():
                for e in entities:
                    normalizer.normalize_entity(e, context_text=text)
            return entities
        return []

    # Index Haiku entities by lowered text
    haiku_by_key: dict[str, ExtractedEntity] = {}
    if haiku:
        for e in haiku.entities:
            haiku_by_key[e.text.lower().strip()] = e

    # Start with all Sonnet entities, boosting confirmed ones
    merged = []
    sonnet_keys: set[str] = set()
    for e in sonnet.entities:
        key = e.text.lower().strip()
        sonnet_keys.add(key)
        if key in haiku_by_key:
            e.confidence = min(1.0, e.confidence + 0.10)
        merged.append(e)

    # Normalize Sonnet entities first so we can check canonical_id overlap
    sonnet_canonical_ids: set[str] = set()
    if normalizer and dictionary and dictionary.is_available():
        for e in merged:
            normalizer.normalize_entity(e, context_text=text)
            if e.canonical_id:
                sonnet_canonical_ids.add(e.canonical_id)

    # Gap-fill: Haiku-only entities that are grounded
    gap_filled = 0
    for key, e in haiku_by_key.items():
        if key in sonnet_keys:
            continue  # Already in Sonnet output

        # Only gap-fill certain types
        if e.type.upper() not in _GAPFILL_ENTITY_TYPES:
            continue

        # Try to ground it via normalization
        if normalizer and dictionary and dictionary.is_available():
            normalizer.normalize_entity(e, context_text=text)

        if not e.canonical_id:
            continue  # Can't ground it — don't add noise

        # Don't add if Sonnet already has an entity with the same canonical ID
        # (e.g., Haiku says "sodium" but Sonnet already has "Na+" with same ID)
        if e.canonical_id in sonnet_canonical_ids:
            continue

        # Passed all guards — add at reduced confidence
        e.confidence = 0.70
        merged.append(e)
        sonnet_canonical_ids.add(e.canonical_id)
        gap_filled += 1

    if gap_filled:
        logger.info("Entity gap-fill: added %d Haiku-only grounded entities", gap_filled)

    return merged


def _score_relationships(
    sonnet: ExtractionResult | None,
    haiku: ExtractionResult | None,
    pattern_rels: list[ExtractedRelationship],
    pairwise_rels: list[ExtractedRelationship],
) -> list[ExtractedRelationship]:
    """Score Sonnet relationships using confirmation from other models.

    Only Sonnet relationships are kept. Other models provide confidence signals:
    - Haiku confirmation: strong signal (independent LLM agrees)
    - Pattern matcher: medium signal (verb evidence exists in text)
    - Pairwise classifier: medium signal (focused same-sentence classification)

    Confidence adjustments:
    - 2+ confirmations: +0.12 (high confidence)
    - 1 confirmation: +0.05 (moderate confidence)
    - 0 confirmations: -0.10 (Sonnet-only, less certain)
    """
    if sonnet is None:
        return []

    def _pair_key(subj: str, obj: str) -> tuple[str, str]:
        a, b = subj.lower().strip(), obj.lower().strip()
        return (min(a, b), max(a, b))

    # Index confirmation sources
    haiku_pairs: set[tuple[str, str]] = set()
    pattern_pairs: set[tuple[str, str]] = set()
    pairwise_pairs: set[tuple[str, str]] = set()

    if haiku:
        for r in haiku.relationships:
            haiku_pairs.add(_pair_key(r.subject, r.object))

    for r in pattern_rels:
        pattern_pairs.add(_pair_key(r.subject, r.object))

    for r in pairwise_rels:
        pairwise_pairs.add(_pair_key(r.subject, r.object))

    scored = []
    for r in sonnet.relationships:
        pk = _pair_key(r.subject, r.object)

        confirmations = 0
        if pk in haiku_pairs:
            confirmations += 1
        if pk in pattern_pairs:
            confirmations += 1
        if pk in pairwise_pairs:
            confirmations += 1

        if confirmations >= 2:
            r.confidence = min(1.0, r.confidence + 0.12)
        elif confirmations == 1:
            r.confidence = min(1.0, r.confidence + 0.05)
        else:
            # No confirmation — reduce confidence
            r.confidence = max(0.5, r.confidence - 0.10)

        scored.append(r)

    return scored


# ---------------------------------------------------------------------------
# Main ensemble orchestrator
# ---------------------------------------------------------------------------

def extract_ensemble(
    text: str,
    dictionary: DictionaryLookup | None = None,
    normalizer: EntityNormalizer | None = None,
    skip_verifier: bool = False,  # kept for API compat, ignored
) -> ExtractionResult:
    """Run the ensemble extraction pipeline.

    Architecture: Sonnet primary + targeted improvements from other models.

    Args:
        text: Biomedical abstract text.
        dictionary: Optional dictionary for pattern matching and hints.
        normalizer: Optional normalizer for post-processing.
        skip_verifier: Ignored (no verifier in this architecture).

    Returns:
        ExtractionResult with confidence-scored entities and relationships.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY — ensemble cannot run LLM models")
        if dictionary:
            return extract_with_patterns(text, dictionary=dictionary)
        return ExtractionResult(text=text, extraction_method="ensemble_failed")

    # Step 1: Sonnet full extraction (primary)
    print("  [1/4] Sonnet full extraction...", end=" ", flush=True)
    sonnet_result = _run_sonnet_extraction(text, api_key, dictionary)
    sonnet_ent_count = len(sonnet_result.entities) if sonnet_result else 0
    sonnet_rel_count = len(sonnet_result.relationships) if sonnet_result else 0
    print(f"{sonnet_ent_count}E/{sonnet_rel_count}R", flush=True)

    # Step 2: Haiku extraction (entity gap-fill + relationship confirmation)
    print("  [2/4] Haiku extraction...", end=" ", flush=True)
    haiku_result = _run_haiku_extraction(text, api_key)
    haiku_ent_count = len(haiku_result.entities) if haiku_result else 0
    haiku_rel_count = len(haiku_result.relationships) if haiku_result else 0
    print(f"{haiku_ent_count}E/{haiku_rel_count}R", flush=True)

    # Step 3: Merge entities (Sonnet + grounded Haiku gap-fill)
    print("  Merging entities...", end=" ", flush=True)
    merged_entities = _merge_entities(
        sonnet_result, haiku_result, normalizer, dictionary, text,
    )
    sonnet_keys = set()
    if sonnet_result:
        sonnet_keys = {e.text.lower().strip() for e in sonnet_result.entities}
    haiku_only = sum(
        1 for e in merged_entities
        if e.text.lower().strip() not in sonnet_keys
    )
    haiku_confirmed = 0
    if haiku_result:
        haiku_keys = {e.text.lower().strip() for e in haiku_result.entities}
        haiku_confirmed = sum(
            1 for e in merged_entities
            if e.text.lower().strip() in sonnet_keys
            and e.text.lower().strip() in haiku_keys
        )
    gap_str = f" +{haiku_only} gap-filled" if haiku_only else ""
    print(
        f"{len(merged_entities)}E ({haiku_confirmed} confirmed{gap_str})",
        flush=True,
    )

    # Step 4: Pattern matcher — relationship confirmation using merged entities
    print("  [3/4] Pattern matcher...", end=" ", flush=True)
    pattern_result = extract_with_patterns(
        text, precomputed_entities=merged_entities,
    )
    print(f"{len(pattern_result.relationships)}R", flush=True)

    # Step 5: Pairwise classifier — same-sentence pairs for confirmation
    print("  [4/4] Pairwise classifier...", end=" ", flush=True)
    pairwise_rels = classify_pairs(
        text, merged_entities, batch_size=8,
        tier_1_only=True,
    )
    print(f"{len(pairwise_rels)}R", flush=True)

    # Step 6: Score relationships using confirmation signals
    # If Sonnet produced results, only Sonnet relationships survive.
    # If Sonnet failed (0 entities), fall back to Haiku as primary.
    sonnet_failed = sonnet_result is None or len(sonnet_result.entities) == 0
    if sonnet_failed and haiku_result and len(haiku_result.relationships) > 0:
        print("  (Sonnet failed — using Haiku as primary) ", end="", flush=True)
        primary_result = haiku_result
    else:
        primary_result = sonnet_result

    print("  Scoring relationships...", end=" ", flush=True)
    scored_relationships = _score_relationships(
        primary_result,
        haiku_result if not sonnet_failed else None,
        pattern_result.relationships,
        pairwise_rels,
    )

    # Count confirmation stats
    confirmed_count = sum(1 for r in scored_relationships if r.confidence > 0.85)
    unconfirmed_count = sum(1 for r in scored_relationships if r.confidence < 0.85)
    print(
        f"{len(scored_relationships)}R "
        f"({confirmed_count} confirmed, {unconfirmed_count} unconfirmed)",
        flush=True,
    )

    # Entities are already normalized in _merge_entities (needed for grounding check)

    return ExtractionResult(
        text=text,
        entities=merged_entities,
        relationships=scored_relationships,
        extraction_method="ensemble",
    )
