"""Ensemble extraction — dictionary-first entities, two-stage Sonnet pipeline.

Architecture:
  Stage 1: Entity Extraction
    1a. Dictionary scan (pattern_matcher._scan_entities_in_text)
    1b. Regex variant detector (variant_detector.detect_variants)
    1c. Sonnet entity gap-fill (asks for ADDITIONAL entities only)
    1d. Recover spans via string matching
    1e. Normalize all entities

  Stage 2: Relationship Extraction
    2a. Sonnet relationship extraction x N runs at temperature 0.5
    2b. Self-consistency: keep relationships appearing in majority of runs
    2c. Pattern matcher confirmation (for confidence scoring)

Strategy: Dictionary + regex are precise (high confidence). Sonnet fills gaps.
Relationship self-consistency replaces Haiku confirmation + pairwise classifier.
"""

import logging
import math
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
from .teacher_prompt import (
    build_entity_gapfill_prompt,
    parse_entity_gapfill_response,
)
from .teacher_prompt_conservative import (
    build_relationship_prompt,
    parse_relationship_response,
)
from .pattern_matcher import extract_with_patterns, _scan_entities_in_text
from .variant_detector import detect_variants
from .pattern_matcher import _STOPWORDS

logger = logging.getLogger(__name__)

# Entities that Sonnet sometimes hallucinates but are not real biomedical entities.
# Supplements the pattern_matcher _STOPWORDS which only applies to dictionary scan.
_GAPFILL_BLOCKLIST = {
    # Geographic / demographic terms Sonnet sometimes tags as ORGANISM
    "japan", "japanese", "china", "chinese", "korea", "korean",
    "european", "caucasian", "caucasians", "african", "asian",
    "american", "french", "german", "italian", "spanish", "british",
    # Study population descriptors
    "patient", "patients", "human", "humans", "subjects",
    "proband", "family", "families", "controls",
    # Generic molecular biology terms
    "dna", "rna", "mrna", "cdna", "protein", "gene", "peptide",
    "mutation", "mutations", "polymorphism", "polymorphisms",
    "wild-type", "wild type", "recombinant",
    "exon", "intron", "codon", "chromosome",
    # Methodology / techniques
    "pcr", "western blotting", "elisa", "flow cytometry",
    "real-time pcr", "sequencing",
}


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def _call_claude_tool_use(
    request_body: dict,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict | None:
    """Call Claude API with tool use and return parsed response JSON.

    Args:
        request_body: Dict with system, messages, tools, tool_choice.
        api_key: Anthropic API key.
        model: Model ID.
        max_tokens: Max output tokens.
        temperature: Sampling temperature.

    Returns:
        Full API response JSON, or None on failure.
    """
    try:
        import httpx
    except ImportError:
        return None

    try:
        with httpx.Client(timeout=90) as client:
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
                        "temperature": temperature,
                        "system": request_body["system"],
                        "messages": request_body["messages"],
                        "tools": request_body.get("tools", []),
                        "tool_choice": request_body.get("tool_choice"),
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
            return resp.json()
    except Exception as e:
        logger.warning("API call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Stage 1: Entity Extraction
# ---------------------------------------------------------------------------

def _extract_entities_dict_regex(
    text: str,
    dictionary: DictionaryLookup | None,
) -> list[ExtractedEntity]:
    """Stage 1a+1b: Dictionary scan + regex variant detection.

    Returns entities with high confidence (dictionary-grounded or regex-matched).
    """
    entities: list[ExtractedEntity] = []

    # 1a. Dictionary scan
    if dictionary and dictionary.is_available():
        dict_entities = _scan_entities_in_text(text, dictionary)
        for e in dict_entities:
            e.confidence = 0.95
        entities.extend(dict_entities)

    # 1b. Regex variant detection
    regex_variants = detect_variants(text)
    # Dedup against dictionary entities (by text_lower)
    existing_texts = {e.text.lower() for e in entities}
    for v in regex_variants:
        if v["text"].lower() not in existing_texts:
            entities.append(ExtractedEntity(
                text=v["text"],
                type="VARIANT",
                start=v["start"],
                end=v["end"],
                confidence=0.95,
            ))
            existing_texts.add(v["text"].lower())

    return entities


def _extract_entities_sonnet_gapfill(
    text: str,
    known_entities: list[ExtractedEntity],
    api_key: str,
) -> list[ExtractedEntity]:
    """Stage 1c: Sonnet entity gap-fill — find entities missed by dict+regex.

    Returns only NEW entities (not already in known_entities).
    """
    # Format known entities for the prompt
    known_dicts = [{"text": e.text, "type": e.type} for e in known_entities]

    request_body = build_entity_gapfill_prompt(text, known_dicts)
    model = os.environ.get("BIOEXTRACT_MODEL", "claude-sonnet-4-6")
    response = _call_claude_tool_use(request_body, api_key, model, temperature=0.0)

    if not response:
        return []

    new_entities_raw = parse_entity_gapfill_response(response)

    # Filter out entities already in known list, blocklisted, or stopwords
    known_texts = {e.text.lower() for e in known_entities}
    new_entities = []
    for e in new_entities_raw:
        text_lower = e["text"].lower()
        if text_lower in known_texts:
            continue
        if text_lower in _GAPFILL_BLOCKLIST or text_lower in _STOPWORDS:
            continue
        new_entities.append(ExtractedEntity(
            text=e["text"],
            type=e["type"],
            start=0,  # Will be recovered in _recover_entity_spans
            end=0,
            confidence=0.85,
        ))
        known_texts.add(text_lower)

    return new_entities


def _recover_entity_spans(text: str, entities: list[ExtractedEntity]):
    """Stage 1d: Recover character offsets for entities without spans.

    Modifies entities in place. Uses case-insensitive search, prefers first occurrence.
    """
    lower_text = text.lower()
    for e in entities:
        if e.start == 0 and e.end == 0:
            idx = lower_text.find(e.text.lower())
            if idx >= 0:
                # Use original text casing
                e.text = text[idx:idx + len(e.text)]
                e.start = idx
                e.end = idx + len(e.text)


# ---------------------------------------------------------------------------
# Stage 2: Relationship Extraction
# ---------------------------------------------------------------------------

def _extract_relationships_with_consistency(
    text: str,
    entities: list[ExtractedEntity],
    api_key: str,
    n_runs: int = 3,
) -> list[ExtractedRelationship]:
    """Stage 2a+2b: Sonnet relationship extraction with self-consistency.

    Runs extraction n_runs times at temperature=0.5, keeps relationships
    appearing in a majority of runs. Direction determined by majority vote.
    """
    entity_dicts = [{"text": e.text, "type": e.type} for e in entities]
    entity_texts = {e.text.lower() for e in entities}
    model = os.environ.get("BIOEXTRACT_MODEL", "claude-sonnet-4-6")

    # Collect all runs
    all_runs: list[list[dict]] = []
    for run_idx in range(n_runs):
        request_body = build_relationship_prompt(text, entity_dicts)
        temp = 0.5 if n_runs > 1 else 0.0
        response = _call_claude_tool_use(
            request_body, api_key, model,
            max_tokens=4096, temperature=temp,
        )
        if response:
            rels = parse_relationship_response(response, entity_texts)
            all_runs.append(rels)
        else:
            all_runs.append([])

    if not any(all_runs):
        return []

    # Aggregate: count each relationship across runs
    # Key: (subject_lower, object_lower, type)
    from collections import Counter

    rel_counts: Counter[tuple[str, str, str]] = Counter()
    rel_directions: dict[tuple[str, str, str], list[str]] = {}
    # Track original casing from first occurrence
    rel_original: dict[tuple[str, str, str], tuple[str, str]] = {}

    for run_rels in all_runs:
        for r in run_rels:
            key = (r["subject"].lower(), r["object"].lower(), r["type"])
            rel_counts[key] += 1
            rel_directions.setdefault(key, []).append(r["direction"])
            if key not in rel_original:
                rel_original[key] = (r["subject"], r["object"])

    # Keep relationships appearing in majority of runs
    threshold = math.ceil(n_runs / 2)
    results = []
    for key, count in rel_counts.items():
        if count >= threshold:
            subj_lower, obj_lower, rel_type = key
            original_subj, original_obj = rel_original[key]

            # Direction: majority vote
            directions = rel_directions[key]
            direction = max(set(directions), key=directions.count)

            # Confidence based on consistency
            confidence = round(count / n_runs, 2)
            # Map to confidence range
            if count == n_runs:
                confidence = 0.95
            else:
                confidence = 0.70

            results.append(ExtractedRelationship(
                subject=original_subj,
                predicate="",
                object=original_obj,
                type=rel_type,
                direction=direction,
                negated=False,
                context=RelationshipContext(),
                confidence=confidence,
            ))

    return results


# ---------------------------------------------------------------------------
# Main ensemble orchestrator
# ---------------------------------------------------------------------------

def extract_ensemble(
    text: str,
    dictionary: DictionaryLookup | None = None,
    normalizer: EntityNormalizer | None = None,
    skip_verifier: bool = False,  # kept for API compat, ignored
    n_runs: int | None = None,
) -> ExtractionResult:
    """Run the two-stage ensemble extraction pipeline.

    Stage 1: Dictionary + regex + Sonnet gap-fill → entities
    Stage 2: Sonnet self-consistency → relationships

    Args:
        text: Biomedical abstract text.
        dictionary: Optional dictionary for entity scanning and normalization.
        normalizer: Optional normalizer for post-processing.
        skip_verifier: Ignored (no verifier in this architecture).
        n_runs: Number of relationship extraction runs for self-consistency
            (default: 3, set to 1 for budget mode).

    Returns:
        ExtractionResult with confidence-scored entities and relationships.
    """
    if n_runs is None:
        n_runs = int(os.environ.get("BIOEXTRACT_CONSISTENCY_RUNS", "3"))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY — ensemble cannot run LLM models")
        if dictionary:
            return extract_with_patterns(text, dictionary=dictionary)
        return ExtractionResult(text=text, extraction_method="ensemble_failed")

    # -----------------------------------------------------------------------
    # Stage 1: Entities
    # -----------------------------------------------------------------------

    # 1a+1b: Dictionary + regex
    print("  [1/3] Dictionary + regex entities...", end=" ", flush=True)
    dict_regex_entities = _extract_entities_dict_regex(text, dictionary)
    print(f"{len(dict_regex_entities)}E", flush=True)

    # 1c: Sonnet gap-fill
    print("  [2/3] Sonnet entity gap-fill...", end=" ", flush=True)
    sonnet_entities = _extract_entities_sonnet_gapfill(text, dict_regex_entities, api_key)
    print(f"+{len(sonnet_entities)}E", flush=True)

    # Merge
    merged_entities = dict_regex_entities + sonnet_entities

    # 1d: Recover spans for Sonnet entities
    _recover_entity_spans(text, merged_entities)

    # 1e: Normalize all entities
    if normalizer and dictionary and dictionary.is_available():
        for e in merged_entities:
            normalizer.normalize_entity(e, context_text=text)

    print(f"  Merged: {len(merged_entities)}E "
          f"({len(dict_regex_entities)} dict/regex, {len(sonnet_entities)} gap-fill)",
          flush=True)

    # -----------------------------------------------------------------------
    # Stage 2: Relationships
    # -----------------------------------------------------------------------

    print(f"  [3/3] Sonnet relationships (x{n_runs} runs)...", end=" ", flush=True)
    relationships = _extract_relationships_with_consistency(
        text, merged_entities, api_key, n_runs=n_runs,
    )

    # Pattern matcher confirmation — boost confidence for confirmed relationships
    pattern_result = extract_with_patterns(text, precomputed_entities=merged_entities)
    pattern_pairs: set[tuple[str, str]] = set()
    for r in pattern_result.relationships:
        a, b = r.subject.lower(), r.object.lower()
        pattern_pairs.add((min(a, b), max(a, b)))

    for r in relationships:
        a, b = r.subject.lower(), r.object.lower()
        pair = (min(a, b), max(a, b))
        if pair in pattern_pairs:
            r.confidence = min(1.0, r.confidence + 0.05)

    # Count stats
    high_conf = sum(1 for r in relationships if r.confidence >= 0.90)
    print(f"{len(relationships)}R ({high_conf} high-confidence)", flush=True)

    return ExtractionResult(
        text=text,
        entities=merged_entities,
        relationships=relationships,
        extraction_method="ensemble",
    )
