"""Shared utilities for progressive student model training.

Provides data loading, diverse sampling, and evaluation metrics
used by both Approach A (single generative) and Approach B (split pipeline).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_data(path: str) -> list[dict]:
    """Load JSONL training/test data."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def diverse_sample(examples: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Select n examples via greedy set-cover over entity/relationship types.

    Prioritizes examples that contribute new entity types or relationship types
    not yet covered by the selected set. Falls back to random selection once
    all types are covered.
    """
    import random

    if n >= len(examples):
        return list(examples)

    rng = random.Random(seed)

    def get_types(ex):
        """Extract the set of entity and relationship types from an example."""
        types = set()
        for e in ex.get("entities", []):
            types.add(("entity", e["type"]))
        for r in ex.get("relationships", []):
            types.add(("rel", r["type"]))
        return types

    selected = []
    selected_indices = set()
    covered_types = set()

    # Greedy set-cover: pick example that covers the most uncovered types
    for _ in range(n):
        best_idx = None
        best_new = -1
        # Shuffle candidates to break ties randomly
        candidates = list(range(len(examples)))
        rng.shuffle(candidates)

        for idx in candidates:
            if idx in selected_indices:
                continue
            types = get_types(examples[idx])
            new_count = len(types - covered_types)
            if new_count > best_new:
                best_new = new_count
                best_idx = idx

        if best_idx is not None:
            selected.append(examples[best_idx])
            selected_indices.add(best_idx)
            covered_types |= get_types(examples[best_idx])

    logger.info(
        "Sampled %d examples covering %d types (from %d available)",
        len(selected), len(covered_types), len(examples),
    )
    return selected


def compute_metrics(predictions: list[dict], gold: list[dict]) -> dict:
    """Compute entity F1, relationship F1, direction accuracy, and JSON parse rate.

    Args:
        predictions: List of dicts with 'entities' and 'relationships' keys.
            May contain None entries for unparseable outputs.
        gold: List of gold-standard dicts (same format).

    Returns:
        Dict with entity_f1, relationship_f1, direction_accuracy, json_parse_rate,
        and detailed counts.
    """
    entity_tp = entity_fp = entity_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    direction_correct = direction_total = 0
    parse_success = 0

    for pred, gold_ex in zip(predictions, gold):
        if pred is None:
            # Unparseable output — all gold items are false negatives
            entity_fn += len(gold_ex.get("entities", []))
            rel_fn += len(gold_ex.get("relationships", []))
            continue

        parse_success += 1

        # Entity evaluation: (text_lower, type) match
        gold_entities = {
            (e["text"].lower(), e["type"])
            for e in gold_ex.get("entities", [])
        }
        pred_entities = {
            (e["text"].lower(), e["type"])
            for e in pred.get("entities", [])
        }
        entity_tp += len(gold_entities & pred_entities)
        entity_fp += len(pred_entities - gold_entities)
        entity_fn += len(gold_entities - pred_entities)

        # Relationship evaluation: (subject_lower, object_lower, type) match
        gold_rels = {
            (r["subject"].lower(), r["object"].lower(), r["type"])
            for r in gold_ex.get("relationships", [])
        }
        pred_rels = {
            (r["subject"].lower(), r["object"].lower(), r["type"])
            for r in pred.get("relationships", [])
        }
        rel_tp += len(gold_rels & pred_rels)
        rel_fp += len(pred_rels - gold_rels)
        rel_fn += len(gold_rels - pred_rels)

        # Direction accuracy on matched relationships
        gold_dir = {
            (r["subject"].lower(), r["object"].lower(), r["type"]): r.get("direction", "neutral")
            for r in gold_ex.get("relationships", [])
        }
        for r in pred.get("relationships", []):
            key = (r["subject"].lower(), r["object"].lower(), r["type"])
            if key in gold_dir:
                direction_total += 1
                if r.get("direction") == gold_dir[key]:
                    direction_correct += 1

    def f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total = len(gold)
    return {
        "entity_f1": round(f1(entity_tp, entity_fp, entity_fn), 4),
        "relationship_f1": round(f1(rel_tp, rel_fp, rel_fn), 4),
        "direction_accuracy": round(
            direction_correct / direction_total if direction_total > 0 else 0.0, 4
        ),
        "json_parse_rate": round(parse_success / total if total > 0 else 0.0, 4),
        "entity_counts": {"tp": entity_tp, "fp": entity_fp, "fn": entity_fn},
        "relationship_counts": {"tp": rel_tp, "fp": rel_fp, "fn": rel_fn},
        "n_examples": total,
    }


def compute_ner_metrics(predictions: list[list[dict]], gold: list[list[dict]]) -> dict:
    """Compute NER-specific entity F1 (text + type match).

    Args:
        predictions: List of entity lists (each entity is a dict with text, type).
        gold: List of gold entity lists.
    """
    tp = fp = fn = 0
    for pred_ents, gold_ents in zip(predictions, gold):
        gold_set = {(e["text"].lower(), e["type"]) for e in gold_ents}
        pred_set = {(e["text"].lower(), e["type"]) for e in pred_ents}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "entity_f1": round(f1(tp, fp, fn), 4),
        "entity_precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4),
        "entity_recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "entity_counts": {"tp": tp, "fp": fp, "fn": fn},
    }
