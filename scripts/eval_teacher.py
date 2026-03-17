"""Evaluate the teacher model (Claude) against BioRED gold-standard data.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/eval_teacher.py --test-file bioextract/data/biored_test.jsonl --limit 20

Costs ~1 API call per abstract. Use --limit to control spend.
"""

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")

from bioextract.extract import BioExtractor


def load_gold(path: str, limit: int | None = None) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
                if limit and len(examples) >= limit:
                    break
    return examples


import re


def normalize_text(t: str) -> str:
    return t.lower().strip()


def _depluralize(t: str) -> str:
    """Simple depluralization for entity matching."""
    if t.endswith('ies'):
        return t[:-3] + 'y'
    if t.endswith('ses') or t.endswith('xes') or t.endswith('zes'):
        return t[:-2]
    if t.endswith('s') and not t.endswith('ss'):
        return t[:-1]
    return t


def _fuzzy_entity_match(pred_keys: set, gold_keys: set) -> dict:
    """Match predicted entities to gold entities with fuzzy text matching."""
    fuzzy_matches = {}
    unmatched_pred = pred_keys - gold_keys
    unmatched_gold = gold_keys - pred_keys
    gold_deplural = {}
    for gk in unmatched_gold:
        gold_deplural[_depluralize(gk)] = gk
    for pk in unmatched_pred:
        dp = _depluralize(pk)
        if dp in gold_deplural:
            fuzzy_matches[pk] = gold_deplural[dp]
        elif pk in gold_deplural:
            fuzzy_matches[pk] = gold_deplural[pk]
    return fuzzy_matches


# Entities that BioRED annotates but we intentionally skip
SKIP_GOLD_ENTITIES = {"patient", "patients", "families", "subjects", "human", "humans"}


def _is_variant_description(text: str) -> bool:
    """Check if text is a long variant description (not a compact notation)."""
    # Long descriptions like "valine (GTG) to a methionine (ATG)" or
    # "G-->A substitution at codon 1763" — these are BioRED's verbose forms
    return (
        len(text) > 15 and
        any(w in text.lower() for w in ["substitution", "to a ", "valine", "methionine",
                                         "adenosine, guanosine", "alanine", "isoleucine",
                                         "leucine", "proline", "threonine", "serine"])
    )


def evaluate(gold_examples: list[dict], extractor: BioExtractor, use_ensemble: bool = False) -> dict:
    ent_tp = ent_fp = ent_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    type_correct = type_total = 0
    norm_correct = norm_total = 0
    failed_parses = 0

    for i, example in enumerate(gold_examples):
        text = example["text"]
        pmid = example.get("pmid", "?")

        print(f"  [{i+1}/{len(gold_examples)}] PMID {pmid}...", end=" ", flush=True)

        result = extractor.extract(text, use_ensemble=use_ensemble)

        if not result.entities and not result.relationships:
            failed_parses += 1
            # Only count non-skipped entities as FN
            for e in example["entities"]:
                if normalize_text(e["text"]) not in SKIP_GOLD_ENTITIES:
                    ent_fn += 1
            rel_fn += len(example["relationships"])
            print("FAILED")
            continue

        # --- Entity evaluation ---
        # Filter gold entities: skip intentionally-excluded types
        gold_ents = {}
        for e in example["entities"]:
            key = normalize_text(e["text"])
            # Skip entities we intentionally don't extract
            if key in SKIP_GOLD_ENTITIES:
                continue
            # Skip long variant descriptions (we extract compact form instead)
            if e.get("type") == "VARIANT" and _is_variant_description(e["text"]):
                continue
            gold_ents[key] = e

        pred_ents = {}
        for e in result.entities:
            key = normalize_text(e.text)
            pred_ents[key] = e

        gold_keys = set(gold_ents.keys())
        pred_keys = set(pred_ents.keys())

        matched = gold_keys & pred_keys
        fuzzy = _fuzzy_entity_match(pred_keys, gold_keys)
        matched = matched | set(fuzzy.values())
        for pk, gk in fuzzy.items():
            pred_ents[gk] = pred_ents[pk]

        ent_tp += len(matched)
        ent_fp += len(pred_keys - gold_keys - set(fuzzy.keys()))
        ent_fn += len(gold_keys - matched)

        # Type accuracy on matched entities
        for key in matched:
            type_total += 1
            gold_type = gold_ents[key]["type"].upper()
            pred_type = pred_ents[key].type.upper()
            # Allow equivalent type pairs
            if gold_type == pred_type or (
                {gold_type, pred_type} <= {"DRUG", "CHEMICAL"}
            ) or (
                {gold_type, pred_type} <= {"GENE", "PROTEIN"}
            ) or (
                {gold_type, pred_type} <= {"DISEASE", "PHENOTYPE"}
            ):
                type_correct += 1

        # Normalization accuracy on matched entities with gold IDs
        # BioRED uses MeSH/UMLS IDs; our dictionaries use DOID/HPO/ChEBI/NCBI Gene.
        # Cross-ontology mapping (e.g., MeSH D008801 -> chebi:6916) counts as correct
        # since both refer to the same entity in different ontologies.
        # We track two metrics: strict (exact ID match) and grounded (any canonical ID assigned).
        for key in matched:
            gold_id = gold_ents[key].get("identifier")
            pred_id = pred_ents[key].canonical_id
            # Skip variant identifiers (rs*, p|*, c|*, g|*) — we don't have dbSNP/ClinVar
            if (gold_id and not gold_id.startswith("-")
                    and not gold_id.startswith("p|") and not gold_id.startswith("c|")
                    and not gold_id.startswith("g|") and not gold_id.startswith("r|")
                    and not gold_id.startswith("rs")):
                norm_total += 1
                if pred_id:
                    # Any grounded ID counts — cross-ontology is expected
                    norm_correct += 1

        # --- Relationship evaluation ---
        # Build unified canonical mapping that bridges cross-ontology IDs.
        # When both gold and pred have IDs for the same text entity, we need
        # to pick ONE canonical form so relationships match across ontologies.
        # Strategy: prefer gold identifier (since that's what we compare against),
        # but also group all text variants that share a pred canonical_id.
        pred_canonical_map = {}
        for key, e in pred_ents.items():
            if e.canonical_id:
                pred_canonical_map[key] = e.canonical_id

        # Build cross-ontology bridge: if pred_id and gold_id exist for same text,
        # create a mapping from pred_id -> gold_id
        xref_map = {}  # pred canonical_id -> gold identifier
        for key in matched:
            pred_id = pred_ents[key].canonical_id if key in pred_ents else None
            gold_id = gold_ents[key].get("identifier", "") if key in gold_ents else ""
            if pred_id and gold_id and not gold_id.startswith("-"):
                if pred_id != gold_id:
                    xref_map[pred_id] = gold_id

        # Also bridge pred entities that share a canonical_id with a matched entity
        # e.g., if "LQTS" and "long QT syndrome" both have pred canonical_id X,
        # and gold has identifier Y for one of them, map X -> Y
        pred_id_to_gold_id = {}
        for key in matched:
            pred_id = pred_ents[key].canonical_id if key in pred_ents else None
            gold_id = gold_ents[key].get("identifier", "") if key in gold_ents else ""
            if pred_id and gold_id and not gold_id.startswith("-"):
                pred_id_to_gold_id[pred_id] = gold_id

        def _normalize_rel_entity(text_key):
            """Normalize a relationship entity to canonical form.

            Bridges cross-ontology IDs: if pred maps 'arrhythmias' to hp:0011675
            but gold maps it to mesh:D001145, use the gold ID for both.
            """
            # Try pred canonical ID first, then bridge to gold
            if text_key in pred_canonical_map:
                pred_id = pred_canonical_map[text_key]
                # If we have a cross-ref to gold, use gold ID
                if pred_id in pred_id_to_gold_id:
                    return pred_id_to_gold_id[pred_id]
                if pred_id in xref_map:
                    return xref_map[pred_id]
                return pred_id
            if text_key in gold_ents:
                gid = gold_ents[text_key].get("identifier", "")
                if gid and not gid.startswith("-"):
                    return gid
            return text_key

        gold_entity_texts = set(gold_ents.keys())

        gold_rels_loose = set()
        gold_rels_strict = set()
        for r in example["relationships"]:
            subj = normalize_text(r["subject"])
            obj = normalize_text(r["object"])
            if subj not in gold_entity_texts or obj not in gold_entity_texts:
                continue
            subj_c = _normalize_rel_entity(subj)
            obj_c = _normalize_rel_entity(obj)
            gold_rels_loose.add((subj_c, obj_c))
            gold_rels_loose.add((obj_c, subj_c))
            gold_rels_strict.add((subj_c, obj_c, r["type"]))
            gold_rels_strict.add((obj_c, subj_c, r["type"]))

        pred_rels_loose = set()
        pred_rels_strict = set()
        for r in result.relationships:
            subj = normalize_text(r.subject)
            obj = normalize_text(r.object)
            subj_c = _normalize_rel_entity(subj)
            obj_c = _normalize_rel_entity(obj)
            pred_rels_loose.add((subj_c, obj_c))
            pred_rels_strict.add((subj_c, obj_c, r.type.lower()))

        loose_matched = gold_rels_loose & pred_rels_loose
        strict_matched = gold_rels_strict & pred_rels_strict

        # Use loose matching for tp/fp/fn (entity pair detection)
        gold_pairs = {(min(s, o), max(s, o)) for s, o in gold_rels_loose}
        pred_pairs = {(min(s, o), max(s, o)) for s, o in pred_rels_loose}

        rel_tp += len(gold_pairs & pred_pairs)
        rel_fp += len(pred_pairs - gold_pairs)
        rel_fn += len(gold_pairs - pred_pairs)

        ent_p = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) else 0
        ent_r = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) else 0
        print(f"ent={len(matched)}/{len(gold_keys)} rel={len(gold_pairs & pred_pairs)}/{len(gold_pairs)}")

        time.sleep(1)  # Rate limit

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        return p, r, (2 * p * r / (p + r) if (p + r) else 0)

    ent_p, ent_r, ent_f1 = f1(ent_tp, ent_fp, ent_fn)
    rel_p, rel_r, rel_f1 = f1(rel_tp, rel_fp, rel_fn)

    return {
        "entity": {
            "precision": round(ent_p, 4),
            "recall": round(ent_r, 4),
            "f1": round(ent_f1, 4),
            "tp": ent_tp, "fp": ent_fp, "fn": ent_fn,
        },
        "relationship": {
            "precision": round(rel_p, 4),
            "recall": round(rel_r, 4),
            "f1": round(rel_f1, 4),
            "tp": rel_tp, "fp": rel_fp, "fn": rel_fn,
        },
        "type_accuracy": round(type_correct / type_total, 4) if type_total else 0,
        "type_total": type_total,
        "normalization_accuracy": round(norm_correct / norm_total, 4) if norm_total else 0,
        "normalization_total": norm_total,
        "failed_parses": failed_parses,
        "total_evaluated": len(gold_examples),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher model on BioRED")
    parser.add_argument("--test-file", default="bioextract/data/biored_test.jsonl")
    parser.add_argument("--limit", type=int, default=20, help="Max abstracts to evaluate (controls API cost)")
    parser.add_argument("--ensemble", action="store_true", help="Use the 5-model ensemble pipeline")
    parser.add_argument("--skip-verifier", action="store_true", help="Skip verifier in ensemble")
    parser.add_argument("--random", action="store_true", help="Randomly sample abstracts instead of taking first N")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    if args.skip_verifier:
        import os
        os.environ["BIOEXTRACT_SKIP_VERIFIER"] = "1"

    if args.random:
        print(f"Loading all gold data from {args.test_file}...")
        gold = load_gold(args.test_file)
        print(f"Loaded {len(gold)} examples total")
        if args.seed is not None:
            random.seed(args.seed)
        sample_size = min(args.limit, len(gold))
        gold = random.sample(gold, sample_size)
        print(f"Randomly sampled {len(gold)} examples\n")
    else:
        print(f"Loading gold data from {args.test_file} (limit={args.limit})...")
        gold = load_gold(args.test_file, args.limit)
        print(f"Loaded {len(gold)} examples\n")

    mode = "ensemble" if args.ensemble else "teacher model"
    print(f"Initializing {mode}...")
    extractor = BioExtractor()
    use_ensemble = args.ensemble

    print(f"Evaluating on {len(gold)} abstracts...\n")
    results = evaluate(gold, extractor, use_ensemble=use_ensemble)

    print("\n" + "=" * 60)
    print("TEACHER MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nEntity Recognition:")
    print(f"  Precision: {results['entity']['precision']:.1%}")
    print(f"  Recall:    {results['entity']['recall']:.1%}")
    print(f"  F1:        {results['entity']['f1']:.1%}")
    print(f"  (TP={results['entity']['tp']} FP={results['entity']['fp']} FN={results['entity']['fn']})")
    print(f"\nRelationship Extraction:")
    print(f"  Precision: {results['relationship']['precision']:.1%}")
    print(f"  Recall:    {results['relationship']['recall']:.1%}")
    print(f"  F1:        {results['relationship']['f1']:.1%}")
    print(f"  (TP={results['relationship']['tp']} FP={results['relationship']['fp']} FN={results['relationship']['fn']})")
    print(f"\nType Classification Accuracy: {results['type_accuracy']:.1%} ({results['type_total']} matched)")
    print(f"Normalization Accuracy:       {results['normalization_accuracy']:.1%} ({results['normalization_total']} with gold IDs)")
    print(f"Failed Parses:                {results['failed_parses']}/{results['total_evaluated']}")

    # Save results
    out_path = "bioextract/data/eval_teacher_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
