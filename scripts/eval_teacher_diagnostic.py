"""Diagnostic evaluation of teacher model — identifies specific error patterns.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/eval_teacher_diagnostic.py --test-file bioextract/data/biored_test.jsonl --limit 10
"""

import argparse
import json
import random
import sys
import time
from collections import Counter

sys.path.insert(0, ".")

from bioextract.extract import BioExtractor


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
    """Match predicted entities to gold entities with fuzzy text matching.

    Returns a dict mapping predicted key -> gold key for fuzzy matches.
    Handles plural/singular variants.
    """
    fuzzy_matches = {}
    unmatched_pred = pred_keys - gold_keys
    unmatched_gold = gold_keys - pred_keys

    # Build depluralized lookup for gold
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
    return (
        len(text) > 15 and
        any(w in text.lower() for w in ["substitution", "to a ", "valine", "methionine",
                                         "adenosine, guanosine", "alanine", "isoleucine",
                                         "leucine", "proline", "threonine", "serine"])
    )


def load_gold(path: str, limit: int | None = None) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
                if limit and len(examples) >= limit:
                    break
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="bioextract/data/biored_test.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--ensemble", action="store_true",
                        help="Use the 5-model ensemble pipeline")
    parser.add_argument("--skip-verifier", action="store_true",
                        help="Skip the Sonnet verification step in ensemble")
    parser.add_argument("--random", action="store_true",
                        help="Randomly sample abstracts instead of taking first N")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    if args.skip_verifier:
        import os
        os.environ["BIOEXTRACT_SKIP_VERIFIER"] = "1"

    if args.random:
        gold_examples = load_gold(args.test_file)
        print(f"Loaded {len(gold_examples)} examples total")
        if args.seed is not None:
            random.seed(args.seed)
        sample_size = min(args.limit, len(gold_examples))
        gold_examples = random.sample(gold_examples, sample_size)
        print(f"Randomly sampled {len(gold_examples)} examples")
    else:
        gold_examples = load_gold(args.test_file, args.limit)
    extractor = BioExtractor()
    use_ensemble = args.ensemble

    # Collect error patterns
    fp_entities = []        # predicted but not in gold
    fn_entities = []        # in gold but not predicted
    type_mismatches = []    # matched text, wrong type
    norm_failures = []      # matched text, wrong canonical ID
    rel_fps = []            # predicted relationships not in gold
    rel_fns = []            # gold relationships not predicted
    pred_type_counts = Counter()
    gold_type_counts = Counter()

    for i, example in enumerate(gold_examples):
        text = example["text"]
        pmid = example.get("pmid", "?")
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(gold_examples)}] PMID {pmid}")
        print(f"{'='*70}")

        result = extractor.extract(text, use_ensemble=use_ensemble)

        if not result.entities:
            print("  FAILED TO EXTRACT")
            continue

        # --- Entity analysis ---
        gold_ents = {}
        for e in example["entities"]:
            key = normalize_text(e["text"])
            # Skip entities we intentionally don't extract
            if key in SKIP_GOLD_ENTITIES:
                continue
            # Skip long variant descriptions
            if e.get("type") == "VARIANT" and _is_variant_description(e["text"]):
                continue
            gold_ents[key] = e
            gold_type_counts[e["type"]] += 1

        pred_ents = {}
        for e in result.entities:
            key = normalize_text(e.text)
            pred_ents[key] = e
            pred_type_counts[e.type.upper()] += 1

        gold_keys = set(gold_ents.keys())
        pred_keys = set(pred_ents.keys())

        matched = gold_keys & pred_keys

        # Fuzzy matching for plural/singular variants
        fuzzy = _fuzzy_entity_match(pred_keys, gold_keys)
        matched = matched | set(fuzzy.values())  # Add gold keys matched fuzzily
        # Map fuzzy-matched pred entities to their gold counterparts for type/norm checks
        for pk, gk in fuzzy.items():
            pred_ents[gk] = pred_ents[pk]  # alias so type checks work

        fps = pred_keys - gold_keys - set(fuzzy.keys())
        fns = gold_keys - matched

        fuzzy_str = f" (+{len(fuzzy)} fuzzy)" if fuzzy else ""
        print(f"\n  ENTITIES: {len(matched)} matched{fuzzy_str}, {len(fps)} FP, {len(fns)} FN")

        if fps:
            print(f"\n  False Positives (predicted, not in gold):")
            for key in sorted(fps)[:10]:
                e = pred_ents[key]
                fp_entities.append({"text": e.text, "type": e.type, "pmid": pmid})
                print(f"    - [{e.type}] \"{e.text}\"")

        if fns:
            print(f"\n  False Negatives (in gold, not predicted):")
            for key in sorted(fns)[:10]:
                e = gold_ents[key]
                fn_entities.append({"text": e["text"], "type": e["type"], "pmid": pmid})
                print(f"    - [{e['type']}] \"{e['text']}\"")

        # Type mismatches on matched entities
        type_errors = []
        for key in matched:
            gold_type = gold_ents[key]["type"].upper()
            pred_type = pred_ents[key].type.upper()
            if (gold_type != pred_type
                    and not ({gold_type, pred_type} <= {"DRUG", "CHEMICAL"})
                    and not ({gold_type, pred_type} <= {"GENE", "PROTEIN"})
                    and not ({gold_type, pred_type} <= {"DISEASE", "PHENOTYPE"})):
                type_errors.append((pred_ents[key].text, gold_type, pred_type))
                type_mismatches.append({"text": pred_ents[key].text, "gold": gold_type, "pred": pred_type})

        if type_errors:
            print(f"\n  Type Mismatches ({len(type_errors)}):")
            for text, gold_t, pred_t in type_errors[:10]:
                print(f"    - \"{text}\": gold={gold_t}, pred={pred_t}")

        # Normalization check on matched entities
        # Cross-ontology (MeSH -> DOID/HPO/ChEBI) counts as grounded, not a failure
        norm_errors = []
        for key in matched:
            gold_id = gold_ents[key].get("identifier", "")
            pred_id = pred_ents[key].canonical_id or ""
            # Skip variant identifiers (rs*, p|*, c|*, g|*) — we don't have dbSNP/ClinVar
            if (gold_id and not gold_id.startswith("-")
                    and not gold_id.startswith("p|") and not gold_id.startswith("c|")
                    and not gold_id.startswith("g|") and not gold_id.startswith("r|")
                    and not gold_id.startswith("rs")):
                if not pred_id:
                    norm_errors.append((pred_ents[key].text, gold_id, pred_id))
                    norm_failures.append({"text": pred_ents[key].text, "gold_id": gold_id, "pred_id": pred_id})

        if norm_errors:
            print(f"\n  Normalization Failures ({len(norm_errors)}):")
            for text, gid, pid in norm_errors[:10]:
                print(f"    - \"{text}\": gold={gid}, pred={pid or 'NONE'}")

        # --- Relationship analysis ---
        # Build canonical ID mapping for both gold and predicted entities
        # This lets us match "LQTS" and "long QT syndrome" as the same entity
        # when both resolve to the same canonical ID
        def _entity_key(text_key, ents_dict, is_pred=False):
            """Get canonical key for an entity — canonical_id if available, else text."""
            if text_key in ents_dict:
                e = ents_dict[text_key]
                if is_pred:
                    cid = e.canonical_id
                else:
                    cid = e.get("identifier", "")
                if cid and not cid.startswith("-"):
                    return cid
            return text_key

        # Also build a merged canonical map from predicted entities
        # so "lqts" and "long qt syndrome" both map to the same canonical ID
        pred_canonical_map = {}  # text -> canonical_id
        for key, e in pred_ents.items():
            if e.canonical_id:
                pred_canonical_map[key] = e.canonical_id

        # Build cross-ontology bridge: when pred and gold assign different IDs
        # to the same text entity, prefer the gold ID for consistency
        pred_id_to_gold_id = {}
        for key in matched:
            pred_id = pred_ents[key].canonical_id if key in pred_ents else None
            gold_id = gold_ents[key].get("identifier", "") if key in gold_ents else ""
            if pred_id and gold_id and not gold_id.startswith("-"):
                if pred_id != gold_id:
                    pred_id_to_gold_id[pred_id] = gold_id

        def _normalize_rel_entity(text_key):
            """Normalize a relationship entity to canonical form.

            Bridges cross-ontology IDs so relationships match even when
            pred uses HPO and gold uses MeSH for the same concept.
            """
            if text_key in pred_canonical_map:
                pred_id = pred_canonical_map[text_key]
                if pred_id in pred_id_to_gold_id:
                    return pred_id_to_gold_id[pred_id]
                return pred_id
            if text_key in gold_ents:
                gid = gold_ents[text_key].get("identifier", "")
                if gid and not gid.startswith("-"):
                    return gid
            return text_key

        gold_entity_texts = set(gold_ents.keys())
        gold_pairs = set()
        gold_rels_detail = {}
        for r in example["relationships"]:
            subj = normalize_text(r["subject"])
            obj = normalize_text(r["object"])
            # Skip relationships involving filtered-out entities
            if subj not in gold_entity_texts or obj not in gold_entity_texts:
                continue
            # Use canonical form for matching
            subj_c = _normalize_rel_entity(subj)
            obj_c = _normalize_rel_entity(obj)
            pair = (min(subj_c, obj_c), max(subj_c, obj_c))
            gold_pairs.add(pair)
            gold_rels_detail[pair] = r["type"]

        pred_pairs = set()
        pred_rels_detail = {}
        for r in result.relationships:
            subj = normalize_text(r.subject)
            obj = normalize_text(r.object)
            subj_c = _normalize_rel_entity(subj)
            obj_c = _normalize_rel_entity(obj)
            pair = (min(subj_c, obj_c), max(subj_c, obj_c))
            pred_pairs.add(pair)
            pred_rels_detail[pair] = r.type

        rel_matched = gold_pairs & pred_pairs
        rel_fp = pred_pairs - gold_pairs
        rel_fn = gold_pairs - pred_pairs

        # Build reverse lookup: canonical_id -> text for readable output
        id_to_text = {}
        for key, e in pred_ents.items():
            if e.canonical_id:
                id_to_text[e.canonical_id] = e.text
        for key, e in gold_ents.items():
            gid = e.get("identifier", "")
            if gid and not gid.startswith("-"):
                id_to_text[gid] = e["text"]

        def _readable_pair(pair):
            """Make a canonical pair human-readable."""
            a_text = id_to_text.get(pair[0], pair[0])
            b_text = id_to_text.get(pair[1], pair[1])
            return f"\"{a_text}\" ({pair[0]}) <-> \"{b_text}\" ({pair[1]})"

        print(f"\n  RELATIONSHIPS: {len(rel_matched)} matched, {len(rel_fp)} FP, {len(rel_fn)} FN")

        if rel_fp:
            print(f"\n  Relationship FPs (predicted, not in gold):")
            for pair in sorted(rel_fp)[:10]:
                rtype = pred_rels_detail.get(pair, "?")
                rel_fps.append({"pair": pair, "type": rtype, "pmid": pmid})
                print(f"    - {_readable_pair(pair)} [{rtype}]")

        if rel_fn:
            print(f"\n  Relationship FNs (in gold, not predicted):")
            for pair in sorted(rel_fn)[:10]:
                rtype = gold_rels_detail.get(pair, "?")
                rel_fns.append({"pair": pair, "type": rtype, "pmid": pmid})
                print(f"    - {_readable_pair(pair)} [{rtype}]")

        time.sleep(1)

    # --- Summary ---
    print(f"\n\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")

    print(f"\nEntity FP patterns (what the teacher over-extracts):")
    fp_type_counts = Counter(e["type"] for e in fp_entities)
    for t, c in fp_type_counts.most_common(10):
        print(f"  {t}: {c}")

    print(f"\nEntity FN patterns (what the teacher misses):")
    fn_type_counts = Counter(e["type"] for e in fn_entities)
    for t, c in fn_type_counts.most_common(10):
        print(f"  {t}: {c}")

    print(f"\nType confusion matrix (top errors):")
    confusion = Counter((m["gold"], m["pred"]) for m in type_mismatches)
    for (g, p), c in confusion.most_common(10):
        print(f"  {g} -> {p}: {c}")

    print(f"\nNormalization failures: {len(norm_failures)}")
    no_pred_id = sum(1 for n in norm_failures if not n["pred_id"])
    wrong_pred_id = len(norm_failures) - no_pred_id
    print(f"  No predicted ID: {no_pred_id}")
    print(f"  Wrong predicted ID: {wrong_pred_id}")
    if norm_failures[:5]:
        print(f"  Examples:")
        for n in norm_failures[:10]:
            print(f"    \"{n['text']}\": gold={n['gold_id']}, pred={n['pred_id'] or 'NONE'}")

    print(f"\nPredicted type distribution:")
    for t, c in pred_type_counts.most_common():
        print(f"  {t}: {c}")

    print(f"\nGold type distribution:")
    for t, c in gold_type_counts.most_common():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
