"""Normalization-focused diagnostic — shows exactly how each entity normalizes.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/eval_normalization_diagnostic.py --test-file bioextract/data/biored_test.jsonl --limit 5
"""

import argparse
import json
import sys
import time

sys.path.insert(0, ".")

from bioextract.extract import BioExtractor
from bioextract.normalize import EntityNormalizer, _clean_text, COMMON_ABBREVIATIONS, AMBIGUOUS_TERMS
from bioextract.dictionaries.lookup import DictionaryLookup


def normalize_text(t: str) -> str:
    return t.lower().strip()


SKIP_GOLD_ENTITIES = {"patient", "patients", "families", "subjects", "human", "humans"}


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
    parser.add_argument("--entity-only", action="store_true",
                        help="Skip extraction, just test normalization on gold entities")
    args = parser.parse_args()

    gold_examples = load_gold(args.test_file, args.limit)
    dictionary = DictionaryLookup()
    normalizer = EntityNormalizer(dictionary)

    if not args.entity_only:
        extractor = BioExtractor()

    norm_hits = 0
    norm_misses = 0
    norm_wrong = 0
    cross_ontology_matches = 0

    for i, example in enumerate(gold_examples):
        text = example["text"]
        pmid = example.get("pmid", "?")
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(gold_examples)}] PMID {pmid}")
        print(f"{'='*70}")

        if not args.entity_only:
            result = extractor.extract(text)
            if not result.entities:
                print("  EXTRACTION FAILED")
                continue

            # Build pred entity lookup
            pred_ents = {}
            for e in result.entities:
                pred_ents[normalize_text(e.text)] = e

        # Check normalization for each gold entity
        gold_ents = {}
        for e in example["entities"]:
            key = normalize_text(e["text"])
            if key in SKIP_GOLD_ENTITIES:
                continue
            gold_ents[key] = e

        print(f"\n  NORMALIZATION DETAILS ({len(gold_ents)} gold entities):")

        for key, gold_e in sorted(gold_ents.items()):
            gold_id = gold_e.get("identifier", "")
            gold_type = gold_e["type"]
            entity_text = gold_e["text"]

            # Skip variant identifiers
            if (gold_id.startswith("p|") or gold_id.startswith("c|") or
                gold_id.startswith("g|") or gold_id.startswith("r|") or
                gold_id.startswith("rs") or gold_id.startswith("-")):
                continue

            if not args.entity_only and key in pred_ents:
                pred_id = pred_ents[key].canonical_id or "NONE"
                pred_name = pred_ents[key].canonical_name or ""

                if gold_id and pred_id != "NONE":
                    if pred_id == gold_id:
                        status = "EXACT"
                        norm_hits += 1
                    else:
                        # Check if it's a cross-ontology match (both valid, different ontology)
                        gold_prefix = gold_id.split(":")[0] if ":" in gold_id else gold_id[0]
                        pred_prefix = pred_id.split(":")[0] if ":" in pred_id else pred_id[0]
                        if gold_prefix != pred_prefix:
                            status = "XREF"
                            cross_ontology_matches += 1
                            norm_hits += 1
                        else:
                            status = "WRONG"
                            norm_wrong += 1
                elif gold_id and pred_id == "NONE":
                    status = "MISS"
                    norm_misses += 1
                elif not gold_id:
                    status = "N/A"
                    continue
                else:
                    status = "OK"
                    norm_hits += 1

                marker = {"EXACT": "+", "XREF": "~", "MISS": "X", "WRONG": "!!", "OK": "+", "N/A": "-"}[status]
                detail = f"gold={gold_id}, pred={pred_id}"
                if status == "XREF":
                    detail += f" (cross-ontology: {pred_name})"
                elif status == "MISS":
                    # Show why it missed — try looking up manually
                    matches = dictionary.exact_match(entity_text, gold_type)
                    if matches:
                        detail += f" [exact match available: {matches[0].canonical_id}]"
                    else:
                        matches = dictionary.search(entity_text, gold_type, limit=3)
                        if matches:
                            detail += f" [FTS candidates: {', '.join(m.canonical_id for m in matches)}]"
                        else:
                            detail += " [NOT IN DICTIONARY]"
                elif status == "WRONG":
                    detail += f" (pred_name={pred_name})"

                print(f"    [{marker}] [{gold_type}] \"{entity_text}\" — {detail}")

            elif args.entity_only:
                # Test normalization directly on gold entity text
                from bioextract.schema import ExtractedEntity
                test_entity = ExtractedEntity(
                    text=entity_text,
                    type=gold_type,
                    start=gold_e.get("start", 0),
                    end=gold_e.get("end", 0),
                )
                normalized = normalizer.normalize_entity(test_entity, context_text=text)
                pred_id = normalized.canonical_id or "NONE"
                pred_name = normalized.canonical_name or ""

                if gold_id and pred_id != "NONE":
                    gold_prefix = gold_id.split(":")[0] if ":" in gold_id else gold_id[0]
                    pred_prefix = pred_id.split(":")[0] if ":" in pred_id else pred_id[0]
                    if pred_id == gold_id:
                        status = "EXACT"
                        norm_hits += 1
                    elif gold_prefix != pred_prefix:
                        status = "XREF"
                        cross_ontology_matches += 1
                        norm_hits += 1
                    else:
                        status = "WRONG"
                        norm_wrong += 1
                elif gold_id and pred_id == "NONE":
                    status = "MISS"
                    norm_misses += 1
                else:
                    continue

                marker = {"EXACT": "+", "XREF": "~", "MISS": "X", "WRONG": "!!"}[status]
                detail = f"gold={gold_id}, pred={pred_id}"
                if status == "XREF":
                    detail += f" ({pred_name})"
                elif status == "MISS":
                    matches = dictionary.exact_match(entity_text, None)
                    if matches:
                        detail += f" [untyped match: {matches[0].canonical_id} ({matches[0].entity_type})]"
                    else:
                        matches = dictionary.search(entity_text, None, limit=3)
                        if matches:
                            detail += f" [FTS: {', '.join(f'{m.canonical_id}({m.entity_type})' for m in matches[:3])}]"
                        else:
                            detail += " [NOT IN DICTIONARY]"
                elif status == "WRONG":
                    detail += f" (pred_name={pred_name})"

                print(f"    [{marker}] [{gold_type}] \"{entity_text}\" — {detail}")

            else:
                print(f"    [-] [{gold_type}] \"{entity_text}\" — NOT EXTRACTED")

        if not args.entity_only:
            # Show relationship cross-ontology issues
            pred_canonical = {normalize_text(e.text): e.canonical_id for e in result.entities if e.canonical_id}
            xref_issues = []
            for key in set(pred_ents.keys()) & set(gold_ents.keys()):
                pred_id = pred_ents[key].canonical_id
                gold_id = gold_ents[key].get("identifier", "")
                if pred_id and gold_id and not gold_id.startswith("-") and pred_id != gold_id:
                    xref_issues.append((gold_ents[key]["text"], gold_id, pred_id))

            if xref_issues:
                print(f"\n  CROSS-ONTOLOGY ID MISMATCHES ({len(xref_issues)}):")
                for text, gid, pid in xref_issues:
                    print(f"    \"{text}\": gold={gid}, pred={pid}")

        time.sleep(1)

    # Summary
    total = norm_hits + norm_misses + norm_wrong
    print(f"\n\n{'='*70}")
    print("NORMALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total evaluated: {total}")
    print(f"  Correct:         {norm_hits} ({norm_hits/total:.1%})" if total else "")
    print(f"    - Exact match: {norm_hits - cross_ontology_matches}")
    print(f"    - Cross-ontology: {cross_ontology_matches}")
    print(f"  Missed (no ID):  {norm_misses} ({norm_misses/total:.1%})" if total else "")
    print(f"  Wrong ID:        {norm_wrong} ({norm_wrong/total:.1%})" if total else "")


if __name__ == "__main__":
    main()
