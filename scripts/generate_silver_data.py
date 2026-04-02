"""Generate silver training data using the teacher model (Claude Sonnet ensemble).

Reads BioRED abstracts, runs the teacher on each one, saves output as JSONL
in the same format as biored_train.jsonl so it can be passed directly to
run_progressive() as train_path.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_silver_data.py \
        --input bioextract/data/biored_train.jsonl \
        --output bioextract/data/silver_n100.jsonl \
        --n 100

Then in Colab:
    from bioextract.model.train_progressive import run_progressive
    results = run_progressive(scales=[100], train_path="bioextract/data/silver_n100.jsonl")
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

from bioextract.model.data_utils import diverse_sample, load_data
from bioextract.model.ensemble import extract_ensemble
from bioextract.dictionaries.lookup import DictionaryLookup
from bioextract.normalize import EntityNormalizer


def extract_to_biored(text: str, pmid: str, result) -> dict:
    """Convert ExtractionResult to BioRED JSONL format."""
    entities = [
        {
            "text": e.text,
            "type": e.type,
            "start": e.start,
            "end": e.end,
        }
        for e in result.entities
    ]
    relationships = [
        {
            "subject": r.subject,
            "object": r.object,
            "type": r.type,
            "direction": r.direction,
        }
        for r in result.relationships
    ]
    return {
        "pmid": pmid,
        "text": text,
        "entities": entities,
        "relationships": relationships,
        "source": result.extraction_method,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate silver data with teacher model")
    parser.add_argument("--input", default="bioextract/data/biored_train.jsonl",
                        help="Input JSONL file with abstracts")
    parser.add_argument("--output", default="bioextract/data/silver_n100.jsonl",
                        help="Output JSONL file for silver labels")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of abstracts to label (0 = all)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Self-consistency runs for relationship extraction")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds to sleep between API calls (rate limit)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input}...")
    examples = load_data(args.input)
    print(f"  {len(examples)} examples loaded")

    if args.n and args.n < len(examples):
        print(f"Sampling {args.n} diverse examples...")
        examples = diverse_sample(examples, args.n)
        print(f"  {len(examples)} selected")

    print(f"Initializing teacher pipeline...")
    dictionary = DictionaryLookup()
    normalizer = EntityNormalizer(dictionary) if dictionary.is_available() else None

    output_path = args.output
    written = 0
    failed = 0

    with open(output_path, "w") as out:
        for i, ex in enumerate(examples):
            text = ex["text"]
            pmid = ex.get("pmid", str(i))
            print(f"[{i+1}/{len(examples)}] PMID {pmid}...", end=" ", flush=True)

            try:
                result = extract_ensemble(
                    text,
                    dictionary=dictionary,
                    normalizer=normalizer,
                    n_runs=args.n_runs,
                )
                record = extract_to_biored(text, pmid, result)
                out.write(json.dumps(record) + "\n")
                out.flush()
                n_ent = len(record["entities"])
                n_rel = len(record["relationships"])
                print(f"{n_ent} entities, {n_rel} relationships")
                written += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1

            if i < len(examples) - 1:
                time.sleep(args.sleep)

    print(f"\nDone: {written} written, {failed} failed → {output_path}")
    print(f"\nTo train the student in Colab:")
    print(f"  from bioextract.model.train_progressive import run_progressive")
    print(f"  results = run_progressive(scales=[{args.n}], train_path='{output_path}')")


if __name__ == "__main__":
    main()
