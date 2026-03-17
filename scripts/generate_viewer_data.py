"""Generate extraction viewer data — runs model on abstracts, saves JSON for the HTML viewer.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_viewer_data.py --test-file bioextract/data/biored_test.jsonl --limit 5
    python scripts/generate_viewer_data.py --test-file bioextract/data/biored_test.jsonl --random --seed 42 --limit 10
    python scripts/generate_viewer_data.py --text "BRCA1 mutations increase breast cancer risk."

Then open scripts/extraction_viewer.html in a browser.
"""

import argparse
import json
import os
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


def extract_to_dict(result) -> dict:
    """Convert ExtractionResult to a JSON-serializable dict."""
    entities = []
    for e in result.entities:
        entities.append({
            "text": e.text,
            "type": e.type,
            "start": e.start,
            "end": e.end,
            "canonical_id": e.canonical_id,
            "canonical_name": e.canonical_name,
            "confidence": round(e.confidence, 3),
        })

    relationships = []
    for r in result.relationships:
        relationships.append({
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "type": r.type,
            "direction": r.direction,
            "negated": r.negated,
            "confidence": round(r.confidence, 3),
            "context": {
                "organism": r.context.organism if r.context else None,
                "cell_type": r.context.cell_type if r.context else None,
                "experiment_type": r.context.experiment_type if r.context else None,
            },
        })

    return {
        "entities": entities,
        "relationships": relationships,
        "method": result.extraction_method,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate extraction viewer data")
    parser.add_argument("--test-file", default="bioextract/data/biored_test.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--text", type=str, default=None,
                        help="Extract from a single text string (no gold comparison)")
    parser.add_argument("--output", default="scripts/viewer_data.json")
    args = parser.parse_args()

    extractor = BioExtractor()
    viewer_data = []

    if args.text:
        # Single text mode — no gold standard
        print(f"Extracting from provided text...")
        result = extractor.extract(args.text, use_ensemble=args.ensemble)
        viewer_data.append({
            "pmid": "custom",
            "text": args.text,
            "gold": None,
            "predicted": extract_to_dict(result),
        })
    else:
        # BioRED test mode — with gold comparison
        if args.random:
            examples = load_gold(args.test_file)
            print(f"Loaded {len(examples)} examples total")
            if args.seed is not None:
                random.seed(args.seed)
            examples = random.sample(examples, min(args.limit, len(examples)))
            print(f"Randomly sampled {len(examples)} examples")
        else:
            examples = load_gold(args.test_file, args.limit)
            print(f"Loaded {len(examples)} examples")

        for i, example in enumerate(examples):
            text = example["text"]
            pmid = example.get("pmid", "?")
            print(f"\n[{i+1}/{len(examples)}] PMID {pmid}...", flush=True)

            result = extractor.extract(text, use_ensemble=args.ensemble)

            viewer_data.append({
                "pmid": pmid,
                "text": text,
                "gold": {
                    "entities": example.get("entities", []),
                    "relationships": example.get("relationships", []),
                },
                "predicted": extract_to_dict(result),
            })

            time.sleep(1)

    with open(args.output, "w") as f:
        json.dump(viewer_data, f, indent=2)

    # Also generate a self-contained HTML file with data embedded
    html_template = os.path.join(os.path.dirname(args.output), "extraction_viewer.html")
    html_output = args.output.replace(".json", ".html")
    if os.path.exists(html_template):
        with open(html_template) as f:
            html = f.read()

        # Inject data by replacing the empty DATA initializer
        data_json = json.dumps(viewer_data)
        html = html.replace("let DATA = [];", f"let DATA = {data_json};")
        with open(html_output, "w") as f:
            f.write(html)
        print(f"\nSaved {len(viewer_data)} abstracts to {args.output}")
        print(f"Self-contained viewer: {html_output}")
        print(f"  open {html_output}")
    else:
        print(f"\nSaved {len(viewer_data)} abstracts to {args.output}")
        print(f"Load {args.output} in scripts/extraction_viewer.html")


if __name__ == "__main__":
    main()
