"""Populate the annotation queue with abstracts pre-labeled by the student model.

Reads abstracts from a JSONL source, runs the student model (or teacher if
student is unavailable) for pre-labeling, and inserts into the annotation DB.

Usage:
    # From BioRED test set (20 abstracts, student pre-labels)
    python scripts/populate_annotation_queue.py \
        --source bioextract/data/biored_test.jsonl \
        --n 20

    # From silver data
    python scripts/populate_annotation_queue.py \
        --source bioextract/data/silver_n100.jsonl

    # Skip PMIDs already in queue
    python scripts/populate_annotation_queue.py \
        --source bioextract/data/biored_test.jsonl \
        --skip-existing

    # Use full teacher ensemble (slower, better pre-labels)
    python scripts/populate_annotation_queue.py \
        --source bioextract/data/biored_test.jsonl \
        --ensemble
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioextract.extract import BioExtractor
from bioextract.model.data_utils import diverse_sample, load_data

DB_PATH = Path(__file__).parent.parent / "biolink.db"


def get_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS annotation_abstracts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            pmid       TEXT,
            text       TEXT NOT NULL,
            source     TEXT NOT NULL,
            prelabels  TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS annotations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            abstract_id     INTEGER NOT NULL REFERENCES annotation_abstracts(id),
            annotator_name  TEXT NOT NULL,
            entities        TEXT NOT NULL,
            relationships   TEXT NOT NULL,
            phase           TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );
    """)
    conn.commit()


def result_to_prelabels(result) -> dict:
    """Convert ExtractionResult to prelabels dict for storage."""
    entities = [
        {"text": e.text, "type": e.type, "start": e.start, "end": e.end,
         "confidence": round(e.confidence, 3)}
        for e in result.entities
    ]
    relationships = [
        {"subject": r.subject, "object": r.object, "type": r.type,
         "direction": r.direction, "confidence": round(r.confidence, 3)}
        for r in result.relationships
    ]
    return {"entities": entities, "relationships": relationships,
            "method": result.extraction_method}


def main():
    parser = argparse.ArgumentParser(description="Populate annotation queue with student pre-labels")
    parser.add_argument("--source", required=True,
                        help="Input JSONL file (BioRED format or silver_nXXX.jsonl)")
    parser.add_argument("--n", type=int, default=0,
                        help="Max abstracts to add (0 = all, uses diverse_sample)")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite DB path")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip PMIDs already in the annotation queue")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use full teacher ensemble instead of student (slower)")
    args = parser.parse_args()

    db_path = Path(args.db)
    examples = load_data(args.source)
    print(f"Loaded {len(examples)} examples from {args.source}")

    if args.n and args.n < len(examples):
        examples = diverse_sample(examples, args.n)
        print(f"Sampled {len(examples)} diverse examples")

    conn = get_db(db_path)
    ensure_tables(conn)

    # Collect existing PMIDs to skip
    existing_pmids: set[str] = set()
    if args.skip_existing:
        rows = conn.execute(
            "SELECT pmid FROM annotation_abstracts WHERE pmid IS NOT NULL"
        ).fetchall()
        existing_pmids = {r["pmid"] for r in rows}
        print(f"{len(existing_pmids)} PMIDs already in queue — will skip")

    print("Initializing extractor (student if available, else teacher)...")
    extractor = BioExtractor()
    model_label = "teacher" if extractor._use_teacher else "student"
    print(f"  Using: {model_label}")

    now = datetime.now(timezone.utc).isoformat()
    added = skipped = failed = 0

    for i, ex in enumerate(examples):
        pmid = str(ex.get("pmid", ""))
        text = ex["text"]

        if pmid and args.skip_existing and pmid in existing_pmids:
            skipped += 1
            continue

        print(f"[{i+1}/{len(examples)}] PMID {pmid or '(no pmid)'}...", end=" ", flush=True)

        try:
            result = extractor.extract(text, use_ensemble=args.ensemble)
            prelabels = result_to_prelabels(result)
            source = result.extraction_method or model_label
        except Exception as e:
            print(f"extraction failed: {e}")
            prelabels = {"entities": [], "relationships": [], "method": "none"}
            source = "none"
            failed += 1

        conn.execute(
            "INSERT INTO annotation_abstracts (pmid, text, source, prelabels, created_at) "
            "VALUES (?,?,?,?,?)",
            (pmid or None, text, source, json.dumps(prelabels), now),
        )
        conn.commit()
        print(f"{len(prelabels['entities'])} entities, {len(prelabels['relationships'])} relationships")
        added += 1

    conn.close()
    print(f"\nDone: {added} added, {skipped} skipped, {failed} failed")
    print("Open /annotate in BioLink Hub to start annotating.")


if __name__ == "__main__":
    main()
