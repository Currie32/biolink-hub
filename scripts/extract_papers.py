"""Extract entities and relationships from ingested papers using BioExtract.

Reads papers from the DB, runs each through the BioExtractor (Claude teacher
or student model), and stores extracted entities, relationships, and evidence.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/extract_papers.py [--limit 50] [--db biolink.db]
"""

import argparse
import html
import json
import sqlite3
import sys
import time

# Add project root to path
sys.path.insert(0, ".")

from bioextract.extract import BioExtractor
from pipeline.build_db import (
    SCHEMA,
    insert_entities,
    insert_evidence_items,
    insert_relationships,
)
from pipeline.sources.base import Entity, Relationship


def main():
    parser = argparse.ArgumentParser(description="Extract from ingested papers")
    parser.add_argument("--db", default="biolink.db", help="Database path")
    parser.add_argument("--limit", type=int, default=50, help="Max papers to process")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    # Track which papers have been processed (regardless of extraction results)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS processed_papers ("
        "  paper_id TEXT PRIMARY KEY, "
        "  processed_at TEXT DEFAULT (datetime('now'))"
        ")"
    )
    conn.commit()

    # Get papers that haven't been processed yet
    rows = conn.execute(
        "SELECT p.id, p.title, p.abstract FROM papers p "
        "WHERE p.abstract IS NOT NULL AND p.abstract != '' "
        "AND p.id NOT IN (SELECT paper_id FROM processed_papers) "
        "LIMIT ?",
        (args.limit,),
    ).fetchall()

    if not rows:
        print("No unprocessed papers found.")
        return

    print(f"Processing {len(rows)} papers...")
    extractor = BioExtractor()

    total_entities = 0
    total_rels = 0
    total_evidence = 0

    for i, row in enumerate(rows):
        paper_id = row["id"]
        abstract = html.unescape(row["abstract"])
        title = html.unescape(row["title"] or "")

        print(f"  [{i+1}/{len(rows)}] {paper_id}: {title[:60]}...", end=" ", flush=True)

        result = extractor.extract(abstract)

        # Mark as processed regardless of results
        conn.execute("INSERT OR IGNORE INTO processed_papers (paper_id) VALUES (?)", (paper_id,))
        conn.commit()

        if not result.entities and not result.relationships:
            print("(no extractions)")
            continue

        # Convert extracted entities to Entity objects
        # Only keep entities that were normalized to a canonical database ID
        entities = []
        entity_id_map = {}  # text -> entity_id
        for e in result.entities:
            if not e.canonical_id:
                continue  # Skip entities not grounded to a real database

            eid = e.canonical_id
            entity_id_map[e.text] = eid

            name = html.unescape(e.canonical_name or e.text)
            text = html.unescape(e.text)

            entities.append(Entity(
                id=eid,
                type=e.type.lower(),
                name=name,
                description="",
                synonyms=[text] if name != text else [],
            ))

        # Convert extracted relationships to Relationship objects
        relationships = []
        evidence_items = []
        for r in result.relationships:
            source_id = entity_id_map.get(r.subject)
            target_id = entity_id_map.get(r.object)
            if not source_id or not target_id:
                continue

            relationships.append(Relationship(
                source_id=source_id,
                target_id=target_id,
                type=r.type,
                source_db="bioextract",
                confidence=r.confidence,
                evidence={
                    "paper_id": paper_id,
                    "direction": r.direction,
                    "predicate": r.predicate,
                    "extraction_method": result.extraction_method,
                },
            ))

            # Build evidence item (relationship_id filled after insert)
            evidence_items.append({
                "paper_id": paper_id,
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": r.type,
                "effect_direction": r.type,  # Use rel type as direction
                "experiment_type": r.context.experiment_type,
                "organism": r.context.organism,
                "cell_type": r.context.cell_type,
                "confidence": r.confidence,
                "extraction_method": result.extraction_method,
            })

        # Insert into DB
        n_ent = insert_entities(conn, entities)
        n_rel = insert_relationships(conn, relationships)
        total_entities += n_ent
        total_rels += n_rel

        # Link evidence to relationships
        for ev in evidence_items:
            rel_row = conn.execute(
                "SELECT id FROM relationships WHERE source_id = ? AND target_id = ? AND type = ? LIMIT 1",
                (ev["source_id"], ev["target_id"], ev["rel_type"]),
            ).fetchone()
            if rel_row:
                ev["relationship_id"] = rel_row["id"]
                # Remove temp fields
                for k in ("source_id", "target_id", "rel_type"):
                    del ev[k]

        valid_evidence = [ev for ev in evidence_items if "relationship_id" in ev]
        n_ev = insert_evidence_items(conn, valid_evidence)
        total_evidence += n_ev

        print(f"{len(result.entities)}E {len(relationships)}R {n_ev}ev")
        time.sleep(8)  # Rate limit: ~5-6 calls/min at 10K output tokens/min

    conn.close()
    print(f"\nDone. Totals: {total_entities} entities, {total_rels} relationships, {total_evidence} evidence items")


if __name__ == "__main__":
    main()
