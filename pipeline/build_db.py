"""Build SQLite database from source data."""

import json
import sqlite3
from pathlib import Path

from .sources.base import Entity, Relationship

DEFAULT_DB_PATH = Path(__file__).parent.parent / "biolink.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    synonyms TEXT,
    external_ids TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES entities(id),
    target_id TEXT NOT NULL REFERENCES entities(id),
    type TEXT NOT NULL,
    source_db TEXT NOT NULL,
    confidence REAL,
    evidence TEXT,
    UNIQUE(source_id, target_id, type, source_db)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);

CREATE VIRTUAL TABLE IF NOT EXISTS entity_search USING fts5(
    entity_id, name, synonyms, description
);

CREATE TABLE IF NOT EXISTS entity_summary (
    entity_id TEXT PRIMARY KEY REFERENCES entities(id),
    related_counts TEXT,
    top_connections TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    journal TEXT,
    year INTEGER,
    abstract TEXT,
    source_url TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS evidence_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_id INTEGER REFERENCES relationships(id),
    paper_id TEXT REFERENCES papers(id),
    sentence TEXT,
    section TEXT,
    effect_direction TEXT,
    experiment_type TEXT,
    organism TEXT,
    cell_type TEXT,
    confidence REAL,
    extraction_method TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_evidence_relationship ON evidence_items(relationship_id);
CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence_items(paper_id);
CREATE INDEX IF NOT EXISTS idx_evidence_direction ON evidence_items(effect_direction);
"""


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create database and tables."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def insert_entities(conn: sqlite3.Connection, entities: list[Entity]) -> int:
    """Insert entities, returning count of rows inserted.

    Uses INSERT OR IGNORE for new entities, then merges non-empty fields
    into existing rows so NLP-extracted entities don't overwrite curated data.
    """
    count = 0
    for e in entities:
        try:
            # Try insert first (new entity)
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, type, name, description, synonyms, external_ids, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    e.id,
                    e.type,
                    e.name,
                    e.description,
                    json.dumps(e.synonyms),
                    json.dumps(e.external_ids),
                    json.dumps(e.metadata),
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                # New row inserted
                conn.execute(
                    "INSERT OR REPLACE INTO entity_search (entity_id, name, synonyms, description) "
                    "VALUES (?, ?, ?, ?)",
                    (e.id, e.name, " ".join(e.synonyms), e.description),
                )
                count += 1
            else:
                # Row exists — merge non-empty fields only
                updates = []
                params = []
                if e.description:
                    updates.append("description = ?")
                    params.append(e.description)
                if e.synonyms:
                    # Merge synonym lists
                    existing = conn.execute("SELECT synonyms FROM entities WHERE id = ?", (e.id,)).fetchone()
                    existing_syns = json.loads(existing[0]) if existing and existing[0] else []
                    merged = list(dict.fromkeys(existing_syns + e.synonyms))
                    updates.append("synonyms = ?")
                    params.append(json.dumps(merged))
                if e.external_ids:
                    existing = conn.execute("SELECT external_ids FROM entities WHERE id = ?", (e.id,)).fetchone()
                    existing_ids = json.loads(existing[0]) if existing and existing[0] else {}
                    existing_ids.update(e.external_ids)
                    updates.append("external_ids = ?")
                    params.append(json.dumps(existing_ids))
                if e.metadata:
                    existing = conn.execute("SELECT metadata FROM entities WHERE id = ?", (e.id,)).fetchone()
                    existing_meta = json.loads(existing[0]) if existing and existing[0] else {}
                    existing_meta.update(e.metadata)
                    updates.append("metadata = ?")
                    params.append(json.dumps(existing_meta))

                if updates:
                    params.append(e.id)
                    conn.execute(
                        f"UPDATE entities SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )
                count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def insert_relationships(conn: sqlite3.Connection, relationships: list[Relationship]) -> int:
    """Insert relationships, returning count of rows inserted."""
    count = 0
    for r in relationships:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO relationships (source_id, target_id, type, source_db, confidence, evidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    r.source_id,
                    r.target_id,
                    r.type,
                    r.source_db,
                    r.confidence,
                    json.dumps(r.evidence),
                ),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def insert_papers(conn: sqlite3.Connection, papers: list) -> int:
    """Insert papers into the papers table.

    Args:
        papers: List of Paper dataclass instances (from pubmed_abstracts source)
            or dicts with keys: pmid/id, title, authors, journal, year, abstract, source_url, metadata.
    """
    count = 0
    for p in papers:
        if hasattr(p, "pmid"):
            # Paper dataclass
            paper_id = f"pmid:{p.pmid}"
            title = p.title
            authors = json.dumps(p.authors)
            journal = p.journal
            year = p.year
            abstract = p.abstract
            source_url = p.source_url
            metadata = json.dumps(p.metadata)
        else:
            # Dict
            paper_id = p.get("id") or f"pmid:{p.get('pmid', '')}"
            title = p.get("title", "")
            authors = json.dumps(p.get("authors", []))
            journal = p.get("journal", "")
            year = p.get("year", 0)
            abstract = p.get("abstract", "")
            source_url = p.get("source_url", "")
            metadata = json.dumps(p.get("metadata", {}))

        try:
            conn.execute(
                "INSERT OR IGNORE INTO papers (id, title, authors, journal, year, abstract, source_url, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (paper_id, title, authors, journal, year, abstract, source_url, metadata),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def insert_evidence_items(conn: sqlite3.Connection, items: list[dict]) -> int:
    """Insert evidence items linking relationships to papers.

    Each item dict should have:
        relationship_id, paper_id, sentence, section, effect_direction,
        experiment_type, organism, cell_type, confidence, extraction_method
    """
    count = 0
    for item in items:
        try:
            conn.execute(
                "INSERT INTO evidence_items "
                "(relationship_id, paper_id, sentence, section, effect_direction, "
                "experiment_type, organism, cell_type, confidence, extraction_method) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.get("relationship_id"),
                    item.get("paper_id"),
                    item.get("sentence"),
                    item.get("section"),
                    item.get("effect_direction"),
                    item.get("experiment_type"),
                    item.get("organism"),
                    item.get("cell_type"),
                    item.get("confidence"),
                    item.get("extraction_method"),
                ),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def link_evidence_to_relationships(conn: sqlite3.Connection) -> int:
    """Link evidence_items to relationships by matching source/target/type.

    Call after both relationships and evidence_items are inserted.
    Updates evidence_items.relationship_id where it's currently NULL.
    """
    cursor = conn.execute("""
        UPDATE evidence_items
        SET relationship_id = (
            SELECT r.id FROM relationships r
            JOIN entities e1 ON r.source_id = e1.id
            JOIN entities e2 ON r.target_id = e2.id
            WHERE (e1.name = evidence_items.sentence OR e1.id = evidence_items.sentence)
            AND (e2.name = evidence_items.section OR e2.id = evidence_items.section)
            LIMIT 1
        )
        WHERE relationship_id IS NULL
    """)
    conn.commit()
    return cursor.rowcount


def build(db_path: Path = DEFAULT_DB_PATH, sources: list | None = None) -> Path:
    """Run the full build pipeline."""
    from .sources.ncbi_gene import NCBIGene
    from .sources.dgidb import DGIdb

    if sources is None:
        sources = [NCBIGene(), DGIdb()]

    # Remove old DB
    if db_path.exists():
        db_path.unlink()

    conn = init_db(db_path)

    total_entities = 0
    total_rels = 0

    for source in sources:
        print(f"Fetching from {source.name}...")
        source.fetch()

        print(f"Parsing {source.name}...")
        entities, relationships = source.parse()

        n_ent = insert_entities(conn, entities)
        n_rel = insert_relationships(conn, relationships)
        total_entities += n_ent
        total_rels += n_rel
        print(f"  {source.name}: {n_ent} entities, {n_rel} relationships")

        # Handle paper sources specially
        if hasattr(source, "get_papers"):
            papers = source.get_papers()
            if papers:
                n_papers = insert_papers(conn, papers)
                print(f"  {source.name}: {n_papers} papers")

    conn.close()
    print(f"\nDone. Total: {total_entities} entities, {total_rels} relationships")
    print(f"Database: {db_path}")
    return db_path
