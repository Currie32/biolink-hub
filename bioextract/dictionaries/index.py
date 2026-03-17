"""Build a unified SQLite lookup index from downloaded dictionary TSV files.

The index supports:
- Exact name/synonym lookup
- FTS5 full-text search for fuzzy matching
- Trigram-based similarity (via application-level comparison)

Schema:
    terms: canonical_id, name, entity_type, source_db
    synonyms: canonical_id, synonym (one row per synonym)
    term_search: FTS5 virtual table over name + synonyms
"""

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DICT_DIR = Path(__file__).parent / "data"
INDEX_PATH = DICT_DIR / "dictionary.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS terms (
    canonical_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    source_db TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS synonyms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id TEXT NOT NULL REFERENCES terms(canonical_id),
    synonym TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_synonyms_canonical ON synonyms(canonical_id);
CREATE INDEX IF NOT EXISTS idx_synonyms_text ON synonyms(synonym COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_terms_name ON terms(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_terms_type ON terms(entity_type);

CREATE VIRTUAL TABLE IF NOT EXISTS term_search USING fts5(
    canonical_id,
    name,
    all_names,
    entity_type
);
"""


def build_index(dict_dir: Path | None = None, output: Path | None = None) -> Path:
    """Build the unified dictionary index from all TSV files in dict_dir.

    TSV format per line: canonical_id\tname\tentity_type\tsource_db\tsynonyms_json
    """
    dict_dir = dict_dir or DICT_DIR
    output = output or INDEX_PATH

    if output.exists():
        output.unlink()

    conn = sqlite3.connect(str(output))
    conn.executescript(SCHEMA)

    total_terms = 0
    total_synonyms = 0

    for tsv_file in sorted(dict_dir.glob("*.tsv")):
        if tsv_file.stat().st_size == 0:
            logger.info("Skipping empty file: %s", tsv_file.name)
            continue

        logger.info("Indexing %s...", tsv_file.name)
        batch_terms = []
        batch_synonyms = []
        batch_fts = []

        with open(tsv_file) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                canonical_id, name, entity_type, source_db, synonyms_json = parts[:5]

                try:
                    syns = json.loads(synonyms_json)
                except json.JSONDecodeError:
                    syns = []

                batch_terms.append((canonical_id, name, entity_type, source_db))

                for syn in syns:
                    batch_synonyms.append((canonical_id, syn))

                # FTS: combine name + all synonyms for search
                all_names = " | ".join([name] + syns)
                batch_fts.append((canonical_id, name, all_names, entity_type))

        # Batch insert
        conn.executemany(
            "INSERT OR IGNORE INTO terms (canonical_id, name, entity_type, source_db) VALUES (?, ?, ?, ?)",
            batch_terms,
        )
        conn.executemany(
            "INSERT INTO synonyms (canonical_id, synonym) VALUES (?, ?)",
            batch_synonyms,
        )
        conn.executemany(
            "INSERT INTO term_search (canonical_id, name, all_names, entity_type) VALUES (?, ?, ?, ?)",
            batch_fts,
        )
        conn.commit()

        total_terms += len(batch_terms)
        total_synonyms += len(batch_synonyms)
        logger.info("  %s: %d terms, %d synonyms", tsv_file.name, len(batch_terms), len(batch_synonyms))

    conn.close()
    logger.info("Index built: %d terms, %d synonyms → %s", total_terms, total_synonyms, output)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index()
