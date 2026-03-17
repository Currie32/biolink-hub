"""Dictionary lookup for entity normalization.

Provides exact match, FTS search, and fuzzy matching against the unified dictionary index.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

INDEX_PATH = Path(__file__).parent / "data" / "dictionary.db"

# Map extracted types to dictionary types (dictionary uses uppercase)
TYPE_ALIASES = {
    "drug": "CHEMICAL",
    "chemical": "CHEMICAL",
    "gene": "GENE",
    "protein": "GENE",
    "disease": "DISEASE",
    "phenotype": "PHENOTYPE",
    "pathway": "PATHWAY",
    "cell_type": "CELL_TYPE",
    "tissue": "TISSUE",
    "organism": "ORGANISM",
    "biological_process": "BIOLOGICAL_PROCESS",
    "molecular_function": "MOLECULAR_FUNCTION",
    "anatomical_structure": "ANATOMICAL_STRUCTURE",
    "variant": "VARIANT",
}


def _normalize_type(entity_type: str | None) -> str | None:
    """Normalize entity type for dictionary matching."""
    if entity_type is None:
        return None
    return TYPE_ALIASES.get(entity_type.lower(), entity_type.upper())


def _normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip hyphens, collapse whitespace."""
    return re.sub(r'[\s-]+', ' ', text.lower()).strip()


def _word_variants(text: str) -> list[str]:
    """Generate word-order and hyphen variants for a term.

    "alpha-synuclein" -> ["alpha-synuclein", "alpha synuclein", "synuclein alpha"]
    "beta-1 adrenoceptor" -> [..., "adrenoceptor beta 1"]
    "mu opioid receptor" -> [..., "opioid receptor mu", "opioid receptor mu 1"]
    """
    variants = [text]
    # Replace hyphens with spaces
    dehyphenated = text.replace("-", " ")
    if dehyphenated != text:
        variants.append(dehyphenated)
    # Reverse word order for 2-word terms
    words = dehyphenated.split()
    if len(words) == 2:
        variants.append(f"{words[1]} {words[0]}")
        variants.append(f"{words[1]}-{words[0]}")
    # For 3+ word terms: move first word(s) to end
    # "beta 1 adrenoceptor" -> "adrenoceptor beta 1"
    # "mu opioid receptor" -> "opioid receptor mu"
    if len(words) >= 3:
        # Move first word to end
        variants.append(" ".join(words[1:] + words[:1]))
        # Move first word to end + append "1" (handles "mu opioid receptor" -> "opioid receptor mu 1")
        variants.append(" ".join(words[1:] + words[:1]) + " 1")
        # Move first two words to end
        if len(words) >= 3:
            variants.append(" ".join(words[2:] + words[:2]))
    return variants


@dataclass
class DictMatch:
    """A dictionary match result."""
    canonical_id: str
    name: str
    entity_type: str
    source_db: str
    match_type: str  # exact, synonym, fts, fuzzy
    score: float  # 1.0 for exact, lower for fuzzy


class DictionaryLookup:
    """Lookup interface for the biomedical dictionary index."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or INDEX_PATH
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"Dictionary index not found at {self.db_path}. "
                    f"Run 'bioextract dictionaries download && bioextract dictionaries index' first."
                )
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def exact_match(self, text: str, entity_type: str | None = None) -> list[DictMatch]:
        """Find exact name matches (case-insensitive).

        Also tries hyphen/word-order variants: "alpha-synuclein" matches
        "synuclein alpha" and vice versa.
        """
        conn = self._get_conn()
        type_clause = ""
        type_params: list = []
        normalized_type = _normalize_type(entity_type)
        if normalized_type:
            type_clause = " AND entity_type = ?"
            type_params = [normalized_type]

        results: list[DictMatch] = []
        seen_ids: set[str] = set()

        for variant in _word_variants(text):
            # Check term names
            params = [variant] + type_params
            rows = conn.execute(
                f"SELECT * FROM terms WHERE name = ? COLLATE NOCASE{type_clause}",
                params,
            ).fetchall()

            for r in rows:
                if r["canonical_id"] not in seen_ids:
                    seen_ids.add(r["canonical_id"])
                    is_original = (variant == text)
                    results.append(DictMatch(
                        canonical_id=r["canonical_id"],
                        name=r["name"],
                        entity_type=r["entity_type"],
                        source_db=r["source_db"],
                        match_type="exact",
                        score=1.0 if is_original else 0.95,
                    ))

            # Also check synonyms
            syn_type_clause = type_clause.replace("entity_type", "t.entity_type")
            syn_rows = conn.execute(
                "SELECT s.canonical_id, s.synonym, t.name, t.entity_type, t.source_db "
                "FROM synonyms s JOIN terms t ON t.canonical_id = s.canonical_id "
                f"WHERE s.synonym = ? COLLATE NOCASE{syn_type_clause}",
                params,
            ).fetchall()

            for r in syn_rows:
                if r["canonical_id"] not in seen_ids:
                    seen_ids.add(r["canonical_id"])
                    is_original = (variant == text)
                    results.append(DictMatch(
                        canonical_id=r["canonical_id"],
                        name=r["name"],
                        entity_type=r["entity_type"],
                        source_db=r["source_db"],
                        match_type="synonym",
                        score=0.95 if is_original else 0.9,
                    ))

        return results

    def search(self, text: str, entity_type: str | None = None, limit: int = 10) -> list[DictMatch]:
        """Full-text search across names and synonyms.

        Tries phrase match first, then individual words (handles hyphenated
        terms and word reordering).
        """
        conn = self._get_conn()

        type_filter = ""
        type_params: list = []
        normalized_type = _normalize_type(entity_type)
        if normalized_type:
            type_filter = " AND entity_type = ?"
            type_params = [normalized_type]

        # Strategy 1: phrase match (exact string)
        query = text.replace('"', '""')
        fts_query = f'"{query}"'
        results = self._fts_query(conn, fts_query, type_filter, type_params, limit)
        if results:
            return results

        # Strategy 2: phrase match on dehyphenated/reordered variants
        for variant in _word_variants(text)[1:]:  # skip original, already tried
            vquery = variant.replace('"', '""')
            results = self._fts_query(conn, f'"{vquery}"', type_filter, type_params, limit)
            if results:
                return results

        # Strategy 3: individual word match (all words must appear)
        words = re.split(r'[\s\-]+', text)
        if len(words) > 1:
            # Each word quoted separately, AND-ed together by FTS5 default
            word_query = " ".join(f'"{w}"' for w in words if len(w) > 1)
            results = self._fts_query(conn, word_query, type_filter, type_params, limit)
            if results:
                return results

        # Strategy 4: LIKE fallback (searches terms + synonyms)
        return self._like_search(text, entity_type, limit)

    def _fts_query(
        self,
        conn: sqlite3.Connection,
        fts_query: str,
        type_filter: str,
        type_params: list,
        limit: int,
    ) -> list[DictMatch]:
        """Execute an FTS5 query and return results."""
        params = [fts_query] + type_params + [limit]
        try:
            rows = conn.execute(
                f"SELECT canonical_id, name, entity_type, rank "
                f"FROM term_search WHERE term_search MATCH ?{type_filter} "
                f"ORDER BY rank LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        results = []
        for r in rows:
            term = conn.execute(
                "SELECT source_db FROM terms WHERE canonical_id = ?",
                (r["canonical_id"],),
            ).fetchone()
            source_db = term["source_db"] if term else "unknown"

            results.append(DictMatch(
                canonical_id=r["canonical_id"],
                name=r["name"],
                entity_type=r["entity_type"],
                source_db=source_db,
                match_type="fts",
                score=min(1.0, max(0.5, 1.0 + r["rank"] / 10)),
            ))

        return results

    def _like_search(self, text: str, entity_type: str | None, limit: int) -> list[DictMatch]:
        """Fallback LIKE search across terms and synonyms."""
        conn = self._get_conn()
        type_clause = ""
        type_params: list = []
        normalized_type = _normalize_type(entity_type)
        if normalized_type:
            type_clause = " AND entity_type = ?"
            type_params = [normalized_type]

        results: list[DictMatch] = []
        seen_ids: set[str] = set()

        # Normalize: replace hyphens with % for flexible matching
        like_pattern = f"%{text.replace('-', '%')}%"

        # Search term names
        rows = conn.execute(
            f"SELECT * FROM terms WHERE name LIKE ? COLLATE NOCASE{type_clause} LIMIT ?",
            [like_pattern] + type_params + [limit],
        ).fetchall()

        for r in rows:
            seen_ids.add(r["canonical_id"])
            results.append(DictMatch(
                canonical_id=r["canonical_id"],
                name=r["name"],
                entity_type=r["entity_type"],
                source_db=r["source_db"],
                match_type="fuzzy",
                score=0.6,
            ))

        # Search synonyms too
        remaining = limit - len(results)
        if remaining > 0:
            syn_type_clause = type_clause.replace("entity_type", "t.entity_type")
            syn_rows = conn.execute(
                "SELECT s.canonical_id, t.name, t.entity_type, t.source_db "
                "FROM synonyms s JOIN terms t ON t.canonical_id = s.canonical_id "
                f"WHERE s.synonym LIKE ? COLLATE NOCASE{syn_type_clause} LIMIT ?",
                [like_pattern] + type_params + [remaining],
            ).fetchall()

            for r in syn_rows:
                if r["canonical_id"] not in seen_ids:
                    seen_ids.add(r["canonical_id"])
                    results.append(DictMatch(
                        canonical_id=r["canonical_id"],
                        name=r["name"],
                        entity_type=r["entity_type"],
                        source_db=r["source_db"],
                        match_type="fuzzy",
                        score=0.55,
                    ))

        return results

    def is_available(self) -> bool:
        """Check if the dictionary index exists and is queryable."""
        if not self.db_path.exists():
            return False
        try:
            conn = self._get_conn()
            conn.execute("SELECT COUNT(*) FROM terms").fetchone()
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        """Return dictionary statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        by_type = conn.execute(
            "SELECT entity_type, COUNT(*) as count FROM terms GROUP BY entity_type"
        ).fetchall()
        by_source = conn.execute(
            "SELECT source_db, COUNT(*) as count FROM terms GROUP BY source_db"
        ).fetchall()
        return {
            "total_terms": total,
            "by_type": {r["entity_type"]: r["count"] for r in by_type},
            "by_source": {r["source_db"]: r["count"] for r in by_source},
        }
