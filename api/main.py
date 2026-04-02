"""Minimal FastAPI app serving biolink SQLite data."""

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.annotations import router as annotation_router

DB_PATH = Path(__file__).parent.parent / "biolink.db"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"

app = FastAPI(title="BioLink Hub API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(annotation_router)


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row):
    d = dict(row)
    for key in ("synonyms", "external_ids", "metadata", "evidence", "authors"):
        if key in d and d[key]:
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):
    conn = get_db()
    rows = conn.execute(
        "SELECT e.* FROM entity_search s JOIN entities e ON e.id = s.entity_id "
        "WHERE entity_search MATCH ? "
        "AND e.id IN (SELECT source_id FROM relationships UNION SELECT target_id FROM relationships) "
        "ORDER BY rank LIMIT 20",
        (q,),
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/entity/{entity_id:path}")
def get_entity(entity_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
    if not row:
        conn.close()
        return {"error": "not found"}

    entity = row_to_dict(row)

    # Get relationships where this entity is source or target
    rels_out = conn.execute(
        "SELECT r.*, e.name as related_name, e.type as related_type "
        "FROM relationships r JOIN entities e ON e.id = r.target_id "
        "WHERE r.source_id = ?",
        (entity_id,),
    ).fetchall()
    rels_in = conn.execute(
        "SELECT r.*, e.name as related_name, e.type as related_type "
        "FROM relationships r JOIN entities e ON e.id = r.source_id "
        "WHERE r.target_id = ?",
        (entity_id,),
    ).fetchall()

    relationships = []
    for r in rels_out + rels_in:
        rel = row_to_dict(r)
        # Add evidence count for this relationship
        evidence_count = conn.execute(
            "SELECT COUNT(DISTINCT paper_id) FROM evidence_items WHERE relationship_id = ?",
            (rel["id"],),
        ).fetchone()[0]
        rel["evidence_count"] = evidence_count
        relationships.append(rel)

    entity["relationships"] = relationships

    # Get papers that cite this entity (via evidence_items)
    paper_count = conn.execute(
        "SELECT COUNT(DISTINCT ei.paper_id) FROM evidence_items ei "
        "JOIN relationships r ON r.id = ei.relationship_id "
        "WHERE r.source_id = ? OR r.target_id = ?",
        (entity_id, entity_id),
    ).fetchone()[0]
    entity["paper_count"] = paper_count

    conn.close()
    return entity


@app.get("/api/entity/{entity_id:path}/evidence")
def get_entity_evidence(entity_id: str, limit: int = Query(20, ge=1, le=100)):
    """Get evidence items for all relationships involving this entity."""
    conn = get_db()

    rows = conn.execute(
        "SELECT ei.*, p.title as paper_title, p.journal, p.year, p.source_url, "
        "r.source_id, r.target_id, r.type as rel_type, "
        "es.name as source_name, et.name as target_name "
        "FROM evidence_items ei "
        "JOIN relationships r ON r.id = ei.relationship_id "
        "JOIN papers p ON p.id = ei.paper_id "
        "JOIN entities es ON es.id = r.source_id "
        "JOIN entities et ON et.id = r.target_id "
        "WHERE r.source_id = ? OR r.target_id = ? "
        "ORDER BY ei.confidence DESC "
        "LIMIT ?",
        (entity_id, entity_id, limit),
    ).fetchall()

    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/relationship/{rel_id}/evidence")
def get_relationship_evidence(rel_id: int):
    """Get all evidence items for a specific relationship."""
    conn = get_db()

    rows = conn.execute(
        "SELECT ei.*, p.title as paper_title, p.authors, p.journal, p.year, p.source_url "
        "FROM evidence_items ei "
        "JOIN papers p ON p.id = ei.paper_id "
        "WHERE ei.relationship_id = ? "
        "GROUP BY ei.paper_id "
        "ORDER BY ei.confidence DESC",
        (rel_id,),
    ).fetchall()

    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/entities")
def list_entities(type: str = Query(None), limit: int = Query(100, ge=1, le=500)):
    """List entities that have at least one relationship, optionally filtered by type."""
    conn = get_db()
    base = (
        "SELECT e.* FROM entities e "
        "WHERE e.id IN (SELECT source_id FROM relationships UNION SELECT target_id FROM relationships)"
    )
    if type:
        rows = conn.execute(
            base + " AND e.type = ? ORDER BY e.name LIMIT ?",
            (type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            base + " ORDER BY e.name LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/stats")
def stats():
    conn = get_db()
    entities = conn.execute(
        "SELECT e.type, COUNT(*) as count FROM entities e "
        "WHERE e.id IN (SELECT source_id FROM relationships UNION SELECT target_id FROM relationships) "
        "GROUP BY e.type"
    ).fetchall()
    rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

    # Paper and evidence counts (safe even if tables don't exist yet)
    try:
        paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        evidence_count = conn.execute("SELECT COUNT(*) FROM evidence_items").fetchone()[0]
    except sqlite3.OperationalError:
        paper_count = 0
        evidence_count = 0

    conn.close()
    return {
        "entities": {r["type"]: r["count"] for r in entities},
        "total_entities": sum(r["count"] for r in entities),
        "total_relationships": rel_count,
        "total_papers": paper_count,
        "total_evidence": evidence_count,
    }


# Serve frontend static files (if built)
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # Try to serve the exact file, otherwise serve index.html (SPA routing)
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
