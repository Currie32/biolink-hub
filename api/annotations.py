"""Annotation API for biomedical entity and relationship labeling."""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Lazy singleton — loaded on first /fetch call
_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        from bioextract.extract import BioExtractor
        _extractor = BioExtractor()
    return _extractor

DB_PATH = Path(__file__).parent.parent / "biolink.db"

router = APIRouter(prefix="/api/annotate")


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
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


class AnnotationSave(BaseModel):
    annotator_name: str
    entities: list = []
    relationships: list = []
    phase: str = "entities"


@router.get("/queue")
def get_queue():
    conn = get_db()
    ensure_tables(conn)
    abstracts = conn.execute(
        "SELECT id, pmid, source, created_at FROM annotation_abstracts ORDER BY id"
    ).fetchall()
    result = []
    for a in abstracts:
        ann_rows = conn.execute(
            "SELECT annotator_name, phase FROM annotations WHERE abstract_id = ?",
            (a["id"],),
        ).fetchall()
        annotators = [{"name": r["annotator_name"], "phase": r["phase"]} for r in ann_rows]
        completed = sum(1 for r in ann_rows if r["phase"] == "completed")
        result.append({
            "id": a["id"],
            "pmid": a["pmid"],
            "source": a["source"],
            "created_at": a["created_at"],
            "annotators": annotators,
            "completed_count": completed,
        })
    conn.close()
    return result


@router.get("/export")
def export_annotations():
    conn = get_db()
    ensure_tables(conn)
    abstracts = conn.execute("SELECT * FROM annotation_abstracts").fetchall()
    lines = []
    for a in abstracts:
        ann_rows = conn.execute(
            "SELECT * FROM annotations WHERE abstract_id = ? AND phase = 'completed'",
            (a["id"],),
        ).fetchall()
        if not ann_rows:
            continue
        base = ann_rows[0]
        merged_ents = [
            {"text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"]}
            for e in json.loads(base["entities"])
            if e.get("status") in ("accepted", "added", "edited")
        ]
        merged_rels = [
            {
                "subject": r["subject"],
                "object": r["object"],
                "type": r["type"],
                "direction": r["direction"],
            }
            for r in json.loads(base["relationships"])
            if r.get("status") in ("accepted", "added", "edited")
        ]
        lines.append(
            json.dumps(
                {"pmid": a["pmid"], "text": a["text"], "entities": merged_ents, "relationships": merged_rels}
            )
        )
    conn.close()
    content = "\n".join(lines) + ("\n" if lines else "")
    return StreamingResponse(
        iter([content]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=annotations_export.jsonl"},
    )


@router.get("/iaa/{abstract_id}")
def get_iaa(abstract_id: int):
    conn = get_db()
    ensure_tables(conn)
    ann_rows = conn.execute(
        "SELECT * FROM annotations WHERE abstract_id = ?", (abstract_id,)
    ).fetchall()
    if len(ann_rows) < 2:
        conn.close()
        return {"message": "Need at least 2 annotators", "entity_agreement": None, "rel_agreement": None}

    annotators = []
    for r in ann_rows:
        a = dict(r)
        a["entities"] = json.loads(a["entities"])
        a["relationships"] = json.loads(a["relationships"])
        annotators.append(a)

    pairs = [
        (annotators[i], annotators[j])
        for i in range(len(annotators))
        for j in range(i + 1, len(annotators))
    ]
    ent_scores, rel_scores = [], []

    for a1, a2 in pairs:
        def bucket(s):
            return "rejected" if s == "rejected" else "accepted"

        e1 = {(e["start"], e["end"]): bucket(e.get("status", "accepted")) for e in a1["entities"]}
        e2 = {(e["start"], e["end"]): bucket(e.get("status", "accepted")) for e in a2["entities"]}
        common_e = set(e1) & set(e2)
        if common_e:
            ent_scores.append(sum(1 for k in common_e if e1[k] == e2[k]) / len(common_e))

        def rel_val(r):
            if r.get("status") == "rejected":
                return ("none", "none")
            return (r.get("type", ""), r.get("direction", ""))

        r1 = {(r["subject"], r["object"]): rel_val(r) for r in a1["relationships"]}
        r2 = {(r["subject"], r["object"]): rel_val(r) for r in a2["relationships"]}
        common_r = set(r1) & set(r2)
        if common_r:
            rel_scores.append(sum(1 for k in common_r if r1[k] == r2[k]) / len(common_r))

    conn.close()
    return {
        "annotator_count": len(annotators),
        "entity_agreement": round(sum(ent_scores) / len(ent_scores), 3) if ent_scores else None,
        "rel_agreement": round(sum(rel_scores) / len(rel_scores), 3) if rel_scores else None,
        "pair_count": len(pairs),
    }


@router.get("/{abstract_id}")
def get_abstract(abstract_id: int, annotator_name: str = Query(None)):
    conn = get_db()
    ensure_tables(conn)
    row = conn.execute(
        "SELECT * FROM annotation_abstracts WHERE id = ?", (abstract_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Abstract not found")
    abstract = dict(row)
    abstract["prelabels"] = json.loads(abstract["prelabels"])
    existing = None
    if annotator_name:
        ann = conn.execute(
            "SELECT * FROM annotations WHERE abstract_id = ? AND annotator_name = ?",
            (abstract_id, annotator_name),
        ).fetchone()
        if ann:
            existing = dict(ann)
            existing["entities"] = json.loads(existing["entities"])
            existing["relationships"] = json.loads(existing["relationships"])
    conn.close()
    return {"abstract": abstract, "existing_annotation": existing}


class FetchRequest(BaseModel):
    keywords: str
    max_results: int = 10


@router.post("/fetch")
def fetch_abstracts(body: FetchRequest):
    """Search PubMed by keyword, run student model, add results to annotation queue."""
    if not body.keywords.strip():
        raise HTTPException(status_code=400, detail="keywords required")

    conn = get_db()
    ensure_tables(conn)

    # Existing PMIDs to avoid duplicates
    existing_pmids = {
        r[0] for r in conn.execute("SELECT pmid FROM annotation_abstracts WHERE pmid IS NOT NULL").fetchall()
    }

    # Search PubMed
    try:
        with httpx.Client(timeout=30) as client:
            search_resp = client.get(
                f"{NCBI_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": body.keywords,
                    "retmax": body.max_results,
                    "sort": "relevance",
                    "retmode": "json",
                },
            )
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            new_pmids = [p for p in pmids if p not in existing_pmids]

            if not new_pmids:
                conn.close()
                msg = (f"All {len(pmids)} results already in queue"
                       if pmids else "No PubMed results found")
                return {"added": 0, "message": msg}

            time.sleep(0.3)

            # Fetch abstracts
            fetch_resp = client.get(
                f"{NCBI_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(new_pmids),
                    "rettype": "xml",
                    "retmode": "xml",
                },
            )
            fetch_resp.raise_for_status()
            papers = _parse_pubmed_xml(fetch_resp.text)
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=502, detail=f"PubMed fetch failed: {e}")

    # Run extraction and insert
    extractor = _get_extractor()
    added = []
    now = datetime.now(timezone.utc).isoformat()

    for paper in papers:
        if not paper.get("abstract") or paper["pmid"] in existing_pmids:
            continue
        try:
            result = extractor.extract(paper["abstract"], use_ensemble=False)
            prelabels = _result_to_prelabels(result)
            source = result.extraction_method or "student"
        except Exception:
            prelabels = {"entities": [], "relationships": [], "method": "none"}
            source = "none"

        row_id = conn.execute(
            "INSERT INTO annotation_abstracts (pmid, text, source, prelabels, created_at) VALUES (?,?,?,?,?)",
            (paper["pmid"], paper["abstract"], source, json.dumps(prelabels), now),
        ).lastrowid
        conn.commit()
        added.append(row_id)

    conn.close()
    return {"added": len(added), "ids": added}


@router.post("/{abstract_id}")
def save_annotation(abstract_id: int, body: AnnotationSave):
    conn = get_db()
    ensure_tables(conn)
    if not conn.execute(
        "SELECT id FROM annotation_abstracts WHERE id = ?", (abstract_id,)
    ).fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Abstract not found")
    now = datetime.now(timezone.utc).isoformat()
    existing = conn.execute(
        "SELECT id FROM annotations WHERE abstract_id = ? AND annotator_name = ?",
        (abstract_id, body.annotator_name),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE annotations SET entities=?, relationships=?, phase=?, updated_at=? WHERE id=?",
            (
                json.dumps(body.entities),
                json.dumps(body.relationships),
                body.phase,
                now,
                existing["id"],
            ),
        )
    else:
        conn.execute(
            "INSERT INTO annotations "
            "(abstract_id, annotator_name, entities, relationships, phase, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                abstract_id,
                body.annotator_name,
                json.dumps(body.entities),
                json.dumps(body.relationships),
                body.phase,
                now,
            ),
        )
    conn.commit()
    conn.close()
    return {"status": "saved"}


def _result_to_prelabels(result) -> dict:
    """Convert ExtractionResult to prelabels dict for storage."""
    entities = [
        {"text": e.text, "type": e.type, "start": e.start, "end": e.end, "confidence": round(e.confidence, 3)}
        for e in result.entities
    ]
    relationships = [
        {"subject": r.subject, "object": r.object, "type": r.type,
         "direction": r.direction, "confidence": round(r.confidence, 3)}
        for r in result.relationships
    ]
    return {"entities": entities, "relationships": relationships, "method": result.extraction_method}


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML response into list of {pmid, abstract} dicts."""
    import re

    def extract_tag(xml, tag):
        m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', xml, re.DOTALL)
        return re.sub(r'<[^>]+>', '', m.group(1)).strip() if m else ""

    papers = []
    for article in xml_text.split("<PubmedArticle>")[1:]:
        pmid = extract_tag(article, "PMID")
        if not pmid:
            continue
        abstract = extract_tag(article, "AbstractText")
        if not abstract:
            # Try structured abstract (multiple AbstractText elements)
            parts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', article, re.DOTALL)
            abstract = " ".join(re.sub(r'<[^>]+>', '', p).strip() for p in parts)
        if abstract:
            papers.append({"pmid": pmid, "abstract": abstract})

    return papers
