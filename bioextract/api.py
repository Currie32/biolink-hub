"""BioExtract FastAPI service.

Standalone extraction API that can be run independently or integrated
with the BioLink Hub pipeline.

Endpoints:
    POST /extract         — Extract from a single text
    POST /extract/batch   — Extract from multiple texts
    GET  /dictionaries/lookup — Dictionary search
    GET  /health          — Service health check
"""

import logging
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .extract import BioExtractor
from .dictionaries.lookup import DictionaryLookup

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BioExtract",
    description="Biomedical entity and relationship extraction service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialized on first request
_extractor: BioExtractor | None = None
_dictionary: DictionaryLookup | None = None


def get_extractor() -> BioExtractor:
    global _extractor
    if _extractor is None:
        _extractor = BioExtractor()
    return _extractor


def get_dictionary() -> DictionaryLookup:
    global _dictionary
    if _dictionary is None:
        _dictionary = DictionaryLookup()
    return _dictionary


# --- Request/Response models ---

class ExtractRequest(BaseModel):
    text: str
    include_raw: bool = False  # Include raw model output


class BatchExtractRequest(BaseModel):
    texts: list[str]
    include_raw: bool = False


class EntityOut(BaseModel):
    text: str
    type: str
    start: int
    end: int
    canonical_id: Optional[str] = None
    canonical_name: Optional[str] = None
    confidence: float


class RelationshipContextOut(BaseModel):
    organism: Optional[str] = None
    cell_type: Optional[str] = None
    experiment_type: Optional[str] = None


class RelationshipOut(BaseModel):
    subject: str
    predicate: str
    object: str
    type: str
    direction: str
    negated: bool
    context: RelationshipContextOut
    confidence: float


class ExtractResponse(BaseModel):
    entities: list[EntityOut]
    relationships: list[RelationshipOut]
    extraction_method: str


# --- Endpoints ---

@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    """Extract entities and relationships from a single text."""
    extractor = get_extractor()
    result = extractor.extract(req.text)

    return ExtractResponse(
        entities=[
            EntityOut(
                text=e.text,
                type=e.type,
                start=e.start,
                end=e.end,
                canonical_id=e.canonical_id,
                canonical_name=e.canonical_name,
                confidence=e.confidence,
            )
            for e in result.entities
        ],
        relationships=[
            RelationshipOut(
                subject=r.subject,
                predicate=r.predicate,
                object=r.object,
                type=r.type,
                direction=r.direction,
                negated=r.negated,
                context=RelationshipContextOut(
                    organism=r.context.organism,
                    cell_type=r.context.cell_type,
                    experiment_type=r.context.experiment_type,
                ),
                confidence=r.confidence,
            )
            for r in result.relationships
        ],
        extraction_method=result.extraction_method,
    )


@app.post("/extract/batch")
def extract_batch(req: BatchExtractRequest):
    """Extract from multiple texts."""
    extractor = get_extractor()
    results = extractor.extract_batch(req.texts)

    return [
        ExtractResponse(
            entities=[
                EntityOut(
                    text=e.text, type=e.type, start=e.start, end=e.end,
                    canonical_id=e.canonical_id, canonical_name=e.canonical_name,
                    confidence=e.confidence,
                )
                for e in r.entities
            ],
            relationships=[
                RelationshipOut(
                    subject=rel.subject, predicate=rel.predicate, object=rel.object,
                    type=rel.type, direction=rel.direction, negated=rel.negated,
                    context=RelationshipContextOut(
                        organism=rel.context.organism, cell_type=rel.context.cell_type,
                        experiment_type=rel.context.experiment_type,
                    ),
                    confidence=rel.confidence,
                )
                for rel in r.relationships
            ],
            extraction_method=r.extraction_method,
        )
        for r in results
    ]


@app.get("/dictionaries/lookup")
def dictionary_lookup(
    q: str = Query(..., min_length=1),
    type: str | None = None,
    limit: int = Query(10, ge=1, le=100),
):
    """Search the biomedical dictionary."""
    dictionary = get_dictionary()
    if not dictionary.is_available():
        return {"error": "Dictionary index not built. Run dictionary download + index first."}

    # Try exact match first
    exact = dictionary.exact_match(q, entity_type=type)
    if exact:
        return {
            "query": q,
            "match_type": "exact",
            "results": [
                {
                    "canonical_id": m.canonical_id,
                    "name": m.name,
                    "entity_type": m.entity_type,
                    "source_db": m.source_db,
                    "score": m.score,
                }
                for m in exact[:limit]
            ],
        }

    # Fall back to search
    results = dictionary.search(q, entity_type=type, limit=limit)
    return {
        "query": q,
        "match_type": "search",
        "results": [
            {
                "canonical_id": m.canonical_id,
                "name": m.name,
                "entity_type": m.entity_type,
                "source_db": m.source_db,
                "score": m.score,
            }
            for m in results
        ],
    }


@app.get("/health")
def health():
    """Service health check."""
    extractor = get_extractor()
    dictionary = get_dictionary()
    return {
        "status": "ok",
        "extractor": extractor.status,
        "dictionary": {
            "available": dictionary.is_available(),
            "stats": dictionary.stats() if dictionary.is_available() else None,
        },
    }
