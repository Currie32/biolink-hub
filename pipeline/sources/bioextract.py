"""BioExtract integration source.

Sends abstracts to the BioExtract service and converts extraction results
to Entity + Relationship objects with canonical IDs matching the biolink DB.
"""

import json
import logging

import httpx

from .base import Source, Entity, Relationship

logger = logging.getLogger(__name__)

BIOEXTRACT_URL = "http://127.0.0.1:8001"


class BioExtractSource(Source):
    """Extract entities and relationships from paper abstracts via BioExtract."""

    name = "bioextract"

    def __init__(
        self,
        abstracts: list[dict] | None = None,
        bioextract_url: str = BIOEXTRACT_URL,
    ):
        """
        Args:
            abstracts: List of dicts with keys: paper_id, abstract, title.
            bioextract_url: URL of the BioExtract service.
        """
        self.abstracts = abstracts or []
        self.bioextract_url = bioextract_url
        self._results: list[dict] = []

    def fetch(self) -> None:
        """Send abstracts to BioExtract for extraction."""
        self._results = []

        if not self.abstracts:
            logger.info("No abstracts to process")
            return

        with httpx.Client(timeout=120) as client:
            # Check health first
            try:
                health = client.get(f"{self.bioextract_url}/health")
                health.raise_for_status()
                logger.info("BioExtract service healthy: %s", health.json().get("status"))
            except Exception as e:
                logger.error("BioExtract service not available at %s: %s", self.bioextract_url, e)
                return

            # Process in batches
            batch_size = 10
            for i in range(0, len(self.abstracts), batch_size):
                batch = self.abstracts[i:i + batch_size]
                texts = [a["abstract"] for a in batch]
                paper_ids = [a["paper_id"] for a in batch]

                try:
                    resp = client.post(
                        f"{self.bioextract_url}/extract/batch",
                        json={"texts": texts},
                    )
                    resp.raise_for_status()
                    extractions = resp.json()

                    for paper_id, extraction in zip(paper_ids, extractions):
                        self._results.append({
                            "paper_id": paper_id,
                            "extraction": extraction,
                        })
                except Exception as e:
                    logger.error("Extraction failed for batch %d: %s", i, e)

        logger.info("Extracted from %d abstracts", len(self._results))

    def parse(self) -> tuple[list[Entity], list[Relationship]]:
        """Convert BioExtract results to Entity + Relationship objects."""
        entities_map: dict[str, Entity] = {}
        relationships: list[Relationship] = []

        for result in self._results:
            paper_id = result["paper_id"]
            extraction = result["extraction"]

            # Build entities from extraction
            entity_id_map = {}  # text -> entity_id
            for e in extraction.get("entities", []):
                canonical_id = e.get("canonical_id")
                if canonical_id:
                    entity_id = canonical_id
                else:
                    # Generate a provisional ID from text + type
                    entity_id = f"{e['type'].lower()}:{_slugify(e['text'])}"

                entity_id_map[e["text"]] = entity_id

                if entity_id not in entities_map:
                    entities_map[entity_id] = Entity(
                        id=entity_id,
                        type=e["type"].lower(),
                        name=e.get("canonical_name") or e["text"],
                        description="",
                        synonyms=[e["text"]] if e.get("canonical_name") and e["text"] != e.get("canonical_name") else [],
                    )

            # Build relationships from extraction
            for r in extraction.get("relationships", []):
                subject_id = entity_id_map.get(r["subject"])
                object_id = entity_id_map.get(r["object"])

                if not subject_id or not object_id:
                    continue

                ctx = r.get("context", {})
                relationships.append(Relationship(
                    source_id=subject_id,
                    target_id=object_id,
                    type=r["type"],
                    source_db="bioextract",
                    confidence=r.get("confidence", 0.7),
                    evidence={
                        "paper_id": paper_id,
                        "predicate": r.get("predicate", ""),
                        "direction": r.get("direction", "neutral"),
                        "negated": r.get("negated", False),
                        "organism": ctx.get("organism"),
                        "cell_type": ctx.get("cell_type"),
                        "experiment_type": ctx.get("experiment_type"),
                        "extraction_method": extraction.get("extraction_method", "bioextract_v1"),
                    },
                ))

        return list(entities_map.values()), relationships

    def get_evidence_items(self) -> list[dict]:
        """Return evidence items for insertion into the evidence_items table.

        These are more detailed than Relationship objects — one per
        (relationship, paper) pair with sentence-level detail.
        """
        items = []
        for result in self._results:
            paper_id = result["paper_id"]
            extraction = result["extraction"]

            for r in extraction.get("relationships", []):
                ctx = r.get("context", {})
                items.append({
                    "paper_id": paper_id,
                    "subject": r["subject"],
                    "object": r["object"],
                    "relationship_type": r["type"],
                    "effect_direction": _map_direction(r.get("direction", "neutral"), r["type"]),
                    "experiment_type": ctx.get("experiment_type"),
                    "organism": ctx.get("organism"),
                    "cell_type": ctx.get("cell_type"),
                    "confidence": r.get("confidence", 0.7),
                    "extraction_method": extraction.get("extraction_method", "bioextract_v1"),
                })

        return items


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    import re
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_')


def _map_direction(direction: str, rel_type: str) -> str:
    """Map extraction direction/type to effect_direction for evidence_items."""
    direction_map = {
        "activates": "activates",
        "inhibits": "inhibits",
        "upregulates": "upregulates",
        "downregulates": "downregulates",
        "associated_with": "associated",
        "causes": "activates",
        "treats": "inhibits",
        "increases_risk": "upregulates",
        "decreases_risk": "downregulates",
        "binds": "associated",
        "phosphorylates": "activates",
        "expressed_in": "associated",
        "located_in": "associated",
        "regulates": "associated",
        "interacts_with": "associated",
    }
    return direction_map.get(rel_type, "neutral")
