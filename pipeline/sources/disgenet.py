"""DisGeNET source — gene-disease associations.

Uses the DisGeNET REST API to fetch curated gene-disease associations.
Requires a free API key from https://www.disgenet.org/signup/
Set DISGENET_API_KEY environment variable.
"""

import logging
import os
import sqlite3
import time

import httpx

from .base import Source, Entity, Relationship

logger = logging.getLogger(__name__)

DISGENET_BASE = "https://api.disgenet.com/api/v1"

# Map DisGeNET disease vocabularies to our ID format
DISEASE_VOCAB_PRIORITY = ["DO", "MESH", "OMIM", "UMLS"]


class DisGeNET(Source):
    """Fetch gene-disease associations from DisGeNET."""

    name = "disgenet"

    def __init__(
        self,
        db_path: str = "biolink.db",
        min_score: float = 0.3,
        api_key: str | None = None,
    ):
        self.db_path = db_path
        self.min_score = min_score
        self.api_key = api_key or os.environ.get("DISGENET_API_KEY")
        self.raw_data: list[dict] = []

    def _get_gene_symbols(self) -> list[str]:
        """Get gene symbols from existing entities in the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name FROM entities WHERE type = 'gene' ORDER BY name"
        ).fetchall()
        conn.close()
        return [r["name"] for r in rows]

    def fetch(self) -> None:
        """Fetch gene-disease associations from DisGeNET API."""
        if not self.api_key:
            logger.warning(
                "No DISGENET_API_KEY set. Get a free key at https://www.disgenet.org/signup/ "
                "and set the DISGENET_API_KEY environment variable."
            )
            return

        symbols = self._get_gene_symbols()
        if not symbols:
            logger.warning("No gene entities found in database")
            return

        self.raw_data = []
        with httpx.Client(timeout=30) as client:
            # DisGeNET free tier: max 10 gene symbols per request
            for i in range(0, len(symbols), 10):
                batch = symbols[i:i + 10]
                logger.info("DisGeNET: querying %d genes (batch %d)...", len(batch), i // 10 + 1)

                try:
                    resp = client.get(
                        f"{DISGENET_BASE}/gda/summary",
                        params={
                            "gene_symbol": ",".join(batch),
                            "min_score": self.min_score,
                            "source": "ALL",
                        },
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Accept": "application/json",
                        },
                    )
                    if resp.status_code == 401:
                        logger.error("DisGeNET: invalid API key")
                        return
                    if resp.status_code == 429:
                        logger.warning("DisGeNET: rate limited, waiting 60s...")
                        time.sleep(60)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list):
                        self.raw_data.extend(data)
                    elif isinstance(data, dict) and "payload" in data:
                        self.raw_data.extend(data["payload"])
                except Exception as e:
                    logger.error("DisGeNET batch failed: %s", e)

                time.sleep(1)  # Rate limit

        logger.info("DisGeNET: fetched %d associations", len(self.raw_data))

    def parse(self) -> tuple[list[Entity], list[Relationship]]:
        """Parse DisGeNET response into entities and relationships."""
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        seen_diseases: set[str] = set()

        # Build gene symbol -> entity ID map from DB
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, name FROM entities WHERE type = 'gene'").fetchall()
        conn.close()
        gene_map = {r["name"].upper(): r["id"] for r in rows}

        for assoc in self.raw_data:
            gene_symbol = assoc.get("symbolOfGene", "")
            gene_id = gene_map.get(gene_symbol.upper())
            if not gene_id:
                continue

            disease_name = assoc.get("diseaseName", "")
            disease_umls = assoc.get("diseaseUMLSCUI", "")
            score = assoc.get("score", 0)

            if not disease_name or not disease_umls:
                continue

            # Try to find a good disease ID from vocabularies
            disease_id = None
            vocabs = assoc.get("diseaseVocabularies", [])
            for vocab in DISEASE_VOCAB_PRIORITY:
                for v in vocabs:
                    if isinstance(v, str) and v.startswith(f"{vocab}:"):
                        prefix = vocab.lower()
                        suffix = v.split(":", 1)[1]
                        disease_id = f"{prefix}:{suffix}"
                        break
                    elif isinstance(v, dict) and v.get("vocabulary") == vocab:
                        disease_id = f"{vocab.lower()}:{v.get('code', '')}"
                        break
                if disease_id:
                    break

            if not disease_id:
                disease_id = f"umls:{disease_umls}"

            # Create disease entity (deduplicated)
            if disease_id not in seen_diseases:
                seen_diseases.add(disease_id)
                disease_type = assoc.get("diseaseType", "disease")
                entity_type = "phenotype" if disease_type == "phenotype" else "disease"

                entities.append(Entity(
                    id=disease_id,
                    type=entity_type,
                    name=disease_name,
                    external_ids={"UMLS": disease_umls},
                    metadata={
                        "disease_type": disease_type,
                        "disease_classes": assoc.get("diseaseClasses_MSH", []),
                    },
                ))

            # Evidence level
            evidence_level = assoc.get("el", "")
            num_pmids = assoc.get("numPMIDs", 0)

            relationships.append(Relationship(
                source_id=gene_id,
                target_id=disease_id,
                type="associated_with",
                source_db="disgenet",
                confidence=score,
                evidence={
                    "evidence_level": evidence_level,
                    "evidence_index": assoc.get("ei"),
                    "num_publications": num_pmids,
                    "year_initial": assoc.get("yearInitial"),
                    "year_final": assoc.get("yearFinal"),
                },
            ))

        logger.info("DisGeNET: %d disease entities, %d relationships", len(entities), len(relationships))
        return entities, relationships
