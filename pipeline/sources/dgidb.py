"""DGIdb source — drug-gene interactions.

Uses the DGIdb GraphQL API (completely free, no API key needed) to fetch
drug-gene interactions for genes already in the database.
"""

import json
import logging
import sqlite3
from pathlib import Path

import httpx

from .base import Source, Entity, Relationship

logger = logging.getLogger(__name__)

DGIDB_URL = "https://dgidb.org/api/graphql"

# Map DGIdb interaction types to our relationship types
INTERACTION_TYPE_MAP = {
    "inhibitor": "inhibits",
    "antagonist": "inhibits",
    "blocker": "inhibits",
    "negative modulator": "inhibits",
    "antisense oligonucleotide": "inhibits",
    "suppressor": "inhibits",
    "agonist": "activates",
    "activator": "activates",
    "positive allosteric modulator": "activates",
    "stimulator": "activates",
    "inducer": "activates",
    "potentiator": "activates",
    "partial agonist": "activates",
    "binder": "binds",
    "ligand": "binds",
    "substrate": "binds",
    "cofactor": "binds",
    "modulator": "regulates",
    "allosteric modulator": "regulates",
    "vaccine": "targets",
    "antibody": "targets",
    "immunotherapy": "targets",
}

QUERY = """
query($genes: [String!]!) {
  genes(names: $genes) {
    nodes {
      name
      longName
      conceptId
      interactions {
        drug {
          name
          conceptId
          approved
        }
        interactionScore
        interactionTypes {
          type
          directionality
        }
        sources {
          sourceDbName
        }
        publications {
          pmid
        }
      }
    }
  }
}
"""


class DGIdb(Source):
    """Fetch drug-gene interactions from DGIdb."""

    name = "dgidb"

    def __init__(self, db_path: str = "biolink.db", batch_size: int = 20):
        self.db_path = db_path
        self.batch_size = batch_size
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
        """Fetch drug-gene interactions from DGIdb GraphQL API."""
        symbols = self._get_gene_symbols()
        if not symbols:
            logger.warning("No gene entities found in database")
            return

        self.raw_data = []
        with httpx.Client(timeout=30) as client:
            # Query in batches
            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i:i + self.batch_size]
                logger.info("DGIdb: querying %d genes (batch %d)...", len(batch), i // self.batch_size + 1)

                try:
                    resp = client.post(
                        DGIDB_URL,
                        json={"query": QUERY, "variables": {"genes": batch}},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
                    self.raw_data.extend(nodes)
                except Exception as e:
                    logger.error("DGIdb batch failed: %s", e)

        logger.info("DGIdb: fetched data for %d genes", len(self.raw_data))

    def parse(self) -> tuple[list[Entity], list[Relationship]]:
        """Parse DGIdb response into entities and relationships."""
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        seen_drugs: set[str] = set()

        # Build gene symbol -> entity ID map from DB
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, name FROM entities WHERE type = 'gene'").fetchall()
        conn.close()
        gene_map = {r["name"].upper(): r["id"] for r in rows}

        for gene_node in self.raw_data:
            gene_symbol = gene_node.get("name", "")
            gene_id = gene_map.get(gene_symbol.upper())
            if not gene_id:
                continue

            for interaction in gene_node.get("interactions", []):
                drug = interaction.get("drug", {})
                drug_name = drug.get("name", "")
                drug_concept_id = drug.get("conceptId", "")
                approved = drug.get("approved", False)

                if not drug_name:
                    continue

                # Create drug entity ID from DGIdb concept ID
                if drug_concept_id:
                    drug_id = drug_concept_id.lower().replace(":", ":")
                else:
                    continue  # Skip drugs without canonical IDs

                # Create drug entity (deduplicated)
                if drug_id not in seen_drugs:
                    seen_drugs.add(drug_id)
                    entities.append(Entity(
                        id=drug_id,
                        type="drug",
                        name=drug_name,
                        metadata={
                            "approved": approved,
                        },
                        external_ids={"DGIdb": drug_concept_id},
                    ))

                # Determine relationship type
                interaction_types = interaction.get("interactionTypes", [])
                rel_type = "targets"  # default
                for it in interaction_types:
                    mapped = INTERACTION_TYPE_MAP.get(it.get("type", "").lower())
                    if mapped:
                        rel_type = mapped
                        break

                # Confidence from interaction score
                score = interaction.get("interactionScore")
                confidence = 0.9  # curated source baseline

                # Sources
                sources = [s.get("sourceDbName", "") for s in interaction.get("sources", [])]

                # PMIDs
                pmids = [p.get("pmid") for p in interaction.get("publications", []) if p.get("pmid")]

                relationships.append(Relationship(
                    source_id=drug_id,
                    target_id=gene_id,
                    type=rel_type,
                    source_db="dgidb",
                    confidence=confidence,
                    evidence={
                        "sources": sources,
                        "pmids": pmids[:10],
                        "interaction_types": [it.get("type", "") for it in interaction_types],
                        "approved_drug": approved,
                    },
                ))

        logger.info("DGIdb: %d drug entities, %d relationships", len(entities), len(relationships))
        return entities, relationships
