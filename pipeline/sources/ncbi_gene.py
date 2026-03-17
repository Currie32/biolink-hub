"""NCBI Gene source for dementia-related genes."""

import time
import httpx
from .base import Source, Entity, Relationship

# Dementia-related genes to fetch
DEMENTIA_GENES = [
    "SNCA", "GBA1", "APOE", "APP", "PSEN1", "PSEN2",
    "MAPT", "GRN", "C9orf72", "TREM2", "BIN1", "CLU",
    "LRRK2",
]

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class NCBIGene(Source):
    name = "ncbi_gene"

    def __init__(self):
        self.raw_data: list[dict] = []

    def fetch(self) -> None:
        """Fetch gene data from NCBI E-utilities using batch queries."""
        self.raw_data = []
        with httpx.Client(timeout=30) as client:
            # Batch search: one query for all genes
            search_term = " OR ".join(
                f"{s}[Gene Name]" for s in DEMENTIA_GENES
            )
            search_term = f"({search_term}) AND Homo sapiens[Organism]"

            resp = client.get(
                f"{NCBI_BASE}/esearch.fcgi",
                params={
                    "db": "gene",
                    "term": search_term,
                    "retmax": 50,
                    "retmode": "json",
                },
            )
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", [])

            if not id_list:
                print("  No gene IDs found")
                return

            # Rate limit: wait before next request
            time.sleep(0.5)

            # Batch summary: one call with all IDs
            resp = client.get(
                f"{NCBI_BASE}/esummary.fcgi",
                params={
                    "db": "gene",
                    "id": ",".join(id_list),
                    "retmode": "json",
                },
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})

            for gid in id_list:
                summary = result.get(gid)
                if summary and summary.get("name", "").upper() in [g.upper() for g in DEMENTIA_GENES]:
                    self.raw_data.append(summary)

    def parse(self) -> tuple[list[Entity], list[Relationship]]:
        """Convert raw NCBI gene data to Entity objects."""
        entities = []
        for gene in self.raw_data:
            gene_id = str(gene.get("uid", ""))
            symbol = gene.get("name", "")
            description = gene.get("description", "")
            full_name = gene.get("nomenclaturename", "")
            other_aliases = gene.get("otheraliases", "")

            synonyms = [s.strip() for s in other_aliases.split(",") if s.strip()]
            if full_name and full_name != symbol:
                synonyms.insert(0, full_name)

            chrom = gene.get("chromosome", "")
            map_location = gene.get("maplocation", "")
            gene_type = gene.get("geneticSource", "")

            external_ids = {"NCBI_Gene": gene_id}
            nom_id = gene.get("nomenclatureid", "")
            if nom_id:
                external_ids["HGNC"] = nom_id

            entity = Entity(
                id=f"gene:{gene_id}",
                type="gene",
                name=symbol,
                description=description or full_name,
                synonyms=synonyms,
                external_ids=external_ids,
                metadata={
                    "chromosome": chrom,
                    "map_location": map_location,
                    "gene_type": gene_type,
                },
            )
            entities.append(entity)

        return entities, []
