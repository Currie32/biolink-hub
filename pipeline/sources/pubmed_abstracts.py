"""PubMed abstract source — fetch abstracts for dementia genes.

Uses NCBI E-utilities to search PubMed and fetch abstracts.
Abstracts are stored in the papers table and sent through BioExtract
for entity/relationship extraction.
"""

import json
import logging
import time
from dataclasses import dataclass

import httpx

from .base import Source, Entity, Relationship

logger = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Top dementia genes to fetch papers for
DEMENTIA_GENES = [
    "SNCA", "GBA1", "APOE", "APP", "PSEN1", "PSEN2",
    "MAPT", "GRN", "C9orf72", "TREM2", "BIN1", "CLU",
    "LRRK2",
]


@dataclass
class Paper:
    """A PubMed paper with abstract text."""
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: int
    abstract: str
    source_url: str
    metadata: dict


class PubMedAbstracts(Source):
    """Fetch abstracts from PubMed for dementia-related genes."""

    name = "pubmed_abstracts"

    def __init__(
        self,
        genes: list[str] | None = None,
        max_per_gene: int = 50,
        min_year: int = 2020,
        query_suffix: str = "",
    ):
        self.genes = genes or DEMENTIA_GENES
        self.max_per_gene = max_per_gene
        self.min_year = min_year
        self.query_suffix = query_suffix
        self.papers: list[Paper] = []
        self._seen_pmids: set[str] = set()

    def fetch(self) -> None:
        """Fetch abstracts from PubMed for each gene."""
        self.papers = []
        self._seen_pmids = set()

        with httpx.Client(timeout=30) as client:
            for gene in self.genes:
                try:
                    self._fetch_gene_papers(client, gene)
                except Exception as e:
                    logger.error("Failed to fetch papers for %s: %s", gene, e)
                time.sleep(0.5)  # Rate limit

        logger.info("Fetched %d unique papers for %d genes", len(self.papers), len(self.genes))

    def _fetch_gene_papers(self, client: httpx.Client, gene: str) -> None:
        """Fetch papers for a single gene."""
        # Search PubMed
        search_term = f"{gene}[Gene Name] AND {self.min_year}:{2026}[dp] AND hasabstract[text]"
        if self.query_suffix:
            search_term += f" {self.query_suffix}"
        resp = client.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": search_term,
                "retmax": self.max_per_gene,
                "sort": "relevance",
                "retmode": "json",
            },
        )
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            logger.info("  %s: no papers found", gene)
            return

        # Filter already-seen PMIDs
        new_pmids = [p for p in pmids if p not in self._seen_pmids]
        if not new_pmids:
            logger.info("  %s: all %d papers already fetched", gene, len(pmids))
            return

        time.sleep(0.5)

        # Fetch abstracts in batches of 50
        for i in range(0, len(new_pmids), 50):
            batch = new_pmids[i:i + 50]
            self._fetch_abstracts(client, batch, gene)
            if i + 50 < len(new_pmids):
                time.sleep(0.5)

    def _fetch_abstracts(self, client: httpx.Client, pmids: list[str], gene: str) -> None:
        """Fetch abstract details for a batch of PMIDs."""
        resp = client.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "xml",
                "retmode": "xml",
            },
        )
        resp.raise_for_status()

        papers = self._parse_pubmed_xml(resp.text, gene)
        for paper in papers:
            if paper.pmid not in self._seen_pmids and paper.abstract:
                self.papers.append(paper)
                self._seen_pmids.add(paper.pmid)

        logger.info("  %s: fetched %d papers (batch)", gene, len(papers))

    def _parse_pubmed_xml(self, xml_text: str, query_gene: str) -> list[Paper]:
        """Parse PubMed XML response into Paper objects.

        Uses basic string parsing to avoid lxml dependency.
        """
        papers = []

        # Split by article
        articles = xml_text.split("<PubmedArticle>")[1:]
        for article_xml in articles:
            try:
                paper = self._parse_article(article_xml, query_gene)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.debug("Failed to parse article: %s", e)

        return papers

    def _parse_article(self, xml: str, query_gene: str) -> Paper | None:
        """Parse a single PubMed article XML block."""
        pmid = _extract_tag(xml, "PMID")
        if not pmid:
            return None

        title = _extract_tag(xml, "ArticleTitle") or ""
        abstract = _extract_tag(xml, "AbstractText") or ""

        # Handle structured abstracts (multiple AbstractText elements)
        if not abstract:
            abstract_parts = []
            rest = xml
            while "<AbstractText" in rest:
                idx = rest.index("<AbstractText")
                end_idx = rest.index("</AbstractText>", idx)
                tag_content = rest[idx:end_idx + len("</AbstractText>")]
                # Extract label if present
                label = ""
                if 'Label="' in tag_content:
                    label_start = tag_content.index('Label="') + 7
                    label_end = tag_content.index('"', label_start)
                    label = tag_content[label_start:label_end]
                # Extract text content
                text_start = tag_content.index(">") + 1
                text = tag_content[text_start:tag_content.rindex("<")]
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
                rest = rest[end_idx + len("</AbstractText>"):]
            abstract = " ".join(abstract_parts)

        if not abstract:
            return None

        # Clean HTML tags from abstract
        abstract = _strip_tags(abstract)
        title = _strip_tags(title)

        journal = _extract_tag(xml, "Title") or _extract_tag(xml, "ISOAbbreviation") or ""
        year_str = _extract_tag(xml, "Year") or ""
        year = int(year_str) if year_str.isdigit() else 0

        # Authors
        authors = []
        author_parts = xml.split("<Author ")
        for part in author_parts[1:]:
            last = _extract_tag(part, "LastName") or ""
            first = _extract_tag(part, "ForeName") or ""
            if last:
                authors.append(f"{last} {first}".strip())

        return Paper(
            pmid=pmid,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=abstract,
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            metadata={"query_gene": query_gene},
        )

    def parse(self) -> tuple[list[Entity], list[Relationship]]:
        """Return empty — papers go into the papers table, not entities.

        Use insert_papers() and run through BioExtract separately.
        """
        return [], []

    def get_papers(self) -> list[Paper]:
        """Return fetched papers for insertion into the papers table."""
        return self.papers


def _extract_tag(xml: str, tag: str) -> str | None:
    """Extract text content from the first occurrence of an XML tag."""
    start_tag = f"<{tag}"
    end_tag = f"</{tag}>"

    start_idx = xml.find(start_tag)
    if start_idx == -1:
        return None

    # Find the closing > of the opening tag
    close_bracket = xml.find(">", start_idx)
    if close_bracket == -1:
        return None

    end_idx = xml.find(end_tag, close_bracket)
    if end_idx == -1:
        return None

    return xml[close_bracket + 1:end_idx].strip()


def _strip_tags(text: str) -> str:
    """Remove XML/HTML tags from text."""
    import re
    return re.sub(r'<[^>]+>', '', text).strip()
