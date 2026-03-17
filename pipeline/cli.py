"""CLI for biolink-hub pipeline."""

import click
from pathlib import Path


@click.group()
def cli():
    """BioLink Hub -- Open Biomedical Knowledge Explorer."""


@cli.command()
@click.option("--db", default="biolink.db", help="Output database path")
def build(db):
    """Build the SQLite database from all sources."""
    from .build_db import build as run_build

    run_build(db_path=Path(db))


@cli.command()
@click.argument("query")
@click.option("--db", default="biolink.db", help="Database path")
def search(query, db):
    """Search entities by name."""
    import json
    import sqlite3

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT e.* FROM entity_search s JOIN entities e ON e.id = s.entity_id "
        "WHERE entity_search MATCH ? ORDER BY rank LIMIT 10",
        (query,),
    ).fetchall()

    if not rows:
        click.echo("No results found.")
        return

    for row in rows:
        click.echo(f"\n{row['name']} ({row['type']}) -- {row['id']}")
        click.echo(f"  {row['description']}")
        ext = json.loads(row['external_ids'])
        if ext:
            click.echo(f"  External IDs: {ext}")


@cli.command()
@click.argument("entity_id")
@click.option("--db", default="biolink.db", help="Database path")
def info(entity_id, db):
    """Show details for an entity by ID."""
    import json
    import sqlite3

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()

    if not row:
        click.echo(f"Entity not found: {entity_id}")
        return

    click.echo(f"\n{row['name']} ({row['type']})")
    click.echo(f"ID: {row['id']}")
    click.echo(f"Description: {row['description']}")
    click.echo(f"Synonyms: {json.loads(row['synonyms'])}")
    click.echo(f"External IDs: {json.loads(row['external_ids'])}")
    click.echo(f"Metadata: {json.loads(row['metadata'])}")

    # Show relationships
    rels = conn.execute(
        "SELECT r.*, e.name as target_name FROM relationships r "
        "JOIN entities e ON e.id = r.target_id WHERE r.source_id = ?",
        (entity_id,),
    ).fetchall()
    if rels:
        click.echo(f"\nRelationships ({len(rels)}):")
        for r in rels:
            confidence = f" [{r['confidence']:.0%}]" if r['confidence'] else ""
            click.echo(f"  --[{r['type']}]--> {r['target_name']} ({r['target_id']}){confidence}")

    # Show evidence count
    try:
        ev_count = conn.execute(
            "SELECT COUNT(*) FROM evidence_items ei "
            "JOIN relationships r ON r.id = ei.relationship_id "
            "WHERE r.source_id = ? OR r.target_id = ?",
            (entity_id, entity_id),
        ).fetchone()[0]
        if ev_count > 0:
            click.echo(f"\nEvidence items: {ev_count}")
    except sqlite3.OperationalError:
        pass


@cli.command()
@click.option("--port", default=8000, help="API port")
def serve(port):
    """Run the API server."""
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=port, reload=True)


# --- BioExtract commands ---

@cli.group()
def dictionaries():
    """Manage biomedical dictionaries."""


@dictionaries.command("download")
@click.option("--source", default=None, help="Download specific source (ncbi_gene, mesh, disease_ontology, hpo, chebi)")
def dict_download(source):
    """Download biomedical dictionaries for entity normalization."""
    import logging
    logging.basicConfig(level=logging.INFO)

    from bioextract.dictionaries.download import (
        download_all, download_ncbi_gene, download_mesh,
        download_disease_ontology, download_hpo, download_chebi,
        download_ncbi_taxonomy, download_cellosaurus,
    )

    if source:
        funcs = {
            "ncbi_gene": download_ncbi_gene,
            "mesh": download_mesh,
            "disease_ontology": download_disease_ontology,
            "hpo": download_hpo,
            "chebi": download_chebi,
            "ncbi_taxonomy": download_ncbi_taxonomy,
            "cellosaurus": download_cellosaurus,
        }
        if source not in funcs:
            click.echo(f"Unknown source: {source}. Options: {', '.join(funcs.keys())}")
            return
        funcs[source]()
    else:
        download_all()


@dictionaries.command("index")
def dict_index():
    """Build the unified dictionary lookup index."""
    import logging
    logging.basicConfig(level=logging.INFO)

    from bioextract.dictionaries.index import build_index
    build_index()


@dictionaries.command("lookup")
@click.argument("term")
@click.option("--type", "entity_type", default=None, help="Filter by entity type")
def dict_lookup(term, entity_type):
    """Look up a term in the dictionary."""
    from bioextract.dictionaries.lookup import DictionaryLookup

    dl = DictionaryLookup()
    if not dl.is_available():
        click.echo("Dictionary index not built. Run: biolink dictionaries download && biolink dictionaries index")
        return

    # Try exact match
    matches = dl.exact_match(term, entity_type)
    if matches:
        click.echo(f"Exact matches for '{term}':")
        for m in matches:
            click.echo(f"  {m.canonical_id} | {m.name} ({m.entity_type}) [{m.source_db}] score={m.score:.2f}")
        return

    # Fall back to search
    matches = dl.search(term, entity_type)
    if matches:
        click.echo(f"Search results for '{term}':")
        for m in matches:
            click.echo(f"  {m.canonical_id} | {m.name} ({m.entity_type}) [{m.source_db}] score={m.score:.2f}")
    else:
        click.echo(f"No matches for '{term}'")


@cli.command("extract")
@click.argument("text")
def extract_text(text):
    """Extract entities and relationships from text using BioExtract."""
    import json
    from bioextract.extract import BioExtractor

    extractor = BioExtractor()
    result = extractor.extract(text)

    click.echo(f"\nExtraction method: {result.extraction_method}")
    click.echo(f"Entities ({len(result.entities)}):")
    for e in result.entities:
        canonical = f" -> {e.canonical_id} ({e.canonical_name})" if e.canonical_id else ""
        click.echo(f"  [{e.type}] {e.text}{canonical}")

    click.echo(f"\nRelationships ({len(result.relationships)}):")
    for r in result.relationships:
        click.echo(f"  {r.subject} --[{r.type}]--> {r.object} (direction={r.direction})")
        if r.context.organism or r.context.cell_type:
            ctx_parts = []
            if r.context.organism:
                ctx_parts.append(f"organism={r.context.organism}")
            if r.context.cell_type:
                ctx_parts.append(f"cell_type={r.context.cell_type}")
            click.echo(f"    context: {', '.join(ctx_parts)}")


@cli.command("ingest-dgidb")
@click.option("--db", default="biolink.db", help="Database path")
def ingest_dgidb(db):
    """Fetch drug-gene interactions from DGIdb (free, no API key needed)."""
    import logging
    logging.basicConfig(level=logging.INFO)
    import sqlite3
    from .sources.dgidb import DGIdb
    from .build_db import SCHEMA, insert_entities, insert_relationships

    source = DGIdb(db_path=db)
    click.echo("Fetching from DGIdb...")
    source.fetch()

    click.echo("Parsing...")
    entities, relationships = source.parse()
    click.echo(f"Found {len(entities)} drugs, {len(relationships)} interactions")

    if not entities and not relationships:
        click.echo("No data found.")
        return

    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)

    n_ent = insert_entities(conn, entities)
    n_rel = insert_relationships(conn, relationships)
    conn.close()
    click.echo(f"Inserted {n_ent} drug entities, {n_rel} drug-gene relationships")


@cli.command("ingest-disgenet")
@click.option("--db", default="biolink.db", help="Database path")
@click.option("--min-score", default=0.3, help="Minimum GDA score (0-1)")
def ingest_disgenet(db, min_score):
    """Fetch gene-disease associations from DisGeNET (requires DISGENET_API_KEY)."""
    import logging
    logging.basicConfig(level=logging.INFO)
    import sqlite3
    from .sources.disgenet import DisGeNET
    from .build_db import SCHEMA, insert_entities, insert_relationships

    source = DisGeNET(db_path=db, min_score=min_score)
    if not source.api_key:
        click.echo("Set DISGENET_API_KEY environment variable. Get a free key at https://www.disgenet.org/signup/")
        return

    click.echo("Fetching from DisGeNET...")
    source.fetch()

    click.echo("Parsing...")
    entities, relationships = source.parse()
    click.echo(f"Found {len(entities)} diseases, {len(relationships)} associations")

    if not entities and not relationships:
        click.echo("No data found.")
        return

    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)

    n_ent = insert_entities(conn, entities)
    n_rel = insert_relationships(conn, relationships)
    conn.close()
    click.echo(f"Inserted {n_ent} disease entities, {n_rel} gene-disease associations")


@cli.command("ingest-papers")
@click.option("--db", default="biolink.db", help="Database path")
@click.option("--max-per-gene", default=50, help="Max papers per gene")
@click.option("--min-year", default=2020, help="Minimum publication year")
@click.option("--query-suffix", default="", help="Extra PubMed search terms (e.g. 'AND drug therapy')")
def ingest_papers(db, max_per_gene, min_year, query_suffix):
    """Fetch PubMed abstracts and ingest into the database."""
    import sqlite3
    from .sources.pubmed_abstracts import PubMedAbstracts
    from .build_db import init_db, insert_papers

    click.echo("Fetching PubMed abstracts...")
    source = PubMedAbstracts(max_per_gene=max_per_gene, min_year=min_year, query_suffix=query_suffix)
    source.fetch()

    papers = source.get_papers()
    click.echo(f"Fetched {len(papers)} papers")

    if not papers:
        return

    conn = sqlite3.connect(db)
    # Ensure tables exist
    from .build_db import SCHEMA
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row

    n = insert_papers(conn, papers)
    click.echo(f"Inserted {n} papers into database")
    conn.close()
