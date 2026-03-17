"""Download and prepare biomedical dictionaries for entity normalization.

Dictionaries provide canonical IDs, names, and synonyms for biomedical entities.
Each download function fetches data and writes to a standard TSV format:
    canonical_id\tname\tentity_type\tsource_db\tsynonyms_json

Some sources (UMLS) require a license — these are skipped if credentials are missing.
"""

import gzip
import io
import json
import logging
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DICT_DIR = Path(__file__).parent / "data"


def ensure_dir():
    DICT_DIR.mkdir(parents=True, exist_ok=True)


def download_ncbi_gene(output: Path | None = None) -> Path:
    """Download NCBI Gene info for Homo sapiens.

    Source: NCBI Gene FTP (gene_info.gz, filtered to tax_id=9606)
    ~60K human genes with symbols, aliases, descriptions.
    """
    ensure_dir()
    output = output or DICT_DIR / "ncbi_gene.tsv"

    url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
    logger.info("Downloading NCBI Gene data...")

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    entries = []
    with gzip.open(io.BytesIO(resp.content), "rt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                # Parse header
                header = line.lstrip("#").strip().split("\t")
                continue
            if not header:
                continue
            fields = line.strip().split("\t")
            row = dict(zip(header, fields))

            gene_id = row.get("GeneID", "")
            symbol = row.get("Symbol", "")
            description = row.get("description", "")
            synonyms_raw = row.get("Synonyms", "-")

            synonyms = [s.strip() for s in synonyms_raw.split("|") if s.strip() and s != "-"]
            full_name = row.get("Full_name_from_nomenclature_authority", "-")
            if full_name and full_name != "-" and full_name != symbol:
                synonyms.insert(0, full_name)

            entries.append((
                f"gene:{gene_id}",
                symbol,
                "GENE",
                "ncbi_gene",
                json.dumps(synonyms),
            ))

    with open(output, "w") as f:
        for entry in entries:
            f.write("\t".join(entry) + "\n")

    logger.info("NCBI Gene: %d entries written to %s", len(entries), output)
    return output


def download_mesh(output: Path | None = None) -> Path:
    """Download MeSH (Medical Subject Headings) descriptors via XML.

    Source: NLM MeSH XML descriptor file (open access).
    ~31K descriptors covering diseases, chemicals, anatomy, organisms.
    """
    ensure_dir()
    output = output or DICT_DIR / "mesh.tsv"

    url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"
    logger.info("Downloading MeSH XML descriptors (~300MB)... this may take a few minutes")

    with httpx.Client(timeout=600, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    # File may be gzipped or raw XML despite .gz extension
    data = resp.content
    try:
        data = gzip.decompress(data)
    except gzip.BadGzipFile:
        pass  # Already uncompressed

    logger.info("Parsing MeSH XML...")
    root = ET.fromstring(data)

    entries = []
    for record in root.findall("DescriptorRecord"):
        ui_el = record.find("DescriptorUI")
        name_el = record.find("DescriptorName/String")
        if ui_el is None or name_el is None:
            continue

        ui = ui_el.text
        name = name_el.text

        # Collect tree numbers for classification
        tree_numbers = []
        for tn in record.findall(".//TreeNumber"):
            if tn.text:
                tree_numbers.append(tn.text)

        entity_type = _classify_mesh_entry(tree_numbers)

        # Collect synonyms from concepts/terms
        synonyms = []
        for term in record.findall(".//Term/String"):
            if term.text and term.text != name:
                synonyms.append(term.text)
        # Also collect entry terms
        for entry in record.findall(".//ConceptRelation/Concept2UI"):
            pass  # Concepts are already covered by Term elements

        entries.append((
            f"mesh:{ui}",
            name,
            entity_type,
            "mesh",
            json.dumps(list(set(synonyms))),
        ))

    with open(output, "w") as f:
        for entry in entries:
            f.write("\t".join(entry) + "\n")

    logger.info("MeSH: %d entries written to %s", len(entries), output)
    return output


def _classify_mesh_entry(tree_numbers: list[str]) -> str:
    """Classify a MeSH entry by its tree number prefix."""
    # Priority order: prefer more specific classifications
    for tn in tree_numbers:
        if tn.startswith("C"):  # Diseases
            return "DISEASE"
    for tn in tree_numbers:
        if tn.startswith("D"):  # Chemicals and Drugs
            return "CHEMICAL"
    for tn in tree_numbers:
        if tn.startswith("B"):  # Organisms
            return "ORGANISM"
    for tn in tree_numbers:
        if tn.startswith("A"):  # Anatomy
            return "ANATOMICAL_STRUCTURE"
    for tn in tree_numbers:
        if tn.startswith("G"):  # Phenomena and Processes
            return "BIOLOGICAL_PROCESS"
    for tn in tree_numbers:
        if tn.startswith("F"):  # Psychiatry and Psychology
            return "PHENOTYPE"
    return "CHEMICAL"  # Default for unclassified


def download_disease_ontology(output: Path | None = None) -> Path:
    """Download Disease Ontology (DO) terms.

    Source: Disease Ontology OBO file (open access).
    """
    ensure_dir()
    output = output or DICT_DIR / "disease_ontology.tsv"

    url = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"
    logger.info("Downloading Disease Ontology...")

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    entries = []
    current_id = None
    current_name = None
    synonyms = []
    is_obsolete = False

    for line in resp.text.split("\n"):
        line = line.strip()
        if line == "[Term]":
            if current_id and current_name and not is_obsolete:
                entries.append((
                    f"doid:{current_id}",
                    current_name,
                    "DISEASE",
                    "disease_ontology",
                    json.dumps(synonyms),
                ))
            current_id = None
            current_name = None
            synonyms = []
            is_obsolete = False
        elif line.startswith("id: DOID:"):
            current_id = line.split("DOID:")[1].strip()
        elif line.startswith("name: "):
            current_name = line[6:]
        elif line.startswith("synonym: "):
            # Parse: synonym: "text" EXACT|RELATED|BROAD|NARROW []
            try:
                syn_text = line.split('"')[1]
                synonyms.append(syn_text)
            except IndexError:
                pass
        elif line == "is_obsolete: true":
            is_obsolete = True

    # Last entry
    if current_id and current_name and not is_obsolete:
        entries.append((
            f"doid:{current_id}",
            current_name,
            "DISEASE",
            "disease_ontology",
            json.dumps(synonyms),
        ))

    with open(output, "w") as f:
        for entry in entries:
            f.write("\t".join(entry) + "\n")

    logger.info("Disease Ontology: %d entries written to %s", len(entries), output)
    return output


def download_hpo(output: Path | None = None) -> Path:
    """Download Human Phenotype Ontology (HPO) terms.

    Source: HPO OBO file (open access).
    """
    ensure_dir()
    output = output or DICT_DIR / "hpo.tsv"

    url = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"
    logger.info("Downloading HPO...")

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    entries = []
    current_id = None
    current_name = None
    synonyms = []
    is_obsolete = False

    for line in resp.text.split("\n"):
        line = line.strip()
        if line == "[Term]":
            if current_id and current_name and not is_obsolete:
                entries.append((
                    f"hp:{current_id}",
                    current_name,
                    "PHENOTYPE",
                    "hpo",
                    json.dumps(synonyms),
                ))
            current_id = None
            current_name = None
            synonyms = []
            is_obsolete = False
        elif line.startswith("id: HP:"):
            current_id = line.split("HP:")[1].strip()
        elif line.startswith("name: "):
            current_name = line[6:]
        elif line.startswith("synonym: "):
            try:
                syn_text = line.split('"')[1]
                synonyms.append(syn_text)
            except IndexError:
                pass
        elif line == "is_obsolete: true":
            is_obsolete = True

    if current_id and current_name and not is_obsolete:
        entries.append((
            f"hp:{current_id}",
            current_name,
            "PHENOTYPE",
            "hpo",
            json.dumps(synonyms),
        ))

    with open(output, "w") as f:
        for entry in entries:
            f.write("\t".join(entry) + "\n")

    logger.info("HPO: %d entries written to %s", len(entries), output)
    return output


def download_chebi(output: Path | None = None) -> Path:
    """Download ChEBI (Chemical Entities of Biological Interest) terms.

    Source: ChEBI OBO file (open access). Smaller than full SDF.
    """
    ensure_dir()
    output = output or DICT_DIR / "chebi.tsv"

    url = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo"
    logger.info("Downloading ChEBI...")

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    entries = []
    current_id = None
    current_name = None
    synonyms = []
    is_obsolete = False

    for line in resp.text.split("\n"):
        line = line.strip()
        if line == "[Term]":
            if current_id and current_name and not is_obsolete:
                entries.append((
                    f"chebi:{current_id}",
                    current_name,
                    "CHEMICAL",
                    "chebi",
                    json.dumps(synonyms),
                ))
            current_id = None
            current_name = None
            synonyms = []
            is_obsolete = False
        elif line.startswith("id: CHEBI:"):
            current_id = line.split("CHEBI:")[1].strip()
        elif line.startswith("name: "):
            current_name = line[6:]
        elif line.startswith("synonym: "):
            try:
                syn_text = line.split('"')[1]
                synonyms.append(syn_text)
            except IndexError:
                pass
        elif line == "is_obsolete: true":
            is_obsolete = True

    if current_id and current_name and not is_obsolete:
        entries.append((
            f"chebi:{current_id}",
            current_name,
            "CHEMICAL",
            "chebi",
            json.dumps(synonyms),
        ))

    with open(output, "w") as f:
        for entry in entries:
            f.write("\t".join(entry) + "\n")

    logger.info("ChEBI: %d entries written to %s", len(entries), output)
    return output


def download_ncbi_taxonomy(output: Path | None = None) -> Path:
    """Download NCBI Taxonomy names for common organisms.

    Source: NCBI Taxonomy dump (names.dmp from taxdump.tar.gz).
    Filtered to common biomedical organisms to keep size manageable.
    """
    ensure_dir()
    output = output or DICT_DIR / "ncbi_taxonomy.tsv"

    url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
    logger.info("Downloading NCBI Taxonomy (~70MB)...")

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    # Extract names.dmp from tarball
    names_data = None
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar:
            if member.name == "names.dmp":
                f = tar.extractfile(member)
                if f:
                    names_data = f.read().decode("utf-8")
                break

    if not names_data:
        raise RuntimeError("names.dmp not found in taxdump.tar.gz")

    # Parse names.dmp: tax_id | name | unique_name | name_class
    # Keep scientific names and common names
    tax_names: dict[str, dict] = {}
    for line in names_data.split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        tax_id = parts[0]
        name = parts[1]
        name_class = parts[3]

        if tax_id not in tax_names:
            tax_names[tax_id] = {"scientific": None, "synonyms": []}

        if name_class == "scientific name":
            tax_names[tax_id]["scientific"] = name
        elif name_class in ("common name", "genbank common name", "equivalent name", "synonym"):
            tax_names[tax_id]["synonyms"].append(name)

    # Common biomedical organisms — include these tax IDs and any with common names
    # that are likely to appear in biomedical literature
    important_taxa = {
        "9606",   # Homo sapiens
        "10090",  # Mus musculus (mouse)
        "10116",  # Rattus norvegicus (rat)
        "7227",   # Drosophila melanogaster
        "6239",   # C. elegans
        "7955",   # Danio rerio (zebrafish)
        "9615",   # Canis lupus familiaris (dog)
        "9913",   # Bos taurus (cow/bovine)
        "9823",   # Sus scrofa (pig)
        "9986",   # Oryctolagus cuniculus (rabbit)
        "10029",  # Cricetulus griseus (Chinese hamster)
        "9031",   # Gallus gallus (chicken)
        "8364",   # Xenopus tropicalis (frog)
        "4932",   # S. cerevisiae (yeast)
        "562",    # E. coli
        "83333",  # E. coli K-12
        "1773",   # Mycobacterium tuberculosis
        "36329",  # Plasmodium falciparum
        "11103",  # Hepatitis C virus
        "12721",  # HIV-1
        "694009", # SARS-CoV-2
        "9544",   # Macaca mulatta (rhesus macaque)
        "9541",   # Macaca fascicularis (cynomolgus monkey)
        "9598",   # Pan troglodytes (chimpanzee)
        "10036",  # Mesocricetus auratus (Syrian hamster)
        "452646", # Guinea pig (Cavia porcellus)
        "9685",   # Felis catus (cat)
        "9796",   # Equus caballus (horse)
        "9940",   # Ovis aries (sheep)
        "9925",   # Capra hircus (goat)
    }

    entries = []
    for tax_id, info in tax_names.items():
        scientific = info["scientific"]
        if not scientific:
            continue

        # Include important taxa or any species with common names
        syns = info["synonyms"]
        if tax_id not in important_taxa and not syns:
            continue

        # For non-important taxa, only include if they have common names
        # (suggests they're commonly referenced in literature)
        if tax_id not in important_taxa and len(syns) < 1:
            continue

        entries.append((
            f"taxon:{tax_id}",
            scientific,
            "ORGANISM",
            "ncbi_taxonomy",
            json.dumps(syns),
        ))

    with open(output, "w") as f:
        for entry in entries:
            f.write("\t".join(entry) + "\n")

    logger.info("NCBI Taxonomy: %d entries written to %s", len(entries), output)
    return output


def download_cellosaurus(output: Path | None = None) -> Path:
    """Download Cellosaurus cell line database.

    Source: Cellosaurus OBO file from ExPASy (open access).
    ~140K cell lines with synonyms and cross-references.
    """
    ensure_dir()
    output = output or DICT_DIR / "cellosaurus.tsv"

    url = "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo"
    logger.info("Downloading Cellosaurus (~110MB)...")

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    entries = []
    current_id = None
    current_name = None
    synonyms = []
    in_term = False

    for line in resp.text.split("\n"):
        line = line.strip()
        if line == "[Term]":
            # Save previous term
            if in_term and current_id and current_name and current_id.startswith("CVCL_"):
                entries.append((
                    f"cvcl:{current_id}",
                    current_name,
                    "CELL_TYPE",
                    "cellosaurus",
                    json.dumps(synonyms),
                ))
            current_id = None
            current_name = None
            synonyms = []
            in_term = True
        elif line.startswith("[") and line.endswith("]"):
            # [Typedef] or other section — save and stop
            if in_term and current_id and current_name and current_id.startswith("CVCL_"):
                entries.append((
                    f"cvcl:{current_id}",
                    current_name,
                    "CELL_TYPE",
                    "cellosaurus",
                    json.dumps(synonyms),
                ))
            in_term = False
            current_id = None
            current_name = None
            synonyms = []
        elif in_term:
            if line.startswith("id: "):
                current_id = line[4:].strip()
            elif line.startswith("name:"):
                current_name = line[5:].strip()
            elif line.startswith("synonym: "):
                try:
                    syn_text = line.split('"')[1]
                    if syn_text:
                        synonyms.append(syn_text)
                except IndexError:
                    pass

    # Last entry
    if in_term and current_id and current_name and current_id.startswith("CVCL_"):
        entries.append((
            f"cvcl:{current_id}",
            current_name,
            "CELL_TYPE",
            "cellosaurus",
            json.dumps(synonyms),
        ))

    with open(output, "w") as f:
        for entry in entries:
            f.write("\t".join(entry) + "\n")

    logger.info("Cellosaurus: %d entries written to %s", len(entries), output)
    return output


def download_drugbank_vocab(output: Path | None = None) -> Path:
    """Download DrugBank open vocabulary (drug names + synonyms).

    Source: DrugBank open data (no license required for vocabulary).
    Note: Full DrugBank requires academic license. This uses the open subset.
    """
    ensure_dir()
    output = output or DICT_DIR / "drugbank.tsv"

    # DrugBank open structures CSV (includes names)
    url = "https://go.drugbank.com/releases/latest/downloads/all-drugbank-vocabulary"
    logger.info("Downloading DrugBank vocabulary...")
    logger.warning(
        "DrugBank vocabulary requires login. If this fails, manually download from "
        "https://go.drugbank.com/releases/latest#open-data and place at %s",
        output,
    )

    # DrugBank requires auth — provide a placeholder for manual download
    # For now, create an empty file so the index still works
    if not output.exists():
        output.touch()
        logger.info("Created placeholder %s — populate manually from DrugBank open data", output)

    return output


def download_all():
    """Download all available dictionaries."""
    results = {}
    for name, func in [
        ("ncbi_gene", download_ncbi_gene),
        ("mesh", download_mesh),
        ("disease_ontology", download_disease_ontology),
        ("hpo", download_hpo),
        ("chebi", download_chebi),
        ("ncbi_taxonomy", download_ncbi_taxonomy),
        ("cellosaurus", download_cellosaurus),
        ("drugbank", download_drugbank_vocab),
    ]:
        try:
            path = func()
            results[name] = str(path)
            logger.info("OK: %s", name)
        except Exception as e:
            logger.error("FAILED: %s — %s", name, e)
            results[name] = f"error: {e}"
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()
