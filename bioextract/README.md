# BioExtract

Biomedical entity and relationship extraction service. Extracts genes, diseases,
drugs, pathways, and their relationships from biomedical text. Normalizes
mentions to canonical database IDs.

## Quick Start

All commands run from the project root (`biolink-hub/`).

```bash
source .venv/bin/activate

# 1. Download dictionaries (open-access, ~5 min)
biolink dictionaries download

# 2. Build the lookup index
biolink dictionaries index

# 3. Test a lookup
biolink dictionaries lookup "Alzheimer's disease"
biolink dictionaries lookup APOE
biolink dictionaries lookup "alpha-synuclein"

# 4. Extract from text (requires ANTHROPIC_API_KEY until student model is trained)
export ANTHROPIC_API_KEY=sk-ant-...
biolink extract "TREM2 reduces neuroinflammation in mouse microglia"
```

## Architecture

```
Text in
  -> Extraction model (student or Claude teacher fallback)
     -> raw entity mentions + relationship triples
  -> Dictionary matching (NCBI Gene, MeSH, Disease Ontology, HPO, ChEBI)
     -> candidate canonical IDs per mention
  -> Normalization (exact match, fuzzy match, abbreviation disambiguation)
     -> grounded entities with canonical IDs + typed relationships
Structured JSON out
```

### Extraction modes

BioExtract has two extraction modes, selected automatically:

| Mode | When used | Cost | Speed |
|------|-----------|------|-------|
| **Student model** | `bioextract/model/trained/` directory exists with a trained model | Free (local CPU) | ~100-300ms/abstract |
| **Claude teacher** | No student model found, `ANTHROPIC_API_KEY` is set | ~$0.001/abstract (Haiku) | ~1-2s/abstract |

On first use without a trained student model, BioExtract falls back to the Claude
teacher. Once you train a student model (see [Training](#training-the-student-model)),
it switches to free local inference automatically.

### Entity types

GENE, PROTEIN, DISEASE, DRUG, CHEMICAL, PATHWAY, CELL_TYPE, TISSUE, ORGANISM,
BIOLOGICAL_PROCESS, MOLECULAR_FUNCTION, PHENOTYPE, VARIANT, ANATOMICAL_STRUCTURE

### Relationship types (directional, subject -> object)

activates, inhibits, upregulates, downregulates, associated_with, causes,
treats, increases_risk, decreases_risk, binds, phosphorylates, expressed_in,
located_in, regulates, interacts_with

---

## Dictionaries

Dictionaries map entity names and synonyms to canonical database IDs. They
enable normalization: "alpha-synuclein" -> gene:6622 (SNCA).

### Available sources

| Source | Content | License | Size |
|--------|---------|---------|------|
| **NCBI Gene** | Human gene symbols, aliases, descriptions | Open | ~60K entries |
| **MeSH** | Diseases, chemicals, anatomy, organisms | Open | ~30K entries |
| **Disease Ontology** | Disease classification with synonyms | Open | ~12K entries |
| **HPO** | Human phenotype terms | Open | ~18K entries |
| **ChEBI** | Chemical entities | Open | ~150K entries |
| **DrugBank** | Drug names, trade names | Requires login | Manual download |

### Commands

```bash
# Download all (skips DrugBank, which requires manual download)
biolink dictionaries download

# Download one source
biolink dictionaries download --source ncbi_gene
biolink dictionaries download --source mesh
biolink dictionaries download --source disease_ontology
biolink dictionaries download --source hpo
biolink dictionaries download --source chebi

# Build the unified index (run after downloading)
biolink dictionaries index

# Test lookups
biolink dictionaries lookup "Alzheimer's disease"
biolink dictionaries lookup TREM2
biolink dictionaries lookup "amyloid beta"
biolink dictionaries lookup TREM2 --type GENE
```

### DrugBank (manual)

DrugBank vocabulary requires a free account. Download from
https://go.drugbank.com/releases/latest#open-data and place the TSV at
`bioextract/dictionaries/data/drugbank.tsv` in this format:

```
canonical_id\tname\tentity_type\tsource_db\tsynonyms_json
```

Then re-run `biolink dictionaries index`.

### How the index works

Dictionary TSV files are merged into a single SQLite database
(`bioextract/dictionaries/data/dictionary.db`) with:

- **terms** table: canonical_id, name, entity_type, source_db
- **synonyms** table: canonical_id, synonym (one row per synonym)
- **term_search**: FTS5 virtual table for fast full-text search

Lookups use a cascade: exact name match -> synonym match -> FTS5 search -> LIKE
fallback.

---

## CLI Extraction

```bash
# Single text extraction
biolink extract "APOE4 increases risk of Alzheimer disease through amyloid aggregation"
```

Output:

```
Extraction method: llm_claude
Entities (3):
  [GENE] APOE4 -> gene:348 (APOE)
  [DISEASE] Alzheimer disease -> doid:10652 (Alzheimer's disease)
  [BIOLOGICAL_PROCESS] amyloid aggregation

Relationships (1):
  APOE4 --[increases_risk]--> Alzheimer disease (direction=positive)
```

Entities with `->` have been normalized to canonical IDs via the dictionary.
Entities without `->` were not found in the dictionary (normalization failed
or dictionary not available).

---

## API

Run the BioExtract service on port 8001:

```bash
uvicorn bioextract.api:app --host 127.0.0.1 --port 8001 --reload
```

Interactive docs at http://127.0.0.1:8001/docs.

### POST /extract

Extract entities and relationships from a single text.

```bash
curl -X POST http://127.0.0.1:8001/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "TREM2 reduces neuroinflammation in mouse microglia"}'
```

Response:

```json
{
  "entities": [
    {
      "text": "TREM2",
      "type": "GENE",
      "start": 0,
      "end": 5,
      "canonical_id": "gene:54209",
      "canonical_name": "TREM2",
      "confidence": 0.85
    },
    {
      "text": "neuroinflammation",
      "type": "BIOLOGICAL_PROCESS",
      "start": 14,
      "end": 31,
      "canonical_id": null,
      "canonical_name": null,
      "confidence": 0.85
    }
  ],
  "relationships": [
    {
      "subject": "TREM2",
      "predicate": "reduces",
      "object": "neuroinflammation",
      "type": "inhibits",
      "direction": "negative",
      "negated": false,
      "context": {
        "organism": "mouse",
        "cell_type": "microglia",
        "experiment_type": null
      },
      "confidence": 0.85
    }
  ],
  "extraction_method": "llm_claude"
}
```

### POST /extract/batch

Extract from multiple texts in one request.

```bash
curl -X POST http://127.0.0.1:8001/extract/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "APOE4 increases Alzheimer risk",
      "Donepezil inhibits acetylcholinesterase"
    ]
  }'
```

Returns a JSON array of extraction results (same schema as `/extract`).

### GET /dictionaries/lookup

Search the biomedical dictionary directly.

```bash
# Basic lookup
curl "http://127.0.0.1:8001/dictionaries/lookup?q=APOE"

# Filter by entity type
curl "http://127.0.0.1:8001/dictionaries/lookup?q=tau&type=GENE"

# Limit results
curl "http://127.0.0.1:8001/dictionaries/lookup?q=alzheimer&limit=5"
```

### GET /health

Check service status — reports whether the model and dictionary are available.

```bash
curl http://127.0.0.1:8001/health
```

---

## Entity Normalization

After extraction, entity mentions are resolved to canonical IDs through a
4-step pipeline:

1. **Exact match** — Case-insensitive lookup against term names and synonyms.
   "SNCA" matches gene:6622 directly.

2. **Abbreviation disambiguation** — Common biomedical abbreviations are
   resolved using surrounding context. "AD" near "amyloid" and "brain" resolves
   to Alzheimer's disease. "AD" near "skin" and "eczema" resolves to atopic
   dermatitis. Currently handles: AD, PD, MS, ALS.

3. **Full-text search** — FTS5 search across all names and synonyms.
   "alpha synuclein" finds gene:6622 (SNCA) via synonym match.

4. **Fuzzy match** — Strips common suffixes (gene, protein, receptor, disease,
   syndrome) and retries. "TREM2 gene" -> "TREM2" -> gene:54209.

Candidates are ranked by string similarity (Levenshtein + trigram) with bonuses
for entity type consistency and match quality (exact > synonym > FTS > fuzzy).

---

## Paper Ingestion

Fetch PubMed abstracts for dementia genes and store them in the BioLink
database:

```bash
# Fetch abstracts (default: last 5 years, top 50/gene for 13 genes)
biolink ingest-papers

# Customize
biolink ingest-papers --max-per-gene 10 --min-year 2023
```

This populates the `papers` table in `biolink.db`. To extract relationships
from ingested papers, run them through BioExtract (see
[End-to-End Pipeline](#end-to-end-pipeline)).

---

## Training the Student Model

Training replaces the Claude teacher with a free local model. The process uses
Google Colab's free T4 GPU.

### Prerequisites

- Google account (for Colab)
- BioRED dataset for evaluation (free from NCBI)
- `ANTHROPIC_API_KEY` for generating silver training labels

### Step 1: Prepare gold-standard data

Download and convert these human-annotated datasets to our unified JSONL
format. You don't need all three -- BioRED alone is enough to start. BC5CDR
and ChemProt add volume and entity-type diversity.

#### BioRED (recommended starting point)

600 abstracts with 6 entity types (gene, disease, chemical, variant, species,
cell line) and 8 relation types. Double-annotated with novelty labels. This is
the primary evaluation benchmark.

**Download:**
```bash
mkdir -p data/gold
cd data/gold

# BioC JSON format (easiest to parse)
curl -O https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip
unzip BIORED.zip
```

The ZIP contains `Train.BioC.JSON`, `Dev.BioC.JSON`, and `Test.BioC.JSON`.
Each is a BioC-format JSON file containing a collection of documents.

**Raw format** (BioC JSON):
```json
{
  "documents": [
    {
      "id": "11070156",
      "passages": [
        {"offset": 0, "text": "Title text here..."},
        {"offset": 87, "text": "Abstract text here..."}
      ],
      "annotations": [
        {
          "id": "0",
          "infons": {"identifier": "9606", "type": "Species"},
          "locations": [{"offset": 120, "length": 5}],
          "text": "human"
        },
        {
          "id": "1",
          "infons": {"identifier": "6622", "type": "GeneOrGeneProduct"},
          "locations": [{"offset": 200, "length": 4}],
          "text": "SNCA"
        }
      ],
      "relations": [
        {
          "id": "R0",
          "infons": {
            "type": "Association",
            "entity1": "GeneOrGeneProduct",
            "entity2": "DiseaseOrPhenotypicFeature",
            "novel": "Novel"
          },
          "nodes": [
            {"refid": "1", "role": "subject"},
            {"refid": "3", "role": "object"}
          ]
        }
      ]
    }
  ]
}
```

**BioRED entity types -> our types:**

| BioRED type | Our type |
|-------------|----------|
| GeneOrGeneProduct | GENE |
| DiseaseOrPhenotypicFeature | DISEASE |
| ChemicalEntity | CHEMICAL |
| SequenceVariant | VARIANT |
| OrganismTaxon | ORGANISM |
| CellLine | CELL_TYPE |

**BioRED relation types -> our types:**

| BioRED type | Our type | Direction |
|-------------|----------|-----------|
| Positive_Correlation | activates | positive |
| Negative_Correlation | inhibits | negative |
| Association | associated_with | neutral |
| Bind | binds | neutral |
| Comparison | associated_with | neutral |
| Cotreatment | associated_with | neutral |
| Drug_Interaction | interacts_with | neutral |
| Conversion | regulates | neutral |

**Conversion script** (save as `data/convert_biored.py`):
```python
import json
import sys

TYPE_MAP = {
    "GeneOrGeneProduct": "GENE",
    "DiseaseOrPhenotypicFeature": "DISEASE",
    "ChemicalEntity": "CHEMICAL",
    "SequenceVariant": "VARIANT",
    "OrganismTaxon": "ORGANISM",
    "CellLine": "CELL_TYPE",
}

REL_MAP = {
    "Positive_Correlation": ("activates", "positive"),
    "Negative_Correlation": ("inhibits", "negative"),
    "Association": ("associated_with", "neutral"),
    "Bind": ("binds", "neutral"),
    "Comparison": ("associated_with", "neutral"),
    "Cotreatment": ("associated_with", "neutral"),
    "Drug_Interaction": ("interacts_with", "neutral"),
    "Conversion": ("regulates", "neutral"),
}

def convert_biored(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    with open(output_path, "w") as out:
        for doc in data["documents"]:
            # Combine title + abstract
            text = " ".join(p["text"] for p in doc["passages"])
            base_offset = doc["passages"][0]["offset"]

            # Build annotation ID -> entity map
            ann_map = {}
            entities = []
            for ann in doc.get("annotations", []):
                etype = TYPE_MAP.get(ann["infons"]["type"])
                if not etype:
                    continue
                loc = ann["locations"][0]
                start = loc["offset"] - base_offset
                end = start + loc["length"]
                entity = {
                    "text": ann["text"],
                    "type": etype,
                    "start": start,
                    "end": end,
                }
                entities.append(entity)
                ann_map[ann["id"]] = entity

            # Build relationships
            relationships = []
            for rel in doc.get("relations", []):
                rel_type_raw = rel["infons"].get("type", "")
                mapped = REL_MAP.get(rel_type_raw)
                if not mapped:
                    continue
                our_type, direction = mapped

                subj_id = rel["nodes"][0]["refid"]
                obj_id = rel["nodes"][1]["refid"]
                subj = ann_map.get(subj_id)
                obj = ann_map.get(obj_id)
                if not subj or not obj:
                    continue

                relationships.append({
                    "subject": subj["text"],
                    "object": obj["text"],
                    "type": our_type,
                    "direction": direction,
                    "negated": False,
                    "context": {},
                })

            out.write(json.dumps({
                "text": text,
                "entities": entities,
                "relationships": relationships,
                "source": "biored",
            }) + "\n")

if __name__ == "__main__":
    convert_biored(sys.argv[1], sys.argv[2])
```

```bash
python data/convert_biored.py "data/gold/Train.BioC.JSON" data/gold/biored_train.jsonl
python data/convert_biored.py "data/gold/Dev.BioC.JSON" data/gold/biored_dev.jsonl
python data/convert_biored.py "data/gold/Test.BioC.JSON" data/gold/biored_test.jsonl
wc -l data/gold/biored_*.jsonl  # Should show ~400 train, ~100 dev, ~100 test
```

#### BC5CDR (adds chemical-disease volume)

1,500 abstracts annotated with chemical and disease entities plus
chemical-induced-disease (CID) relationships. PubTator format.

**Download:** Available through the original BioCreative site or NCBI mirrors.
Also on HuggingFace:
```bash
# Option A: HuggingFace (easiest)
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('bigbio/bc5cdr_corpus', trust_remote_code=True)
print(ds)  # train: 500, validation: 500, test: 500
"

# Option B: Direct download (PubTator format)
curl -O https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip
unzip CDR_Data.zip
```

**PubTator format** (if downloading directly):
```
25763772|t|Title text here.
25763772|a|Abstract text here.
25763772	0	8	Losartan	Chemical	MESH:D019808
25763772	19	31	hypertension	Disease	MESH:D006973
25763772	CID	MESH:D019808	MESH:D006973
```

Each block is separated by a blank line. Lines with `CID` are
chemical-induced-disease relationships. Entity lines are:
`PMID\tstart\tend\ttext\ttype\tID`.

**BC5CDR entity types -> our types:**

| BC5CDR type | Our type |
|-------------|----------|
| Chemical | CHEMICAL |
| Disease | DISEASE |

**Conversion** (PubTator to JSONL):
```python
import json
import sys

def convert_pubtator(input_path, output_path):
    with open(input_path) as f:
        content = f.read()

    with open(output_path, "w") as out:
        for block in content.strip().split("\n\n"):
            lines = block.strip().split("\n")
            title = abstract = ""
            entities = []
            relationships = []

            for line in lines:
                if "|t|" in line:
                    title = line.split("|t|", 1)[1]
                elif "|a|" in line:
                    abstract = line.split("|a|", 1)[1]
                elif "\tCID\t" in line:
                    parts = line.split("\t")
                    # CID lines: PMID\tCID\tchemical_id\tdisease_id
                    # We'll resolve names below
                    relationships.append({
                        "chem_id": parts[2], "dis_id": parts[3]
                    })
                elif "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        start, end = int(parts[1]), int(parts[2])
                        text_span = parts[3]
                        etype = "CHEMICAL" if parts[4] == "Chemical" else "DISEASE"
                        eid = parts[5]
                        entities.append({
                            "text": text_span, "type": etype,
                            "start": start, "end": end, "_id": eid,
                        })

            text = f"{title} {abstract}".strip()
            if not text:
                continue

            # Resolve CID relationships to entity names
            id_to_name = {e["_id"]: e["text"] for e in entities}
            resolved_rels = []
            for r in relationships:
                chem_name = id_to_name.get(r["chem_id"], r["chem_id"])
                dis_name = id_to_name.get(r["dis_id"], r["dis_id"])
                resolved_rels.append({
                    "subject": chem_name, "object": dis_name,
                    "type": "causes", "direction": "positive",
                    "negated": False, "context": {},
                })

            # Strip internal _id field
            clean_entities = [
                {"text": e["text"], "type": e["type"],
                 "start": e["start"], "end": e["end"]}
                for e in entities
            ]

            out.write(json.dumps({
                "text": text,
                "entities": clean_entities,
                "relationships": resolved_rels,
                "source": "bc5cdr",
            }) + "\n")

if __name__ == "__main__":
    convert_pubtator(sys.argv[1], sys.argv[2])
```

```bash
python data/convert_bc5cdr.py CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt data/gold/bc5cdr_train.jsonl
python data/convert_bc5cdr.py CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt data/gold/bc5cdr_dev.jsonl
python data/convert_bc5cdr.py CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt data/gold/bc5cdr_test.jsonl
```

#### ChemProt (adds chemical-protein interactions)

~2,300 abstracts with chemical-protein/gene interaction annotations.
13 relation types grouped into 5 evaluation classes (CPR:3 through CPR:9).

**Download:** Available on HuggingFace or the original site:
```bash
# HuggingFace (easiest)
python -c "
from datasets import load_dataset
ds = load_dataset('bigbio/chemprot', trust_remote_code=True)
print(ds)
"
```

The original data uses a TSV format with separate files for abstracts
(`_abstracts.tsv`), entities (`_entities.tsv`), and relations
(`_relations.tsv`). Each file is tab-separated.

**ChemProt relation groups -> our types:**

| ChemProt group | Description | Our type | Direction |
|----------------|-------------|----------|-----------|
| CPR:3 | Upregulator/activator | activates | positive |
| CPR:4 | Downregulator/inhibitor | inhibits | negative |
| CPR:5 | Agonist | activates | positive |
| CPR:6 | Antagonist | inhibits | negative |
| CPR:9 | Substrate/product | binds | neutral |

If using HuggingFace `datasets`, the conversion is straightforward since
the data is already structured. Map `type` fields using the table above.

#### Combining datasets

```bash
cat data/gold/biored_train.jsonl data/gold/bc5cdr_train.jsonl data/gold/chemprot_train.jsonl > data/gold/gold_train.jsonl
cat data/gold/biored_dev.jsonl data/gold/bc5cdr_dev.jsonl data/gold/chemprot_dev.jsonl > data/gold/gold_dev.jsonl

# Keep BioRED test separate -- always evaluate against BioRED gold only
cp data/gold/biored_test.jsonl data/gold/gold_test.jsonl

wc -l data/gold/gold_*.jsonl
```

Expected totals: ~3,500-4,000 gold examples (train + dev), ~100 test.

#### Target JSONL format reference

Every line in a JSONL file must be a single JSON object:

```json
{
  "text": "The full abstract or passage text.",
  "entities": [
    {"text": "TREM2", "type": "GENE", "start": 4, "end": 9},
    {"text": "Alzheimer's disease", "type": "DISEASE", "start": 45, "end": 64}
  ],
  "relationships": [
    {
      "subject": "TREM2",
      "object": "Alzheimer's disease",
      "type": "associated_with",
      "direction": "neutral",
      "negated": false,
      "context": {"organism": "human", "cell_type": null, "experiment_type": null}
    }
  ],
  "source": "biored"
}
```

Field notes:
- **start/end**: character offsets into `text`. The substring
  `text[start:end]` must equal the entity's `text` field.
- **type** (entity): one of the 14 BioExtract entity types (see
  [Entity types](#entity-types)).
- **type** (relationship): one of the 15 relationship types (see
  [Relationship types](#relationship-types)).
- **direction**: `positive`, `negative`, or `neutral`.
- **context**: all fields optional. Include when annotated in the source data.
- **source**: dataset origin (`biored`, `bc5cdr`, `chemprot`, `claude`).

### Step 2: Validate the teacher

Before using Claude to generate training data, check its quality against
BioRED gold labels. Run Claude on the BioRED abstracts using the teacher
prompt and compare outputs to gold annotations. Target: F1 >= 85%.

```python
from bioextract.model.teacher_prompt import build_extraction_prompt, parse_teacher_output

# For each BioRED abstract:
messages = build_extraction_prompt(abstract_text)
# Call Claude API with messages, then:
result = parse_teacher_output(response_text)
# Compare result to gold annotations
```

### Step 3: Generate silver training data

Run the validated teacher on 5-10K diverse PubMed abstracts. Use Claude Haiku
for cost efficiency (~$5-15 for 10K abstracts).

```python
import json
import httpx

# For each abstract:
messages = build_extraction_prompt(abstract_text, known_entities=dict_hints)
response = call_claude_haiku(messages)  # Your Claude API wrapper
parsed = parse_teacher_output(response)

# Write to JSONL
with open("silver_labels.jsonl", "a") as f:
    f.write(json.dumps({
        "text": abstract_text,
        "entities": parsed["entities"],
        "relationships": parsed["relationships"],
        "source": "claude"
    }) + "\n")
```

Post-process: validate extracted entities against dictionaries, discard examples
where parsing failed. Manually review ~100 samples for quality.

### Step 4: Combine and train

Merge gold + silver data into one JSONL file, then train on Colab:

```bash
cat biored.jsonl bc5cdr.jsonl chemprot.jsonl silver_labels.jsonl > training.jsonl
```

In a Colab notebook:

```python
!pip install transformers datasets peft accelerate

# Upload training.jsonl to Colab, then:
import logging
logging.basicConfig(level=logging.INFO)

from bioextract.model.train import train

train(
    data_path="training.jsonl",
    output_dir="bioextract_v1",
    model_name="scifive",   # SciFive-base, pretrained on PubMed
    epochs=5,
    batch_size=4,
    learning_rate=3e-4,
    use_lora=True,           # Fits in T4 16GB VRAM
)
```

Training takes ~2-4 hours on a T4 for ~10K examples. Checkpoints are saved
each epoch.

**Model options:**

| Model | ID | Notes |
|-------|----|-------|
| **SciFive-base** (default) | `razent/SciFive-base-Pubmed_PMC` | Best biomedical performance at this size |
| **Flan-T5-base** (fallback) | `google/flan-t5-base` | Better generalization if SciFive underperforms |

### Step 5: Evaluate

Always evaluate against the gold-standard BioRED hold-out set (not teacher
labels):

```python
from bioextract.model.train import evaluate_on_biored

metrics = evaluate_on_biored("bioextract_v1", "biored_test.jsonl")
print(metrics)
# {
#   "entity_f1": 0.87,
#   "relationship_f1": 0.76,
#   "direction_accuracy": 0.82,
#   "entity_counts": {"tp": 340, "fp": 28, "fn": 42},
#   "relationship_counts": {"tp": 180, "fp": 32, "fn": 55}
# }
```

Targets: entity F1 > 0.85, relationship F1 > 0.75.

### Step 6: Deploy the trained model

Copy the model files to `bioextract/model/trained/pytorch_model/`:

```bash
# From Colab, download the model directory, then:
mkdir -p bioextract/model/trained/pytorch_model
cp bioextract_v1/* bioextract/model/trained/pytorch_model/
```

BioExtract detects the model on next startup and switches from Claude teacher
to local inference automatically.

### Step 7 (optional): ONNX export for faster CPU inference

Quantize and convert to ONNX for ~5x speedup on CPU:

```python
# pip install optimum[onnxruntime]
from optimum.exporters.onnx import main_export

main_export(
    "bioextract/model/trained/pytorch_model",
    output="bioextract/model/trained/",
    task="seq2seq-lm",
)
```

This creates `bioextract/model/trained/model.onnx`, which BioExtract prefers
over the PyTorch model when both exist.

---

## End-to-End Pipeline

Full workflow from PubMed abstracts to populated knowledge graph:

```bash
# 1. Set up dictionaries (one-time)
biolink dictionaries download
biolink dictionaries index

# 2. Fetch papers
biolink ingest-papers --max-per-gene 20 --min-year 2022

# 3. Start the BioExtract service
uvicorn bioextract.api:app --port 8001 &

# 4. Run extraction pipeline (from Python)
```

```python
import sqlite3
import json
from pipeline.sources.pubmed_abstracts import PubMedAbstracts
from pipeline.sources.bioextract import BioExtractSource
from pipeline.build_db import (
    insert_entities, insert_relationships,
    insert_papers, insert_evidence_items, SCHEMA,
)

# Connect to existing DB
conn = sqlite3.connect("biolink.db")
conn.executescript(SCHEMA)

# Load papers already in the DB
rows = conn.execute("SELECT id, abstract, title FROM papers").fetchall()
abstracts = [{"paper_id": r[0], "abstract": r[1], "title": r[2]} for r in rows]

# Extract via BioExtract service
source = BioExtractSource(abstracts=abstracts)
source.fetch()

# Insert extracted entities and relationships
entities, relationships = source.parse()
n_ent = insert_entities(conn, entities)
n_rel = insert_relationships(conn, relationships)
print(f"Inserted {n_ent} entities, {n_rel} relationships")

# Insert evidence items
evidence = source.get_evidence_items()
n_ev = insert_evidence_items(conn, evidence)
print(f"Inserted {n_ev} evidence items")

conn.close()
```

```bash
# 5. Start the BioLink Hub and browse results
biolink serve
# Open http://127.0.0.1:8000
```

---

## Python API

Use BioExtract directly in Python without the HTTP service:

```python
from bioextract.extract import BioExtractor

extractor = BioExtractor()

# Single extraction
result = extractor.extract("TREM2 reduces neuroinflammation in mouse microglia")

for entity in result.entities:
    print(f"  {entity.type}: {entity.text} -> {entity.canonical_id}")

for rel in result.relationships:
    print(f"  {rel.subject} --[{rel.type}]--> {rel.object}")
    print(f"    direction={rel.direction}, organism={rel.context.organism}")

# Batch extraction
results = extractor.extract_batch([
    "APOE4 increases Alzheimer risk",
    "Donepezil inhibits acetylcholinesterase",
])

# Check status
print(extractor.status)
# {"model": "claude_teacher", "model_available": false,
#  "dictionary_available": true, "dictionary_stats": {...}}
```

### Dictionary lookup

```python
from bioextract.dictionaries.lookup import DictionaryLookup

dl = DictionaryLookup()

# Exact match
matches = dl.exact_match("APOE", entity_type="GENE")
for m in matches:
    print(f"  {m.canonical_id}: {m.name} ({m.entity_type}) score={m.score}")

# Full-text search
matches = dl.search("alzheimer", limit=5)

# Stats
print(dl.stats())
# {"total_terms": 270000, "by_type": {"GENE": 60000, ...}, "by_source": {...}}
```

### Normalization

```python
from bioextract.normalize import EntityNormalizer, disambiguate_abbreviation
from bioextract.schema import ExtractedEntity

normalizer = EntityNormalizer()

entity = ExtractedEntity(text="SNCA", type="GENE", start=0, end=4)
normalizer.normalize_entity(entity, context_text="SNCA aggregation in neurons")
print(entity.canonical_id)    # gene:6622
print(entity.canonical_name)  # SNCA

# Abbreviation disambiguation
result = disambiguate_abbreviation("AD", "amyloid plaques in brain neurons")
print(result)  # "Alzheimer's disease"

result = disambiguate_abbreviation("AD", "topical treatment for skin eczema")
print(result)  # "atopic dermatitis"
```

---

## File Structure

```
bioextract/
  __init__.py
  schema.py              # Entity/relationship type enums + dataclasses
  extract.py             # Main orchestrator (model + dictionary + normalization)
  normalize.py           # 4-step entity normalization pipeline
  api.py                 # FastAPI service (extract, batch, lookup, health)
  Dockerfile             # Container deployment
  requirements.txt       # Python dependencies
  model/
    __init__.py
    teacher_prompt.py    # Claude extraction prompt + output parser
    inference.py         # Student model loading + inference
    train.py             # Colab training script + BioRED evaluation
    trained/             # (created after training)
      pytorch_model/     # HuggingFace model files
      model.onnx         # (optional) ONNX-optimized model
  dictionaries/
    __init__.py
    download.py          # Download scripts for each dictionary source
    index.py             # Build unified SQLite FTS5 index
    lookup.py            # Query interface (exact, search, fuzzy)
    data/                # (created after download)
      ncbi_gene.tsv
      mesh.tsv
      disease_ontology.tsv
      hpo.tsv
      chebi.tsv
      drugbank.tsv       # (manual download)
      dictionary.db      # (created by index command)
```

---

## Confidence Tiers

Extracted data carries confidence scores that reflect provenance:

| Tier | Range | Source | Display |
|------|-------|--------|---------|
| High | 0.9 - 1.0 | Curated databases (DisGeNET, DGIdb) | Green badge |
| Medium | 0.5 - 0.8 | NLP-extracted (BioExtract student or teacher) | Yellow badge |
| Low | 0.1 - 0.4 | Computed hypotheses (drug repurposing, path inference) | Red badge |

The student model assigns 0.7 baseline confidence. The Claude teacher assigns
0.85. Normalization can reduce confidence when matches are fuzzy.

---

## Docker

Build and run the BioExtract service in a container:

```bash
cd bioextract
docker build -t bioextract .
docker run -p 8001:8001 -e ANTHROPIC_API_KEY=sk-ant-... bioextract
```

To include pre-built dictionaries, uncomment the download/index lines in the
Dockerfile, or mount the `dictionaries/data/` directory:

```bash
docker run -p 8001:8001 \
  -v $(pwd)/bioextract/dictionaries/data:/app/dictionaries/data \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  bioextract
```
