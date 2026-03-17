# BioLink Hub -- Progress

## Milestone 1: Project Setup + First Data Source
**Status**: Complete

- [x] Project scaffolding: `pyproject.toml`, CLI (Click), project structure
- [x] `PROGRESS.md` tracking file
- [x] Abstract `Source` base class
- [x] SQLite schema: `entities`, `relationships`, `entity_search` FTS5
- [x] `ncbi_gene.py` source: fetch gene data for dementia-related genes
- [x] `build_db.py`: assemble SQLite from source output
- [x] REST API: `/api/search`, `/api/entity/{id}`, `/api/stats`
- [x] Frontend: search + entity detail views
- [x] **Verify**: Query SQLite for SNCA, APOE, MAPT -- all present with descriptions, synonyms, external IDs

**Results**: 14 gene entities (13 target genes + 1 LRRK2 read-through variant). GBA renamed to GBA1 per NCBI. Batch E-utilities queries (2 API calls total). CLI has `build`, `search`, and `info` commands.

## Milestone 2: BioExtract -- Biomedical Extraction Service
**Status**: Code complete, awaiting dictionary download + model training

### Architecture
`bioextract/` standalone service with:
- **Extraction engine** (`extract.py`) -- orchestrates model inference, dictionary matching, normalization
- **Student model** (`model/inference.py`) -- SciFive-base seq2seq, falls back to Claude teacher
- **Teacher prompt** (`model/teacher_prompt.py`) -- Claude prompt for labeled data generation
- **Training script** (`model/train.py`) -- HuggingFace + LoRA fine-tuning for Colab
- **Evaluation** (`model/train.py:evaluate_on_biored`) -- entity F1, relationship F1, direction accuracy
- **Dictionary layer** (`dictionaries/`) -- NCBI Gene, MeSH, Disease Ontology, HPO, ChEBI, DrugBank
- **Normalization** (`normalize.py`) -- exact match, fuzzy match, abbreviation disambiguation
- **API** (`api.py`) -- FastAPI service: `/extract`, `/extract/batch`, `/dictionaries/lookup`, `/health`

### Completed
- [x] 2a: Dictionary download scripts (NCBI Gene, MeSH, DO, HPO, ChEBI, DrugBank placeholder)
- [x] 2a: SQLite-based unified dictionary index with FTS5
- [x] 2a: Dictionary lookup with exact match, synonym match, FTS search
- [x] 2b: Claude teacher prompt with entity/relationship extraction schema
- [x] 2b: Teacher output parsing with validation
- [x] 2c/2d: Training data format (JSONL) + Colab-ready training script with LoRA
- [x] 2d: BioRED evaluation harness (entity F1, relationship F1, direction accuracy)
- [x] 2e: Normalization pipeline (exact, fuzzy, abbreviation disambiguation, candidate ranking)
- [x] 2f: FastAPI service with extract/batch/lookup/health endpoints
- [x] 2f: Dockerfile for deployment
- [x] 2g: BioLink integration source (`pipeline/sources/bioextract.py`)

### Remaining Steps
- [ ] 2a: Actually download dictionaries (`biolink dictionaries download`)
- [ ] 2a: Verify: "Alzheimer's", "AD", "Alzheimer disease", "senile dementia" all resolve to same canonical ID
- [ ] 2b: Validate Claude teacher against BioRED gold-standard (target: F1 >= 85%)
- [ ] 2c: Download BioRED, BC5CDR, ChemProt gold datasets, convert to unified JSONL
- [ ] 2c: Run Claude teacher on 5-10K PubMed abstracts for silver labels
- [ ] 2d: Train student model on Colab (SciFive-base + LoRA, ~2-4 hours)
- [ ] 2d: Evaluate on BioRED hold-out (target: >85% entity F1, >75% relationship F1)
- [ ] 2f: Verify: send 100 unseen abstracts, measure recall against PubTator

## Milestone 3: Paper Ingestion + Evidence Model
**Status**: Code complete, awaiting M2 dictionary/model + paper fetch

### Completed
- [x] Schema: `papers` and `evidence_items` tables with indexes
- [x] `pipeline/sources/pubmed_abstracts.py` -- PubMed E-utilities fetcher
- [x] `build_db.py` -- `insert_papers()`, `insert_evidence_items()` functions
- [x] API: `/api/entity/{id}` includes evidence counts, paper counts
- [x] API: `/api/entity/{id}/evidence`, `/api/relationship/{id}/evidence` endpoints
- [x] Frontend: confidence badges (high/medium/low tiers)
- [x] Frontend: expandable evidence per relationship (paper citations with metadata)
- [x] CLI: `biolink ingest-papers` command

### Remaining Steps
- [ ] Actually ingest papers (`biolink ingest-papers`)
- [ ] Run papers through BioExtract to populate relationships + evidence
- [ ] Verify: Search APOE -> see diseases, drugs, processes from papers

## Milestone 4: Structured Sources (Diseases + Drugs)
**Status**: Not started

## Milestone 5: Contradiction Detection
**Status**: Not started

## Milestone 6: Pathways + Mechanism Explorer
**Status**: Not started

## Milestone 7: Drug Repurposing Hypotheses
**Status**: Not started

## Milestone 8: Variants + Proteins
**Status**: Not started

## Milestone 9: Graph Visualization
**Status**: Not started

## Milestone 10: Deployment
**Status**: Not started

## Milestone 11: Expand Beyond Dementia
**Status**: Not started
