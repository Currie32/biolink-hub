"""Claude teacher prompt for biomedical entity/relationship extraction.

Used to generate silver-standard training labels for the student model.
Run validated Claude prompt on abstracts -> structured JSON output.
"""

import json

ENTITY_TYPES = [
    "GENE", "DISEASE", "CHEMICAL", "PATHWAY",
    "CELL_TYPE", "ORGANISM", "VARIANT",
]

RELATIONSHIP_TYPES = [
    "activates", "inhibits", "upregulates", "downregulates",
    "associated_with", "causes", "treats", "increases_risk",
    "decreases_risk", "binds", "phosphorylates", "expressed_in",
    "located_in", "regulates", "interacts_with",
]

SYSTEM_PROMPT = """You are a biomedical named entity recognition and relation extraction system.
Given a biomedical abstract, extract entities and relationships.

## Entity types
{entity_types}

## Type definitions and rules

GENE: Genes AND gene products (proteins, receptors, enzymes, channels). Use GENE for all of these.
  - "TREM2" -> GENE. "Na(v)1.5 cardiac sodium channel" -> GENE (use short name "Na(v)1.5").
  - "beta-1 adrenoceptor" -> GENE. "beta-2 receptor" -> GENE. "mu opioid receptor" -> GENE.
  - Extract the gene/receptor name EXACTLY as written in the text. If the text says "beta-2 receptor", extract "beta-2 receptor", not "beta-2 adrenergic receptor".
  - Extract BOTH short symbol and full name when both appear: "CBR3" AND "carbonyl reductase 3".
  - Do NOT create a separate PROTEIN type. Proteins are GENE.

DISEASE: Diseases, disorders, syndromes, AND clinical signs/symptoms.
  - "Alzheimer disease" -> DISEASE. "bradycardia" -> DISEASE. "tachycardia" -> DISEASE.
  - "long QT syndrome" -> DISEASE. "congestive heart failure" -> DISEASE.
  - "pain" -> DISEASE (not "pain perception"). "hypokalemia" -> DISEASE.
  - "cardiotoxic" -> DISEASE (use the exact adjective form from the text, not "cardiotoxicity").
  - "cancer" -> DISEASE. Always extract standalone disease terms even when also mentioned in qualified forms.
  - Keep compound disease names intact: "hepatocerebral mitochondrial DNA depletion syndrome" is ONE entity, not split.
  - "[Gene name] deficiency" -> DISEASE. Example: "Deoxyguanosine kinase (dGK) deficiency" is a DISEASE, not a GENE.
  - Do NOT use PHENOTYPE. All clinical conditions are DISEASE.

CHEMICAL: Drugs, chemicals, metabolites, ions, and small molecules.
  - "donepezil" -> CHEMICAL. "lidocaine" -> CHEMICAL. "sodium" -> CHEMICAL.
  - "doxorubicin" -> CHEMICAL. "ATP" -> CHEMICAL. "anthracycline" -> CHEMICAL.
  - Extract ions and elements: "potassium" -> CHEMICAL. "sodium" -> CHEMICAL.
  - Extract drug classes when they are the subject of relationships: "anthracycline" or "anthracyclines" -> CHEMICAL.
  - Do NOT use DRUG. All drugs are CHEMICAL.

VARIANT: Sequence variants. Extract the COMPLETE standard notation.
  - "V1763M" -> VARIANT (the complete substitution, not fragments like "V", "1763", or "M").
  - "A118G" -> VARIANT. "N40D" -> VARIANT. "rs1799971" -> VARIANT.
  - "c.444-62C>A" -> VARIANT. "NQO1*2" -> VARIANT.
  - Partial allele designations when used as entities: "A118" -> VARIANT, "G118" -> VARIANT.
  - Do NOT extract partial components (individual amino acids, codons, nucleotides) as VARIANT.
  - Do NOT extract "GTG", "ATG", "G-->A" as separate variants if they are part of a larger variant description.
  - CRITICAL: Do NOT extract nucleotides/nucleosides (adenosine, guanosine, cytidine, thymidine) as CHEMICAL when they appear in variant descriptions like "adenosine, guanosine, cytidine, and thymidine in position 118".

ORGANISM: Species only when specifically relevant (e.g., "mouse", "rat", "Drosophila").
  - "Chinese hamster" -> ORGANISM (not "Chinese hamster ovary cells" — that's a cell line).
  - Do NOT extract "patient", "patients", "families", "subjects" as ORGANISM.
  - Do NOT extract "human" or "humans" — these are almost always referring to the study population, not a biologically relevant entity.

CELL_TYPE: Specific cell types and cell lines (e.g., "microglia", "HEK293", "tsA201").
  - "Chinese hamster ovary cells" -> CELL_TYPE (but extract "Chinese hamster" separately as ORGANISM).

PATHWAY: Named biological pathways only (e.g., "NF-kB signaling pathway", "mTOR pathway").
  - Do NOT extract generic biological processes, molecular functions, or phenotypic descriptions.

## What NOT to extract
- Descriptive phrases: "activation curve", "inactivation kinetics", "extensive metabolizers", "poor metabolizers"
- Experimental methods: "site-directed mutagenesis", "DNA sequencing", "mutational analysis"
- Experimental reagents/tools: chemicals used as laboratory tools (e.g., "actinomycin D" used to block transcription) — only extract if they are the subject of a biological relationship, not when they are just experimental tools
- Generic terms: "gene", "protein", "drug", "disease", "mutation", "DNA", "RNA", "mRNA"
- Generic chemical classes: "beta-blocker", "opioid drugs", "cancer therapies", "purine deoxyribonucleosides", "nucleosides", "nucleotides", "amino acids", "peptides"
- Amino acids/nucleosides mentioned in variant descriptions: do NOT extract "valine", "methionine", "adenosine", "guanosine", "cytidine", "thymidine" etc. when they describe a substitution or variant position
- Qualified/compound disease phrases: do NOT extract "anthracycline-related CHF" or "fetal bradycardia" — extract the core disease ("CHF", "bradycardia") separately
- Disease subtype notations: do NOT extract "LQTS-3", "LQTS-2", etc. as separate entities — extract "LQTS" or "long QT syndrome" instead
- Vague/modified disease terms: do NOT extract "pain perception", "addiction" (too vague) — extract "pain" or "drug addiction" (the specific clinical term)
- Tissues, anatomical structures, or organisms like "mammalian", "patient", "patients"

## Deduplication
- Extract each unique entity ONCE using its most common name. Also extract well-known abbreviations as separate entities.
- Example: extract BOTH "long QT syndrome" AND "LQTS", but do NOT also extract "congenital long QT syndrome" (qualified variant).
- Example: extract BOTH "congestive heart failure" AND "CHF".
- Example: extract BOTH "carbonyl reductase 3" AND "CBR3", BOTH "NAD(P)H:quinone oxidoreductase 1" AND "NQO1".
- Extract singular forms for chemicals: "anthracycline" not "anthracyclines" (unless only the plural appears).
- Do NOT extract qualified/modified versions (e.g., "anthracycline-related CHF", "childhood cancer", "fetal bradycardia").

## Relationship types (directional: subject -> object)
{relationship_types}

## Relationship rules

- In subject and object fields, use the EXACT entity text from your entities list. Do not paraphrase or use compound phrases.
- Do NOT create relationships using entity text that is not in your entities list.
- Be THOROUGH: extract ALL relationships where the text states or directly implies a connection. For drug→disease and gene→disease relationships, extract EVERY combination the text supports.

### What is NOT a valid relationship (do NOT extract these):
- Background/world knowledge: Do NOT extract relationships based on what you KNOW about biology, only what the TEXT SAYS. Example: do NOT extract "lidocaine inhibits Na(v)1.5" unless the text explicitly says lidocaine acts on/blocks/inhibits Na(v)1.5. Pharmacological knowledge alone is not sufficient.
- Electrophysiological properties: "tetrodotoxin-sensitive current" or "lidocaine-resistant current" describe channel PROPERTIES from electrophysiology experiments, NOT therapeutic drug→gene relationships. Do NOT extract "TTX inhibits Na(v)1.5" from such descriptions.
- Study context co-occurrence: If a gene/variant is studied IN patients with a disease, that does NOT mean the gene causes or is associated with the disease. Example: "polymorphisms in NQO1 in patients who had childhood cancer" does NOT mean NQO1 is associated_with cancer — cancer is the patient population, not the outcome being studied. Only extract gene→disease if the gene is directly implicated in the disease mechanism or risk.
- Experimental tools: If a chemical (e.g., "actinomycin D") is used as an EXPERIMENTAL REAGENT or method tool in the study design, do NOT create a therapeutic/functional relationship between it and other entities. These are methodology, not biology.
- Substrate/enzyme assay: If a chemical is used as a SUBSTRATE in an enzyme assay (e.g., "enzyme activity assays with doxorubicin"), this is methodology — do NOT extract it as a therapeutic interaction. Only extract gene↔chemical if the text describes a biological/pharmacological mechanism.
- Transitive inference: If A relates to B and B relates to C, do NOT extract A→C unless the text directly states it. Example: if NQO1 relates to anthracyclines and anthracyclines cause CHF, do NOT extract NQO1→CHF. Similarly, if CBR3 metabolizes doxorubicin into doxorubicinol and doxorubicinol causes CHF, do NOT extract CBR3→CHF.
- Class membership: "doxorubicin is an anthracycline" is taxonomy, NOT a biological relationship.

### Key relationship patterns to extract (be exhaustive with these):
1. DRUG treats/controls DISEASE: If a drug is described as treating, controlling, or effective against a disease, extract drug downregulates disease. Extract for EVERY disease the drug treats, including the overall syndrome AND each specific symptom/manifestation.
   - CRITICAL: When a drug treats symptoms of a named syndrome, extract drug→syndrome AS WELL AS drug→each symptom. Example: if lidocaine and mexiletine control arrhythmias that are manifestations of LQTS, extract: lidocaine downregulates LQTS, lidocaine downregulates arrhythmias, mexiletine downregulates LQTS, mexiletine downregulates arrhythmias.
   - If mexiletine also controlled symptoms, extract mexiletine→disease for EVERY disease it controlled.
2. GENE/VARIANT associated_with DISEASE: If a gene or variant is discussed in the context of a disease, extract the association for EVERY disease.
   - "SCN5A mutation causes long QT syndrome with bradycardia and tachycardia" -> SCN5A associated_with LQTS AND SCN5A associated_with bradycardia AND SCN5A associated_with tachycardia
   - When a mutation manifests as a syndrome, extract gene→syndrome AND gene→each symptom.
3. VARIANT associated_with DISEASE: When the text attributes a disease to a specific variant, ALWAYS extract variant→disease for each disease manifestation.
4. DRUG/CHEMICAL associated_with GENE: If a drug's mechanism involving a gene/receptor is stated.
5. CHEMICAL associated_with CHEMICAL: If two chemicals are described as being related, co-administered, or pharmacologically linked.
   - "metoprolol is under genetic control of the debrisoquine/sparteine type" -> metoprolol associated_with debrisoquine AND metoprolol associated_with sparteine
6. GENE associated_with GENE: If a gene encodes, regulates, or interacts with another gene/protein.
7. CHEMICAL upregulates/downregulates DISEASE: If the text states a chemical causes or worsens a disease, use upregulates. If it treats or reduces, use downregulates.
   - "anthracycline-related CHF" -> anthracycline upregulates CHF
8. When a VARIANT is in a GENE and that GENE is associated with a DISEASE, extract BOTH: variant associated_with disease AND gene associated_with disease.

### Additional rules:
- X and Y merely mentioned in the same sentence with no stated connection -> NO RELATIONSHIP
- Prefer "associated_with" when the relationship type is unclear or when describing genetic associations/risk factors.
- Use "upregulates" for positive correlations (increases, causes, activates, risk factor for).
- Use "downregulates" for negative correlations (decreases, treats, inhibits, controls, reduces risk of).
- Mark direction: "positive" (activates/upregulates/causes/increases_risk), "negative" (inhibits/downregulates/decreases_risk/treats), "neutral" (associated_with/binds/expressed_in/located_in/interacts_with).
- Set negated=true if the text explicitly negates the relationship.
- Include context when mentioned: organism, cell_type, experiment_type (in_vitro/in_vivo/clinical/computational).

## Output format
Return ONLY valid JSON:
```json
{{{{
  "entities": [
    {{{{"text": "TREM2", "type": "GENE", "start": 0, "end": 5}}}}
  ],
  "relationships": [
    {{{{
      "subject": "TREM2",
      "predicate": "reduces",
      "object": "neuroinflammation",
      "type": "inhibits",
      "direction": "negative",
      "negated": false,
      "context": {{{{"organism": "mouse", "cell_type": "microglia", "experiment_type": "in_vivo"}}}}
    }}}}
  ]
}}}}
```""".format(
    entity_types=", ".join(ENTITY_TYPES),
    relationship_types=", ".join(RELATIONSHIP_TYPES),
)


def build_extraction_prompt(abstract: str, known_entities: list[dict] | None = None) -> list[dict]:
    """Build Claude API messages for extracting entities/relationships from an abstract.

    Args:
        abstract: The biomedical abstract text.
        known_entities: Optional list of known entities from dictionary lookup,
            each with keys: name, type, canonical_id. Helps the model output
            consistent entity types.

    Returns:
        List of message dicts for Claude API (system + user message).
    """
    user_content = f"Extract all biomedical entities and relationships from this abstract:\n\n{abstract}"

    if known_entities:
        hints = "\n".join(
            f"- {e['name']} ({e['type']}, ID: {e.get('canonical_id', 'unknown')})"
            for e in known_entities
        )
        user_content += (
            f"\n\nKnown entities that may appear in this text "
            f"(use these types/IDs when they match):\n{hints}"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_teacher_output(response_text: str) -> dict:
    """Parse Claude's JSON response into structured extraction data.

    Returns dict with 'entities' and 'relationships' lists.
    Handles common issues like markdown code blocks wrapping JSON.
    """
    text = response_text.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": [], "parse_error": text[:200]}

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    # Build set of entity texts for relationship validation
    entity_texts = {e.get("text", "").lower() for e in entities}

    # Basic validation: filter malformed entries
    valid_entities = []
    for e in entities:
        if all(k in e for k in ("text", "type", "start", "end")):
            # Remap removed types to their replacements
            etype = e["type"]
            if etype == "PROTEIN":
                etype = "GENE"
            elif etype == "DRUG":
                etype = "CHEMICAL"
            elif etype == "PHENOTYPE":
                etype = "DISEASE"
            elif etype in ("BIOLOGICAL_PROCESS", "MOLECULAR_FUNCTION",
                           "TISSUE", "ANATOMICAL_STRUCTURE"):
                continue  # Skip types we no longer support
            e["type"] = etype
            if etype in ENTITY_TYPES:
                valid_entities.append(e)

    # Build fuzzy lookup: for each relationship entity text, find closest entity match
    def _find_entity_match(text: str) -> str | None:
        """Find matching entity text, with fuzzy fallback."""
        t = text.lower()
        if t in entity_texts:
            return t
        # Try stripping parentheses: "Nav1.5" matches "Na(v)1.5"
        for et in entity_texts:
            if et.replace("(", "").replace(")", "") == t.replace("(", "").replace(")", ""):
                return et
        # Try substring match: "cardiac sodium channel" in "Na(v)1.5 cardiac sodium channel"
        for et in entity_texts:
            if t in et or et in t:
                return et
        return None

    valid_rels = []
    for r in relationships:
        if all(k in r for k in ("subject", "object", "type")):
            if r["type"] in RELATIONSHIP_TYPES:
                # Verify subject and object are in entity list (with fuzzy matching)
                subj_match = _find_entity_match(r["subject"])
                obj_match = _find_entity_match(r["object"])
                if subj_match and obj_match:
                    if "context" not in r or not isinstance(r.get("context"), dict):
                        r["context"] = {}
                    valid_rels.append(r)

    return {"entities": valid_entities, "relationships": valid_rels}
