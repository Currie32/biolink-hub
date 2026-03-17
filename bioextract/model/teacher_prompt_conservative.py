"""Conservative extraction prompt for Haiku — precision over recall.

This prompt is deliberately stripped down compared to the Sonnet prompt.
It extracts only what is EXPLICITLY stated in the text, avoiding:
- Implied/inferred relationships
- Mechanistic relationships not directly stated
- Compound/qualified entity forms

The goal is low FP rate. The ensemble merges this with higher-recall models.
"""

import json

ENTITY_TYPES = [
    "GENE", "DISEASE", "CHEMICAL", "CELL_TYPE", "ORGANISM", "VARIANT",
]

RELATIONSHIP_TYPES = [
    "activates", "inhibits", "upregulates", "downregulates",
    "associated_with", "causes", "treats", "increases_risk",
    "decreases_risk", "binds", "regulates", "interacts_with",
]

SYSTEM_PROMPT = """You are a precise biomedical entity and relationship extractor.
Extract ONLY entities and relationships that are EXPLICITLY stated in the text.

## Entity types
{entity_types}

## Rules
- GENE: genes, proteins, receptors, enzymes, channels. Extract both symbol and full name if present.
- DISEASE: diseases, disorders, syndromes, symptoms. Use exact text form.
- CHEMICAL: drugs, chemicals, metabolites, ions.
- VARIANT: sequence variants (e.g., "V1763M", "A118G", "rs1799971", "c.444-62C>A").
- ORGANISM: species (e.g., "mouse", "rat"). Do NOT extract "patient", "human".
- CELL_TYPE: cell types and cell lines.

## CRITICAL extraction constraints
- Extract each entity ONCE using its exact text form.
- Do NOT extract descriptive phrases, methods, or generic terms.
- Do NOT extract amino acids or nucleotides from variant descriptions.
- For relationships: ONLY extract if the text EXPLICITLY states the connection.
- Do NOT infer relationships from co-occurrence alone.
- Do NOT extract mechanistic relationships unless the text directly states them.
- When uncertain, do NOT extract — precision matters more than recall.

## Relationship types
{relationship_types}

## Relationship constraints
- Subject and object MUST be exact entity text from your entity list.
- ONLY extract if a verb, adjective, or explicit phrase connects the two entities.
- "X treats Y", "X causes Y", "X is associated with Y" -> extract.
- "X and Y were studied" -> do NOT extract (co-occurrence, not a relationship).
- "X-related Y" -> extract as associated_with.
- Use "associated_with" when the connection is stated but the type is unclear.

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
      "object": "neuroinflammation",
      "type": "inhibits",
      "direction": "negative",
      "negated": false
    }}}}
  ]
}}}}
```""".format(
    entity_types=", ".join(ENTITY_TYPES),
    relationship_types=", ".join(RELATIONSHIP_TYPES),
)


def build_conservative_prompt(abstract: str) -> list[dict]:
    """Build Claude API messages for conservative extraction."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract entities and relationships:\n\n{abstract}"},
    ]


def parse_conservative_output(response_text: str) -> dict:
    """Parse conservative model output. Same logic as main parser but simpler validation."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": [], "parse_error": text[:200]}

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    entity_texts = {e.get("text", "").lower() for e in entities}

    valid_entities = []
    for e in entities:
        if all(k in e for k in ("text", "type")):
            etype = e["type"]
            if etype == "PROTEIN":
                etype = "GENE"
            elif etype == "DRUG":
                etype = "CHEMICAL"
            elif etype == "PHENOTYPE":
                etype = "DISEASE"
            e["type"] = etype
            if etype in ENTITY_TYPES:
                # Ensure start/end exist
                if "start" not in e:
                    e["start"] = 0
                if "end" not in e:
                    e["end"] = 0
                valid_entities.append(e)

    valid_rels = []
    for r in relationships:
        if all(k in r for k in ("subject", "object", "type")):
            if r["type"] in RELATIONSHIP_TYPES:
                subj = r["subject"].lower()
                obj = r["object"].lower()
                if subj in entity_texts and obj in entity_texts:
                    valid_rels.append(r)

    return {"entities": valid_entities, "relationships": valid_rels}
