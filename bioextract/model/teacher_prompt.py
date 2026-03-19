from __future__ import annotations

"""Entity gap-fill prompt for Sonnet — Stage 1c of the teacher pipeline.

Dictionary scanning and regex variant detection find most entities.
This prompt asks Sonnet to find any ADDITIONAL entities they missed.

Uses tool use for guaranteed valid JSON output.
"""

# Gold entity types only (6)
ENTITY_TYPES = ["GENE", "DISEASE", "CHEMICAL", "VARIANT", "ORGANISM", "CELL_TYPE"]

SYSTEM_PROMPT = """You are a biomedical named entity recognition system.
Given a biomedical abstract and a list of entities already found automatically, \
extract any ADDITIONAL biomedical entities that were missed.

## Entity types
GENE, DISEASE, CHEMICAL, VARIANT, ORGANISM, CELL_TYPE

## Type definitions

GENE: Genes AND gene products (proteins, receptors, enzymes, channels).
  - "TREM2" -> GENE. "Na(v)1.5" -> GENE. "beta-1 adrenoceptor" -> GENE.
  - Extract BOTH symbol and full name when both appear: "CBR3" AND "carbonyl reductase 3".

DISEASE: Diseases, disorders, syndromes, AND clinical signs/symptoms.
  - "Alzheimer disease" -> DISEASE. "bradycardia" -> DISEASE. "long QT syndrome" -> DISEASE.
  - "[Gene] deficiency" -> DISEASE (not GENE).

CHEMICAL: Drugs, chemicals, metabolites, ions, small molecules.
  - "donepezil" -> CHEMICAL. "sodium" -> CHEMICAL. "ATP" -> CHEMICAL.

VARIANT: Sequence variants in standard notation.
  - "V600E" -> VARIANT. "rs1799971" -> VARIANT. "c.444-62C>A" -> VARIANT.

ORGANISM: Species when biologically relevant.
  - "mouse" -> ORGANISM. Do NOT extract "patient", "human".

CELL_TYPE: Specific cell types and cell lines.
  - "microglia" -> CELL_TYPE. "HEK293" -> CELL_TYPE. "MCF7/AdrR" -> CELL_TYPE.

## What NOT to extract
- Descriptive phrases, experimental methods, generic terms (gene, protein, mutation, DNA)
- Amino acids/nucleosides in variant descriptions
- Qualified disease forms ("anthracycline-related CHF") — extract the core disease instead
- Disease subtype notations ("LQTS-3") — extract the parent ("LQTS")

## CRITICAL: Only extract entities NOT already in the provided list.
Do not repeat entities that are already found. Focus on what was missed."""


# Tool schema for structured output
ENTITY_TOOL = {
    "name": "extract_entities",
    "description": "Extract additional biomedical entities not already found.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Entity text as it appears in the abstract"},
                        "type": {"type": "string", "enum": ENTITY_TYPES},
                    },
                    "required": ["text", "type"],
                },
            },
        },
        "required": ["entities"],
    },
}


# --- Few-shot examples (from biored_train.jsonl, never test) ---

_FEWSHOT_1_TEXT = (
    "A family with two consecutive nonsense mutations in BMPR1A causing "
    "juvenile polyposis. We describe a novel germline mutation of BMPR1A in "
    "a family with juvenile polyposis and colon cancer. This mutation consists "
    "of two consecutive substitutions (735-6 TG>AT) that cause two nonsense "
    "mutations (Y245X, G246X), inherited in an autosomal dominant fashion, on "
    "one parental chromosome. This mutation caused protein truncation, and "
    "represents a novel case of consecutive nonsense mutations in human disease."
)

_FEWSHOT_1_KNOWN = [
    {"text": "BMPR1A", "type": "GENE"},
    {"text": "Y245X", "type": "VARIANT"},
    {"text": "G246X", "type": "VARIANT"},
]

_FEWSHOT_1_ADDITIONAL = [
    {"text": "juvenile polyposis", "type": "DISEASE"},
    {"text": "colon cancer", "type": "DISEASE"},
    {"text": "735-6 TG>AT", "type": "VARIANT"},
    {"text": "human", "type": "ORGANISM"},
]

_FEWSHOT_2_TEXT = (
    "Rab6c, a new member of the rab gene family, is involved in drug resistance "
    "in MCF7/AdrR cells. A new Rab6 homolog cDNA, Rab6c, was discovered by a "
    "hypermethylated DNA fragment probe that was isolated from a human multidrug "
    "resistant (MDR) breast cancer cell line, MCF7/AdrR, by the methylation "
    "sensitive-representational difference analysis (MS-RDA) technique. Rab6c was "
    "found to be under-expressed in MCF7/AdrR and MES-SA/Dx5 (a human MDR uterine "
    "sarcoma cell line) compared with their non-MDR parental cell lines. MCF7/AdrR "
    "cells expressing the exogenous Rab6c exhibited less resistance to several "
    "anti-cancer drugs, such as doxorubicin (DOX), taxol, vinblastine, and "
    "vincristine, than the control cells containing the empty vector."
)

_FEWSHOT_2_KNOWN = [
    {"text": "doxorubicin", "type": "CHEMICAL"},
    {"text": "taxol", "type": "CHEMICAL"},
    {"text": "vinblastine", "type": "CHEMICAL"},
    {"text": "vincristine", "type": "CHEMICAL"},
]

_FEWSHOT_2_ADDITIONAL = [
    {"text": "Rab6c", "type": "GENE"},
    {"text": "Rab6", "type": "GENE"},
    {"text": "breast cancer", "type": "DISEASE"},
    {"text": "uterine sarcoma", "type": "DISEASE"},
    {"text": "MCF7/AdrR", "type": "CELL_TYPE"},
    {"text": "MES-SA/Dx5", "type": "CELL_TYPE"},
    {"text": "human", "type": "ORGANISM"},
    {"text": "DOX", "type": "CHEMICAL"},
]


def _format_known_entities(entities: list[dict]) -> str:
    """Format known entities for the prompt."""
    if not entities:
        return "(none found)"
    lines = []
    for e in entities:
        lines.append(f"- {e['text']} ({e['type']})")
    return "\n".join(lines)


def _build_user_message(text: str, known_entities: list[dict]) -> str:
    """Build the user message with abstract and known entities."""
    known_str = _format_known_entities(known_entities)
    return (
        f"The following entities were found automatically in this abstract:\n"
        f"{known_str}\n\n"
        f"Extract any ADDITIONAL biomedical entities that were missed.\n\n"
        f"Abstract:\n{text}"
    )


def build_entity_gapfill_prompt(
    text: str,
    known_entities: list[dict],
) -> dict:
    """Build Claude API request body for entity gap-fill with tool use.

    Args:
        text: The biomedical abstract.
        known_entities: Entities already found by dictionary + regex.

    Returns:
        Dict with system, messages, tools, and tool_choice for the API call.
    """
    # Build few-shot messages
    messages = [
        # Few-shot example 1
        {"role": "user", "content": _build_user_message(_FEWSHOT_1_TEXT, _FEWSHOT_1_KNOWN)},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fs1", "name": "extract_entities",
             "input": {"entities": _FEWSHOT_1_ADDITIONAL}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "fs1", "content": "Entities recorded."},
        ]},
        # Few-shot example 2
        {"role": "user", "content": _build_user_message(_FEWSHOT_2_TEXT, _FEWSHOT_2_KNOWN)},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fs2", "name": "extract_entities",
             "input": {"entities": _FEWSHOT_2_ADDITIONAL}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "fs2", "content": "Entities recorded."},
        ]},
        # Actual request
        {"role": "user", "content": _build_user_message(text, known_entities)},
    ]

    return {
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": [ENTITY_TOOL],
        "tool_choice": {"type": "tool", "name": "extract_entities"},
    }


def parse_entity_gapfill_response(response_json: dict) -> list[dict]:
    """Parse tool use response from entity gap-fill.

    Args:
        response_json: The full API response JSON.

    Returns:
        List of entity dicts with text and type.
    """
    for block in response_json.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "extract_entities":
            entities = block.get("input", {}).get("entities", [])
            # Validate
            valid = []
            for e in entities:
                if "text" in e and "type" in e and e["type"] in ENTITY_TYPES:
                    valid.append({"text": e["text"], "type": e["type"]})
            return valid
    return []
