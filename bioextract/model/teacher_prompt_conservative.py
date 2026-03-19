from __future__ import annotations

"""Relationship extraction prompt for Sonnet — Stage 2a of the teacher pipeline.

Given text and a complete entity list, extract ALL relationships between them.
Uses tool use for guaranteed valid JSON output.
Called multiple times at temperature>0 for self-consistency voting.
"""

# Gold relationship types only (6)
RELATIONSHIP_TYPES = [
    "associated_with", "binds", "upregulates",
    "downregulates", "regulates", "interacts_with",
]

# Direction values
DIRECTIONS = ["positive", "negative", "neutral"]

SYSTEM_PROMPT = """You are a biomedical relationship extraction system.
Given a biomedical abstract and a list of entities found in it, extract ALL \
relationships between the entities that are stated or directly implied in the text.

## Relationship types
- associated_with: general association, genetic association, risk factor, correlation
- binds: physical binding, receptor-ligand interaction, substrate-enzyme
- upregulates: increases, activates, enhances, promotes, causes, induces, risk factor
- downregulates: decreases, inhibits, treats, controls, reduces, suppresses, attenuates
- regulates: modulates, controls (when direction is unclear)
- interacts_with: metabolic interaction, physical interaction, functional interaction

## Direction
- positive: activates, upregulates, causes, increases_risk, enhances
- negative: inhibits, downregulates, treats, decreases_risk, suppresses
- neutral: associated_with, binds, interacts_with, regulates (when direction unclear)

## Rules
- Subject and object MUST exactly match entity text from the provided entity list.
- Be THOROUGH: extract ALL relationships the text states or directly implies.
- For drug→disease: if a drug treats a syndrome, extract drug→syndrome AND drug→each symptom.
- For gene→disease: if a mutation causes a syndrome, extract variant→disease for each manifestation.
- Do NOT extract relationships based on background knowledge — only what the TEXT states.
- Do NOT extract structural facts (variant "located_in" gene) — only functional relationships.
- Do NOT extract class membership (X is a type of Y) — only biological relationships.
- Do NOT infer transitive relationships (A→B and B→C does NOT mean A→C).
- X and Y merely co-occurring in the same sentence with no stated connection → NO relationship."""


# Tool schema for structured output
RELATIONSHIP_TOOL = {
    "name": "extract_relationships",
    "description": "Extract relationships between the provided entities.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Subject entity text (must match entity list)"},
                        "object": {"type": "string", "description": "Object entity text (must match entity list)"},
                        "type": {"type": "string", "enum": RELATIONSHIP_TYPES},
                        "direction": {"type": "string", "enum": DIRECTIONS},
                    },
                    "required": ["subject", "object", "type", "direction"],
                },
            },
        },
        "required": ["relationships"],
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

_FEWSHOT_1_ENTITIES = [
    {"text": "BMPR1A", "type": "GENE"},
    {"text": "juvenile polyposis", "type": "DISEASE"},
    {"text": "colon cancer", "type": "DISEASE"},
    {"text": "735-6 TG>AT", "type": "VARIANT"},
    {"text": "Y245X", "type": "VARIANT"},
    {"text": "G246X", "type": "VARIANT"},
    {"text": "human", "type": "ORGANISM"},
]

_FEWSHOT_1_RELS = [
    {"subject": "BMPR1A", "object": "juvenile polyposis", "type": "associated_with", "direction": "neutral"},
    {"subject": "BMPR1A", "object": "colon cancer", "type": "associated_with", "direction": "neutral"},
    {"subject": "735-6 TG>AT", "object": "juvenile polyposis", "type": "upregulates", "direction": "positive"},
    {"subject": "735-6 TG>AT", "object": "colon cancer", "type": "associated_with", "direction": "neutral"},
    {"subject": "Y245X", "object": "juvenile polyposis", "type": "upregulates", "direction": "positive"},
    {"subject": "Y245X", "object": "colon cancer", "type": "associated_with", "direction": "neutral"},
    {"subject": "Y245X", "object": "G246X", "type": "associated_with", "direction": "neutral"},
    {"subject": "G246X", "object": "juvenile polyposis", "type": "upregulates", "direction": "positive"},
    {"subject": "G246X", "object": "colon cancer", "type": "associated_with", "direction": "neutral"},
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

_FEWSHOT_2_ENTITIES = [
    {"text": "Rab6c", "type": "GENE"},
    {"text": "Rab6", "type": "GENE"},
    {"text": "breast cancer", "type": "DISEASE"},
    {"text": "uterine sarcoma", "type": "DISEASE"},
    {"text": "MCF7/AdrR", "type": "CELL_TYPE"},
    {"text": "MES-SA/Dx5", "type": "CELL_TYPE"},
    {"text": "doxorubicin", "type": "CHEMICAL"},
    {"text": "DOX", "type": "CHEMICAL"},
    {"text": "taxol", "type": "CHEMICAL"},
    {"text": "vinblastine", "type": "CHEMICAL"},
    {"text": "vincristine", "type": "CHEMICAL"},
    {"text": "human", "type": "ORGANISM"},
]

_FEWSHOT_2_RELS = [
    {"subject": "uterine sarcoma", "object": "Rab6c", "type": "associated_with", "direction": "neutral"},
    {"subject": "Rab6c", "object": "DOX", "type": "upregulates", "direction": "positive"},
    {"subject": "Rab6c", "object": "taxol", "type": "upregulates", "direction": "positive"},
    {"subject": "Rab6c", "object": "vinblastine", "type": "upregulates", "direction": "positive"},
    {"subject": "Rab6c", "object": "vincristine", "type": "upregulates", "direction": "positive"},
]


def _format_entity_list(entities: list[dict]) -> str:
    """Format entities for the prompt."""
    lines = []
    for e in entities:
        lines.append(f"- [{e['type']}] {e['text']}")
    return "\n".join(lines)


def _build_user_message(text: str, entities: list[dict]) -> str:
    """Build the user message with abstract and entity list."""
    entity_str = _format_entity_list(entities)
    return (
        f"Entities found in the abstract:\n{entity_str}\n\n"
        f"Extract ALL relationships between these entities from the text.\n\n"
        f"Abstract:\n{text}"
    )


def build_relationship_prompt(
    text: str,
    entities: list[dict],
) -> dict:
    """Build Claude API request body for relationship extraction with tool use.

    Args:
        text: The biomedical abstract.
        entities: Complete entity list (text + type).

    Returns:
        Dict with system, messages, tools, and tool_choice for the API call.
    """
    messages = [
        # Few-shot example 1
        {"role": "user", "content": _build_user_message(_FEWSHOT_1_TEXT, _FEWSHOT_1_ENTITIES)},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fs1", "name": "extract_relationships",
             "input": {"relationships": _FEWSHOT_1_RELS}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "fs1", "content": "Relationships recorded."},
        ]},
        # Few-shot example 2
        {"role": "user", "content": _build_user_message(_FEWSHOT_2_TEXT, _FEWSHOT_2_ENTITIES)},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fs2", "name": "extract_relationships",
             "input": {"relationships": _FEWSHOT_2_RELS}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "fs2", "content": "Relationships recorded."},
        ]},
        # Actual request
        {"role": "user", "content": _build_user_message(text, entities)},
    ]

    return {
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": [RELATIONSHIP_TOOL],
        "tool_choice": {"type": "tool", "name": "extract_relationships"},
    }


def parse_relationship_response(response_json: dict, entity_texts: set[str] | None = None) -> list[dict]:
    """Parse tool use response from relationship extraction.

    Args:
        response_json: The full API response JSON.
        entity_texts: Optional set of valid entity texts (lowered) for validation.

    Returns:
        List of relationship dicts with subject, object, type, direction.
    """
    for block in response_json.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "extract_relationships":
            rels = block.get("input", {}).get("relationships", [])
            valid = []
            for r in rels:
                if all(k in r for k in ("subject", "object", "type", "direction")):
                    if r["type"] in RELATIONSHIP_TYPES and r["direction"] in DIRECTIONS:
                        # Optionally validate entity references
                        if entity_texts is not None:
                            if r["subject"].lower() not in entity_texts:
                                continue
                            if r["object"].lower() not in entity_texts:
                                continue
                        valid.append({
                            "subject": r["subject"],
                            "object": r["object"],
                            "type": r["type"],
                            "direction": r["direction"],
                        })
            return valid
    return []
