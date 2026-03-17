"""Sonnet verifier model — final pass to filter FPs and resolve conflicts.

Takes merged candidate entities and relationships from all ensemble models,
sends to Sonnet for yes/no verification. Adjusts confidence scores based on
verification results.
"""

import json
import logging
import os
import time

from ..schema import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    RelationshipContext,
)

logger = logging.getLogger(__name__)

VERIFY_SYSTEM = """You are a biomedical extraction verifier. Given a biomedical abstract and candidate extractions, verify each one.

For each ENTITY, determine:
- KEEP: This is a valid biomedical entity correctly extracted from the text.
- REMOVE: This is not a real entity, is too generic, is a method name, or was incorrectly typed.

For each RELATIONSHIP, determine:
- KEEP: The text explicitly states or clearly implies this relationship between these entities.
- REMOVE: This relationship is not supported by the text, or the entities don't actually interact in the described way.
- CHANGE_TYPE: The relationship exists but the type is wrong (provide corrected type).

Be strict: if the text does NOT clearly support a relationship, mark it REMOVE. Co-occurrence alone is not a relationship.

Return ONLY valid JSON with your decisions."""

VERIFY_USER_TEMPLATE = """Abstract:
{abstract}

Candidate entities to verify:
{entities_block}

Candidate relationships to verify:
{relationships_block}

Return JSON:
{{
  "entities": [
    {{"id": 1, "verdict": "KEEP"}},
    {{"id": 2, "verdict": "REMOVE", "reason": "method name, not entity"}}
  ],
  "relationships": [
    {{"id": 1, "verdict": "KEEP"}},
    {{"id": 2, "verdict": "REMOVE", "reason": "co-occurrence only"}},
    {{"id": 3, "verdict": "CHANGE_TYPE", "new_type": "downregulates", "reason": "treats relationship"}}
  ]
}}"""


def _build_verify_prompt(
    abstract: str,
    entities: list[ExtractedEntity],
    relationships: list[ExtractedRelationship],
) -> list[dict]:
    """Build verification prompt."""
    ent_lines = []
    for i, e in enumerate(entities, 1):
        ent_lines.append(
            f"  {i}. \"{e.text}\" [{e.type}] (confidence: {e.confidence:.2f})"
        )

    rel_lines = []
    for i, r in enumerate(relationships, 1):
        rel_lines.append(
            f"  {i}. \"{r.subject}\" --[{r.type}]--> \"{r.object}\" "
            f"(confidence: {r.confidence:.2f})"
        )

    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_USER_TEMPLATE.format(
            abstract=abstract[:2000],
            entities_block="\n".join(ent_lines) if ent_lines else "  (none)",
            relationships_block="\n".join(rel_lines) if rel_lines else "  (none)",
        )},
    ]


def _call_sonnet(messages: list[dict], api_key: str) -> str | None:
    """Call Sonnet API."""
    try:
        import httpx
    except ImportError:
        return None

    model = os.environ.get("BIOEXTRACT_VERIFIER_MODEL", "claude-sonnet-4-6")

    try:
        with httpx.Client(timeout=60) as client:
            for attempt in range(3):
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 4096,
                        "system": messages[0]["content"],
                        "messages": [messages[1]],
                    },
                )
                if resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                break

            if resp.status_code != 200:
                logger.warning("Verifier API error %d", resp.status_code)
                return None
            return resp.json()["content"][0]["text"]
    except Exception as e:
        logger.warning("Verifier API call failed: %s", e)
        return None


def verify_extraction(
    text: str,
    entities: list[ExtractedEntity],
    relationships: list[ExtractedRelationship],
) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
    """Verify and filter candidate entities and relationships.

    Args:
        text: The abstract text.
        entities: Merged candidate entities from ensemble.
        relationships: Merged candidate relationships from ensemble.

    Returns:
        Tuple of (verified_entities, verified_relationships) with updated
        confidence scores.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY for verifier")
        return entities, relationships

    if not entities and not relationships:
        return entities, relationships

    messages = _build_verify_prompt(text, entities, relationships)
    response = _call_sonnet(messages, api_key)

    if not response:
        return entities, relationships

    # Parse response
    resp_text = response.strip()
    if resp_text.startswith("```"):
        lines = resp_text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        resp_text = "\n".join(lines)

    try:
        verdicts = json.loads(resp_text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse verifier response")
        return entities, relationships

    # Apply entity verdicts
    ent_verdicts = {
        v.get("id", 0): v
        for v in verdicts.get("entities", [])
    }
    verified_entities = []
    for i, e in enumerate(entities, 1):
        verdict = ent_verdicts.get(i, {})
        decision = verdict.get("verdict", "KEEP")
        if decision == "KEEP":
            # Boost confidence for verified entities
            e.confidence = min(1.0, e.confidence * 1.1)
            verified_entities.append(e)
        elif decision == "REMOVE":
            # Skip removed entities
            logger.debug("Verifier removed entity: %s (%s)", e.text, verdict.get("reason", ""))
        else:
            # Unknown verdict, keep by default
            verified_entities.append(e)

    # Apply relationship verdicts
    rel_verdicts = {
        v.get("id", 0): v
        for v in verdicts.get("relationships", [])
    }
    verified_rels = []
    for i, r in enumerate(relationships, 1):
        verdict = rel_verdicts.get(i, {})
        decision = verdict.get("verdict", "KEEP")
        if decision == "KEEP":
            r.confidence = min(1.0, r.confidence * 1.15)
            verified_rels.append(r)
        elif decision == "CHANGE_TYPE":
            new_type = verdict.get("new_type", r.type)
            verified_rels.append(ExtractedRelationship(
                subject=r.subject,
                predicate=r.predicate,
                object=r.object,
                type=new_type,
                direction=r.direction,
                negated=r.negated,
                context=r.context,
                confidence=min(1.0, r.confidence * 1.1),
            ))
        elif decision == "REMOVE":
            logger.debug(
                "Verifier removed rel: %s -> %s (%s)",
                r.subject, r.object, verdict.get("reason", ""),
            )
        else:
            verified_rels.append(r)

    return verified_entities, verified_rels
