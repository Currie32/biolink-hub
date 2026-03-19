"""Student model inference for biomedical extraction.

Uses the split pipeline (BioLinkBERT-large NER + BioLinkBERT-large RE pair classifier)
for fast CPU/GPU inference. Falls back to Claude teacher if no student
model is available.
"""

import logging
from pathlib import Path

from ..schema import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "trained"

# Split pipeline model directories (set by train_progressive)
NER_DIR = MODEL_DIR / "ner"
RE_DIR = MODEL_DIR / "re"


def _is_split_pipeline_available(ner_dir: Path | None = None, re_dir: Path | None = None) -> bool:
    """Check if a trained split pipeline (NER + RE) exists."""
    ner = ner_dir or NER_DIR
    re = re_dir or RE_DIR
    return (ner / "config.json").exists() and (re / "re_heads.pt").exists()


def is_model_available() -> bool:
    """Check if a trained student model exists."""
    return _is_split_pipeline_available()


def extract_with_split_pipeline(
    text: str,
    ner_dir: str | Path | None = None,
    re_dir: str | Path | None = None,
) -> ExtractionResult:
    """Run extraction using the split NER + RE pipeline.

    Args:
        text: Input text to extract from.
        ner_dir: Path to trained NER model (default: MODEL_DIR/ner).
        re_dir: Path to trained RE model (default: MODEL_DIR/re).

    Returns:
        ExtractionResult with entities and relationships.
    """
    from .train_ner import predict_entities
    from .train_re import predict_relationships

    ner_path = str(ner_dir or NER_DIR)
    re_path = str(re_dir or RE_DIR)

    # Stage 1: NER
    entities = predict_entities(text, ner_path)

    # Stage 2: RE (conditioned on predicted entities)
    relationships = predict_relationships(text, entities, re_path)

    # Convert to ExtractionResult
    result_entities = [
        ExtractedEntity(
            text=e["text"],
            type=e["type"],
            start=e.get("start", 0),
            end=e.get("end", 0),
            confidence=0.7,
        )
        for e in entities
    ]

    result_relationships = [
        ExtractedRelationship(
            subject=r.get("subject", ""),
            predicate="",
            object=r.get("object", ""),
            type=r.get("type", "associated_with"),
            direction=r.get("direction", "neutral"),
            confidence=0.7,
        )
        for r in relationships
    ]

    return ExtractionResult(
        text=text,
        entities=result_entities,
        relationships=result_relationships,
        extraction_method="bioextract_v1_split",
    )


def extract_with_student(text: str) -> ExtractionResult | None:
    """Run extraction using the fine-tuned student model.

    Returns ExtractionResult or None if no model is available.
    """
    if _is_split_pipeline_available():
        logger.info("Using split pipeline (NER + RE)")
        return extract_with_split_pipeline(text)

    return None
