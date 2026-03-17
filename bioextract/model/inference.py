"""Student model inference for biomedical extraction.

Uses a fine-tuned SciFive-base (or Flan-T5-base fallback) model
for fast CPU inference. Falls back to Claude teacher if no student
model is available.

Supports two model types:
  - Generative (seq2seq): single model for entities + relationships
  - Split pipeline: PubMedBERT NER + Flan-T5 RE
"""

import json
import logging
from pathlib import Path

from ..schema import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    RelationshipContext,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "trained"
ONNX_PATH = MODEL_DIR / "model.onnx"

# Split pipeline model directories (set by train_progressive)
NER_DIR = MODEL_DIR / "ner"
RE_DIR = MODEL_DIR / "re"

# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None


def _load_model():
    """Load the trained student model. Returns (model, tokenizer) or (None, None)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    model_path = MODEL_DIR / "pytorch_model"
    if not model_path.exists() and not ONNX_PATH.exists():
        logger.warning("No trained student model found at %s", MODEL_DIR)
        return None, None

    try:
        # Prefer ONNX for faster CPU inference
        if ONNX_PATH.exists():
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            _model = ORTModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
            logger.info("Loaded ONNX student model")
        else:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            _model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            logger.info("Loaded PyTorch student model")

        return _model, _tokenizer
    except Exception as e:
        logger.error("Failed to load student model: %s", e)
        return None, None


def _is_split_pipeline_available(ner_dir: Path | None = None, re_dir: Path | None = None) -> bool:
    """Check if a trained split pipeline (NER + RE) exists."""
    ner = ner_dir or NER_DIR
    re = re_dir or RE_DIR
    return (ner / "config.json").exists() and (re / "adapter_config.json").exists()


def is_model_available() -> bool:
    """Check if any trained student model exists (generative or split pipeline)."""
    return (
        (MODEL_DIR / "pytorch_model").exists()
        or ONNX_PATH.exists()
        or _is_split_pipeline_available()
    )


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

    Auto-detects which model type is available:
      1. Split pipeline (NER + RE) — preferred if available
      2. Generative seq2seq model
    Returns ExtractionResult or None if no model is available.
    """
    # Prefer split pipeline if available
    if _is_split_pipeline_available():
        logger.info("Using split pipeline (NER + RE)")
        return extract_with_split_pipeline(text)

    model, tokenizer = _load_model()
    if model is None:
        return None

    # Encode input
    prompt = f"extract entities and relationships: {text}"
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=2,
        early_stopping=True,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse JSON output
    try:
        data = json.loads(decoded)
    except json.JSONDecodeError:
        logger.warning("Student model produced invalid JSON: %s", decoded[:200])
        return ExtractionResult(text=text, extraction_method="bioextract_v1_failed")

    return _parse_model_output(text, data)


def _parse_model_output(text: str, data: dict) -> ExtractionResult:
    """Convert model JSON output to ExtractionResult."""
    entities = []
    for e in data.get("entities", []):
        entities.append(ExtractedEntity(
            text=e.get("text", ""),
            type=e.get("type", "GENE"),
            start=e.get("start", 0),
            end=e.get("end", 0),
            confidence=0.7,  # student model confidence baseline
        ))

    relationships = []
    for r in data.get("relationships", []):
        ctx = r.get("context", {})
        relationships.append(ExtractedRelationship(
            subject=r.get("subject", ""),
            predicate=r.get("predicate", ""),
            object=r.get("object", ""),
            type=r.get("type", "associated_with"),
            direction=r.get("direction", "neutral"),
            negated=r.get("negated", False),
            context=RelationshipContext(
                organism=ctx.get("organism"),
                cell_type=ctx.get("cell_type"),
                experiment_type=ctx.get("experiment_type"),
            ),
            confidence=0.7,
        ))

    return ExtractionResult(
        text=text,
        entities=entities,
        relationships=relationships,
        extraction_method="bioextract_v1",
    )
