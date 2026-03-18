"""PubMedBERT NER training for Approach B (split pipeline).

Stage 1: Token classification with BIO tagging.
Full fine-tuning (110M params is small enough, no LoRA needed).

Designed to run on Google Colab (T4 GPU). Called by train_progressive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PUBMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Gold entity types and their BIO tags
ENTITY_TYPES = ["GENE", "DISEASE", "CHEMICAL", "VARIANT", "ORGANISM", "CELL_TYPE"]
LABEL_LIST = ["O"]
for etype in ENTITY_TYPES:
    LABEL_LIST.append(f"B-{etype}")
    LABEL_LIST.append(f"I-{etype}")

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def convert_to_bio(text: str, entities: list[dict], tokenizer, max_length: int = 512):
    """Tokenize text and align entity character spans to BIO subword tags.

    Returns:
        dict with input_ids, attention_mask, labels, offset_mapping
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    offsets = encoding["offset_mapping"]
    labels = []

    for i, (token_start, token_end) in enumerate(offsets):
        # Special tokens (CLS, SEP, PAD) get -100
        if token_start == 0 and token_end == 0:
            labels.append(-100)
            continue

        label = "O"
        for ent in entities:
            ent_start = ent["start"]
            ent_end = ent["end"]
            ent_type = ent["type"]

            # Skip entity types not in our gold set
            if ent_type not in ENTITY_TYPES:
                continue

            # Token overlaps entity if token_start < entity_end and token_end > entity_start
            if token_start < ent_end and token_end > ent_start:
                # B-tag if this is the first overlapping token
                if token_start <= ent_start:
                    label = f"B-{ent_type}"
                else:
                    label = f"I-{ent_type}"
                break  # First matching entity wins

        labels.append(LABEL2ID[label])

    encoding["labels"] = labels
    return encoding


def _build_dataset(examples: list[dict], tokenizer, max_length: int = 512):
    """Convert examples to a HuggingFace Dataset for token classification."""
    from datasets import Dataset

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for ex in examples:
        enc = convert_to_bio(ex["text"], ex.get("entities", []), tokenizer, max_length)
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])
        all_labels.append(enc["labels"])

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })


def train_ner(
    examples: list[dict],
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 512,
):
    """Full fine-tune PubMedBERT for NER with BIO tagging.

    Args:
        examples: Training examples with text, entities (with start/end/type).
        output_dir: Where to save the model.
        epochs: Training epochs.
        batch_size: Per-device batch size (capped at 4 on CPU).
        learning_rate: Learning rate.
        max_length: Max token length (512 for PubMedBERT).
    """
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    device_is_gpu = torch.cuda.is_available()
    if not device_is_gpu:
        batch_size = min(batch_size, 4)

    logger.info("Training NER: %d examples, %d epochs, batch_size=%d", len(examples), epochs, batch_size)

    tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT)
    model = AutoModelForTokenClassification.from_pretrained(
        PUBMEDBERT,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Build datasets (90/10 split for eval, unless very small)
    if len(examples) > 2:
        split_idx = max(1, int(len(examples) * 0.9))
        train_ds = _build_dataset(examples[:split_idx], tokenizer, max_length)
        eval_ds = _build_dataset(examples[split_idx:], tokenizer, max_length)
    else:
        train_ds = _build_dataset(examples, tokenizer, max_length)
        eval_ds = None

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        logging_steps=max(1, len(train_ds) // batch_size // 5),
        fp16=device_is_gpu,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("NER model saved to %s", output_dir)


def load_ner_model(model_dir: str):
    """Load a trained NER model and tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_entities(text: str, model_dir: str = None, max_length: int = 512,
                     model=None, tokenizer=None) -> list[dict]:
    """Run NER inference: tokenize → predict BIO tags → merge spans → return entity dicts.

    Returns list of {"text": str, "type": str, "start": int, "end": int}.
    Entity text is extracted from the original text using character offsets (not tokenized text).

    Pass model and tokenizer to avoid reloading on every call.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_ner_model(model_dir)

    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offset_mapping = encoding.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**encoding)
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()

    # Merge B/I spans into entities
    entities = []
    current_entity = None

    for idx, (pred_id, (tok_start, tok_end)) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens
        if tok_start == 0 and tok_end == 0:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        label = ID2LABEL.get(pred_id, "O")

        if label.startswith("B-"):
            # Save previous entity if any
            if current_entity is not None:
                entities.append(current_entity)
            etype = label[2:]
            current_entity = {
                "type": etype,
                "start": tok_start,
                "end": tok_end,
            }
        elif label.startswith("I-") and current_entity is not None:
            etype = label[2:]
            if etype == current_entity["type"]:
                # Extend current entity
                current_entity["end"] = tok_end
            else:
                # Type mismatch — close current, start new
                entities.append(current_entity)
                current_entity = {
                    "type": etype,
                    "start": tok_start,
                    "end": tok_end,
                }
        else:
            # O tag or orphan I- tag
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    # Extract entity text from original text using character offsets
    for ent in entities:
        ent["text"] = text[ent["start"]:ent["end"]]

    return entities
