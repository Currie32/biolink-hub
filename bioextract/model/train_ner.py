"""BioLinkBERT-large NER training with BIOES tagging and sliding window.

Stage 1: Token classification with BIOES tagging.
Full fine-tuning (340M params). Sliding window handles abstracts > 512 tokens.

BIOES improves boundary detection over BIO:
  B = Beginning of multi-token entity
  I = Inside multi-token entity
  O = Outside any entity
  E = End of multi-token entity
  S = Single-token entity

Designed to run on Google Colab (T4 GPU). Called by train_progressive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

BIOLINKBERT = "michiyasunaga/BioLinkBERT-large"

WINDOW_SIZE = 512
WINDOW_STRIDE = 256

# Gold entity types and their BIOES tags
ENTITY_TYPES = ["GENE", "DISEASE", "CHEMICAL", "VARIANT", "ORGANISM", "CELL_TYPE"]
LABEL_LIST = ["O"]
for _etype in ENTITY_TYPES:
    LABEL_LIST.extend([f"B-{_etype}", f"I-{_etype}", f"E-{_etype}", f"S-{_etype}"])

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


# ---------------------------------------------------------------------------
# BIOES label assignment
# ---------------------------------------------------------------------------

def _assign_bioes_labels(
    offsets: list[tuple[int, int]],
    entities: list[dict],
) -> list[int]:
    """Assign BIOES labels to tokens based on entity character spans.

    Args:
        offsets: Token offset mapping [(char_start, char_end), ...].
        entities: Entity dicts with start, end, type.

    Returns:
        List of label IDs (one per token). Special tokens get -100.
    """
    n = len(offsets)

    # Map each token to its entity index (first match wins)
    token_entity = [None] * n
    for ent_idx, ent in enumerate(entities):
        ent_type = ent.get("type", "")
        if ent_type not in ENTITY_TYPES:
            continue
        ent_start, ent_end = ent["start"], ent["end"]
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue  # special token
            if token_entity[i] is not None:
                continue  # already assigned
            if tok_start < ent_end and tok_end > ent_start:
                token_entity[i] = ent_idx

    # Assign BIOES tags based on position within entity span
    labels = []
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == 0 and tok_end == 0:
            labels.append(-100)
            continue

        ent_idx = token_entity[i]
        if ent_idx is None:
            labels.append(LABEL2ID["O"])
            continue

        etype = entities[ent_idx]["type"]
        is_first = (i == 0) or (token_entity[i - 1] != ent_idx)
        is_last = (i == n - 1) or (token_entity[i + 1] != ent_idx)

        if is_first and is_last:
            labels.append(LABEL2ID[f"S-{etype}"])
        elif is_first:
            labels.append(LABEL2ID[f"B-{etype}"])
        elif is_last:
            labels.append(LABEL2ID[f"E-{etype}"])
        else:
            labels.append(LABEL2ID[f"I-{etype}"])

    return labels


# ---------------------------------------------------------------------------
# Sliding window: training
# ---------------------------------------------------------------------------

def _create_training_windows(
    text: str,
    entities: list[dict],
    tokenizer,
    max_length: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> list[dict]:
    """Tokenize text into sliding windows with BIOES labels.

    Short texts produce a single window. Long texts produce overlapping windows
    so that entities near the end are not lost.

    Returns list of dicts with input_ids, attention_mask, labels.
    """
    # Tokenize without special tokens to get raw subwords
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    all_ids = enc["input_ids"]
    all_offsets = enc["offset_mapping"]
    inner_max = max_length - 2  # reserve space for CLS + SEP

    windows = []

    if len(all_ids) <= inner_max:
        # Single window
        input_ids = [tokenizer.cls_token_id] + all_ids + [tokenizer.sep_token_id]
        offsets = [(0, 0)] + all_offsets + [(0, 0)]
        attention_mask = [1] * len(input_ids)
        labels = _assign_bioes_labels(offsets, entities)
        windows.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    else:
        # Sliding windows
        for start in range(0, len(all_ids), stride):
            end = min(start + inner_max, len(all_ids))
            input_ids = (
                [tokenizer.cls_token_id]
                + all_ids[start:end]
                + [tokenizer.sep_token_id]
            )
            offsets = [(0, 0)] + all_offsets[start:end] + [(0, 0)]
            attention_mask = [1] * len(input_ids)
            labels = _assign_bioes_labels(offsets, entities)
            windows.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
            if end >= len(all_ids):
                break

    return windows


def _build_dataset(
    examples: list[dict],
    tokenizer,
    max_length: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
):
    """Convert examples to a HuggingFace Dataset with sliding windows."""
    from datasets import Dataset

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for ex in examples:
        windows = _create_training_windows(
            ex["text"], ex.get("entities", []),
            tokenizer, max_length, stride,
        )
        for w in windows:
            all_input_ids.append(w["input_ids"])
            all_attention_mask.append(w["attention_mask"])
            all_labels.append(w["labels"])

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ner(
    examples: list[dict],
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = WINDOW_SIZE,
):
    """Full fine-tune BioLinkBERT-large for NER with BIOES tagging.

    Uses sliding window to handle abstracts longer than 512 tokens.

    Args:
        examples: Training examples with text, entities (with start/end/type).
        output_dir: Where to save the model.
        epochs: Training epochs.
        batch_size: Per-device batch size (capped at 2 on CPU).
        learning_rate: Learning rate.
        max_length: Max token window size (512 for BioLinkBERT).
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
        batch_size = min(batch_size, 2)

    print(f"  [NER] Config: {len(examples)} examples, {epochs} epochs, "
          f"batch_size={batch_size}, gpu={device_is_gpu}", flush=True)
    print(f"  [NER] Loading BioLinkBERT-large tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BIOLINKBERT)
    model = AutoModelForTokenClassification.from_pretrained(
        BIOLINKBERT,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [NER] Model params: {n_params:,}", flush=True)

    # Build datasets with sliding windows (90/10 split for eval)
    if len(examples) > 2:
        split_idx = max(1, int(len(examples) * 0.9))
        train_ds = _build_dataset(examples[:split_idx], tokenizer, max_length)
        eval_ds = _build_dataset(examples[split_idx:], tokenizer, max_length)
    else:
        train_ds = _build_dataset(examples, tokenizer, max_length)
        eval_ds = None

    print(f"  [NER] Training windows: {len(train_ds)}"
          + (f", eval windows: {len(eval_ds)}" if eval_ds else ""), flush=True)

    # Gradient accumulation to simulate larger effective batch on GPU
    grad_accum = 4 if device_is_gpu else 1

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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

    print(f"  [NER] Starting training...", flush=True)
    trainer.train()
    print(f"  [NER] Training complete. Saving model...", flush=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  [NER] Model saved to {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_ner_model(model_dir: str):
    """Load a trained NER model and tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device).eval()
    return model, tokenizer


def _decode_bioes(
    predictions: list[int],
    offsets: list[tuple[int, int]],
    text: str,
) -> list[dict]:
    """Decode BIOES tag predictions into entity dicts.

    Handles malformed sequences gracefully (orphan I/E treated as B/S).

    Returns list of {"text": str, "type": str, "start": int, "end": int}.
    """
    entities = []
    current = None

    for pred_id, (tok_start, tok_end) in zip(predictions, offsets):
        # Skip special tokens
        if tok_start == 0 and tok_end == 0:
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                entities.append(current)
                current = None
            continue

        label = ID2LABEL.get(pred_id, "O")

        if label.startswith("S-"):
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                entities.append(current)
                current = None
            etype = label[2:]
            entities.append({
                "type": etype,
                "start": tok_start,
                "end": tok_end,
                "text": text[tok_start:tok_end],
            })

        elif label.startswith("B-"):
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                entities.append(current)
            current = {"type": label[2:], "start": tok_start, "end": tok_end}

        elif label.startswith("I-"):
            if current is not None and label[2:] == current["type"]:
                current["end"] = tok_end
            else:
                if current is not None:
                    current["text"] = text[current["start"]:current["end"]]
                    entities.append(current)
                # Orphan I — treat as B
                current = {"type": label[2:], "start": tok_start, "end": tok_end}

        elif label.startswith("E-"):
            if current is not None and label[2:] == current["type"]:
                current["end"] = tok_end
                current["text"] = text[current["start"]:current["end"]]
                entities.append(current)
                current = None
            else:
                if current is not None:
                    current["text"] = text[current["start"]:current["end"]]
                    entities.append(current)
                # Orphan E — treat as S
                entities.append({
                    "type": label[2:],
                    "start": tok_start,
                    "end": tok_end,
                    "text": text[tok_start:tok_end],
                })
                current = None

        else:  # O
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                entities.append(current)
                current = None

    if current is not None:
        current["text"] = text[current["start"]:current["end"]]
        entities.append(current)

    return entities


def _dedup_entities(entities: list[dict]) -> list[dict]:
    """Deduplicate entities from overlapping sliding windows.

    Prefers longer spans when entities of the same type overlap.
    """
    if not entities:
        return entities

    # Sort by start position, then longest first
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

    deduped = []
    for ent in entities:
        overlaps = False
        for kept in deduped:
            if (ent["start"] < kept["end"] and ent["end"] > kept["start"]
                    and ent["type"] == kept["type"]):
                overlaps = True
                break
        if not overlaps:
            deduped.append(ent)

    return deduped


def predict_entities(
    text: str,
    model_dir: str = None,
    max_length: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """Run NER inference with BIOES decoding and sliding window.

    For texts that fit in a single window, runs a single forward pass.
    For longer texts, uses overlapping windows and merges predictions.

    Returns list of {"text": str, "type": str, "start": int, "end": int}.
    Entity text is extracted from the original text using character offsets.

    Pass model and tokenizer to avoid reloading on every call.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_ner_model(model_dir)

    device = next(model.parameters()).device

    # Tokenize without special tokens to get raw subwords
    full_enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = full_enc["input_ids"]
    full_offsets = full_enc["offset_mapping"]
    inner_max = max_length - 2  # CLS + SEP

    # Build windows
    windows = []
    if len(full_ids) <= inner_max:
        windows.append((0, len(full_ids)))
    else:
        for start in range(0, len(full_ids), stride):
            end = min(start + inner_max, len(full_ids))
            windows.append((start, end))
            if end >= len(full_ids):
                break

    # Run each window
    all_entities = []
    for win_start, win_end in windows:
        win_ids = (
            [tokenizer.cls_token_id]
            + full_ids[win_start:win_end]
            + [tokenizer.sep_token_id]
        )
        win_offsets = [(0, 0)] + full_offsets[win_start:win_end] + [(0, 0)]
        win_mask = [1] * len(win_ids)

        encoding = {
            "input_ids": torch.tensor([win_ids], device=device),
            "attention_mask": torch.tensor([win_mask], device=device),
        }

        with torch.no_grad():
            outputs = model(**encoding)
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()

        entities = _decode_bioes(predictions, win_offsets, text)
        all_entities.extend(entities)

    # Deduplicate entities from overlapping windows
    if len(windows) > 1:
        all_entities = _dedup_entities(all_entities)

    return all_entities
