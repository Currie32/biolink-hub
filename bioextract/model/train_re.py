"""BioLinkBERT-large pair classifier for relationship extraction (Stage 2).

SOTA approach following BioREx/BioREDirect:
- Typed entity markers: [S:TYPE] subject [/S:TYPE] ... [O:TYPE] object [/O:TYPE]
- [CLS] token classification with two heads:
  - Relation type (7 classes: no_relation + 6 gold types)
  - Direction (3 classes: positive, negative, neutral) — multi-task, trained on positive pairs only

Enumerate all entity pairs per abstract, classify each.
Much faster than seq2seq: single forward pass per pair, no generation.

Designed to run on Google Colab (T4 GPU). Called by train_progressive.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BIOLINKBERT = "dmis-lab/biobert-base-cased-v1.2"

# Relation types (index 0 = no_relation for negative pairs)
REL_TYPES = [
    "no_relation", "associated_with", "binds", "upregulates",
    "downregulates", "regulates", "interacts_with",
]
REL2ID = {r: i for i, r in enumerate(REL_TYPES)}
ID2REL = {i: r for i, r in enumerate(REL_TYPES)}

# Direction types
DIR_TYPES = ["positive", "negative", "neutral"]
DIR2ID = {d: i for i, d in enumerate(DIR_TYPES)}
ID2DIR = {i: d for i, d in enumerate(DIR_TYPES)}

# Entity types for typed markers
ENTITY_TYPES = ["GENE", "DISEASE", "CHEMICAL", "VARIANT", "ORGANISM", "CELL_TYPE"]


def _special_tokens() -> list[str]:
    """Typed entity marker special tokens to add to tokenizer."""
    tokens = []
    for t in ENTITY_TYPES:
        tokens.extend([f"[S:{t}]", f"[/S:{t}]", f"[O:{t}]", f"[/O:{t}]"])
    return tokens


def insert_entity_markers(text: str, subj: dict, obj: dict) -> str:
    """Insert typed entity markers around subject and object spans in text.

    Subject: [S:TYPE] text [/S:TYPE]
    Object: [O:TYPE] text [/O:TYPE]

    Falls back to prepending markers if spans overlap.
    """
    s_start, s_end = subj["start"], subj["end"]
    o_start, o_end = obj["start"], obj["end"]
    s_type, o_type = subj["type"], obj["type"]

    # Handle overlapping spans by prepending markers
    if s_start < o_end and o_start < s_end:
        s_text = subj.get("text", text[s_start:s_end])
        o_text = obj.get("text", text[o_start:o_end])
        return (f"[S:{s_type}] {s_text} [/S:{s_type}] "
                f"[O:{o_type}] {o_text} [/O:{o_type}] {text}")

    # Non-overlapping: insert inline
    if s_start <= o_start:
        return (text[:s_start]
                + f"[S:{s_type}] " + text[s_start:s_end] + f" [/S:{s_type}]"
                + text[s_end:o_start]
                + f"[O:{o_type}] " + text[o_start:o_end] + f" [/O:{o_type}]"
                + text[o_end:])
    else:
        return (text[:o_start]
                + f"[O:{o_type}] " + text[o_start:o_end] + f" [/O:{o_type}]"
                + text[o_end:s_start]
                + f"[S:{s_type}] " + text[s_start:s_end] + f" [/S:{s_type}]"
                + text[s_end:])


def _build_pairs(
    examples: list[dict], neg_ratio: int = 3, seed: int = 42,
) -> list[dict]:
    """Build training pairs from examples.

    For each example, enumerate all ordered entity pairs.
    Gold relationship pairs get their relation type + direction.
    Remaining pairs are no_relation (negative-sampled at neg_ratio).

    Returns list of {text: str, rel_label: int, dir_label: int}.
    """
    rng = random.Random(seed)
    all_pairs = []

    for ex in examples:
        text = ex["text"]
        entities = ex.get("entities", [])
        relationships = ex.get("relationships", [])

        # Deduplicate entities by (text_lower, type), keep first span
        seen = {}
        unique_ents = []
        for ent in entities:
            key = (ent["text"].lower(), ent["type"])
            if key not in seen:
                seen[key] = len(unique_ents)
                unique_ents.append(ent)

        if len(unique_ents) < 2:
            continue

        # Gold relationship lookup: (subj_lower, obj_lower) → (rel_type, direction)
        gold_rels = {}
        for rel in relationships:
            if rel["type"] in REL2ID:
                gold_rels[(rel["subject"].lower(), rel["object"].lower())] = (
                    rel["type"], rel.get("direction", "neutral"),
                )

        positives = []
        negatives = []

        for i, subj in enumerate(unique_ents):
            for j, obj in enumerate(unique_ents):
                if i == j:
                    continue
                key = (subj["text"].lower(), obj["text"].lower())
                marked = insert_entity_markers(text, subj, obj)

                if key in gold_rels:
                    rel_type, direction = gold_rels[key]
                    positives.append({
                        "text": marked,
                        "rel_label": REL2ID[rel_type],
                        "dir_label": DIR2ID.get(direction, DIR2ID["neutral"]),
                    })
                else:
                    negatives.append({
                        "text": marked,
                        "rel_label": 0,
                        "dir_label": 0,
                    })

        # Negative sampling: cap negatives at neg_ratio * positives
        rng.shuffle(negatives)
        n_neg = min(len(negatives), max(len(positives) * neg_ratio, 2))
        all_pairs.extend(positives)
        all_pairs.extend(negatives[:n_neg])

    rng.shuffle(all_pairs)
    return all_pairs


class REPairClassifier(nn.Module):
    """BioLinkBERT-large pair classifier with relation type + direction heads.

    Takes [CLS] representation from encoder with typed entity markers,
    classifies into relation type (7) and direction (3).
    Direction loss only computed for positive (non-no_relation) pairs.
    """

    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        hidden = bert_model.config.hidden_size
        self.dropout = nn.Dropout(bert_model.config.hidden_dropout_prob)
        self.rel_head = nn.Linear(hidden, len(REL_TYPES))
        self.dir_head = nn.Linear(hidden, len(DIR_TYPES))

    def forward(
        self, input_ids, attention_mask=None,
        rel_labels=None, dir_labels=None, **kwargs,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0])

        rel_logits = self.rel_head(cls)
        dir_logits = self.dir_head(cls)

        loss = None
        if rel_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss = ce(rel_logits, rel_labels)
            # Direction loss only for positive relations (not no_relation)
            if dir_labels is not None:
                pos_mask = rel_labels != 0
                if pos_mask.any():
                    loss = loss + ce(dir_logits[pos_mask], dir_labels[pos_mask])

        return {"loss": loss, "rel_logits": rel_logits, "dir_logits": dir_logits}


def train_re(
    examples: list[dict],
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    neg_ratio: int = 3,
):
    """Train BioLinkBERT-large pair classifier for relationship extraction.

    Full fine-tuning of BioLinkBERT-large (340M params) with two classification heads.

    Args:
        examples: Training examples with text, entities (with start/end/type), relationships.
        output_dir: Where to save the model.
        epochs: Training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate (2e-5 standard for BERT fine-tuning).
        max_length: Max token length.
        neg_ratio: Max negative-to-positive pair ratio per example.
    """
    from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
    from datasets import Dataset

    device_is_gpu = torch.cuda.is_available()
    if not device_is_gpu:
        batch_size = min(batch_size, 2)

    # Build training pairs
    pairs = _build_pairs(examples, neg_ratio=neg_ratio)
    n_pos = sum(1 for p in pairs if p["rel_label"] != 0)
    n_neg = len(pairs) - n_pos
    print(f"  [RE] Built {len(pairs)} pairs ({n_pos} positive, {n_neg} negative)", flush=True)

    if not pairs:
        print("  [RE] WARNING: No training pairs generated. Skipping.", flush=True)
        return

    # Load tokenizer + add entity marker special tokens
    print(f"  [RE] Loading BioLinkBERT-large tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BIOLINKBERT)
    special_tokens = _special_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Load encoder and resize embeddings for new tokens
    bert = AutoModel.from_pretrained(BIOLINKBERT)
    bert.resize_token_embeddings(len(tokenizer))

    model = REPairClassifier(bert)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [RE] Model params: {n_params:,}", flush=True)

    # Tokenize all pairs
    texts = [p["text"] for p in pairs]
    rel_labels = [p["rel_label"] for p in pairs]
    dir_labels = [p["dir_label"] for p in pairs]

    encodings = tokenizer(
        texts, max_length=max_length, truncation=True,
        padding="max_length", return_tensors=None,
    )

    ds = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "rel_labels": rel_labels,
        "dir_labels": dir_labels,
    })

    # Train/eval split
    if len(ds) > 10:
        split_idx = max(2, int(len(ds) * 0.9))
        train_ds = ds.select(range(split_idx))
        eval_ds = ds.select(range(split_idx, len(ds)))
    else:
        train_ds = ds
        eval_ds = None

    # Gradient accumulation to simulate larger effective batch on GPU
    grad_accum = 2 if device_is_gpu else 1

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch" if eval_ds else "no",
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
    )

    print(f"  [RE] Starting training: {len(train_ds)} pairs, {epochs} epochs...", flush=True)
    trainer.train()
    print(f"  [RE] Training complete. Saving model...", flush=True)

    # Save: BERT encoder + tokenizer + classification heads
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save({
        "rel_head": model.rel_head.state_dict(),
        "dir_head": model.dir_head.state_dict(),
    }, out_path / "re_heads.pt")

    print(f"  [RE] Model saved to {output_dir}", flush=True)


def load_re_model(model_dir: str):
    """Load a trained RE pair classifier. Returns (model, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    bert = AutoModel.from_pretrained(model_dir)
    model = REPairClassifier(bert)

    heads_path = Path(model_dir) / "re_heads.pt"
    if heads_path.exists():
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        model.rel_head.load_state_dict(heads["rel_head"])
        model.dir_head.load_state_dict(heads["dir_head"])

    model.to(device).eval()
    return model, tokenizer


def predict_relationships(
    text: str,
    entities: list[dict],
    model_dir: str = None,
    max_length: int = 512,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """Run RE inference: enumerate entity pairs, classify each.

    Returns list of {subject, object, type, direction} for predicted positive relations.
    Pass model and tokenizer to avoid reloading on every call.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_re_model(model_dir)

    if len(entities) < 2:
        return []

    # Deduplicate entities by (text_lower, type)
    seen = {}
    unique_ents = []
    for ent in entities:
        key = (ent["text"].lower(), ent["type"])
        if key not in seen:
            seen[key] = len(unique_ents)
            unique_ents.append(ent)

    if len(unique_ents) < 2:
        return []

    device = next(model.parameters()).device

    # Build all ordered pairs
    pair_texts = []
    pair_info = []
    for i, subj in enumerate(unique_ents):
        for j, obj in enumerate(unique_ents):
            if i == j:
                continue
            marked = insert_entity_markers(text, subj, obj)
            pair_texts.append(marked)
            pair_info.append((subj, obj))

    if not pair_texts:
        return []

    # Batch inference (batch_size=64 to avoid OOM with many entities)
    all_rel_preds = []
    all_dir_preds = []
    infer_batch = 64

    for start in range(0, len(pair_texts), infer_batch):
        batch_texts = pair_texts[start:start + infer_batch]
        encodings = tokenizer(
            batch_texts, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)

        all_rel_preds.extend(outputs["rel_logits"].argmax(dim=-1).cpu().tolist())
        all_dir_preds.extend(outputs["dir_logits"].argmax(dim=-1).cpu().tolist())

    # Filter to positive predictions (not no_relation)
    results = []
    for idx, (subj, obj) in enumerate(pair_info):
        rel_id = all_rel_preds[idx]
        if rel_id != 0:
            results.append({
                "subject": subj["text"],
                "object": obj["text"],
                "type": ID2REL[rel_id],
                "direction": ID2DIR[all_dir_preds[idx]],
            })

    return results
