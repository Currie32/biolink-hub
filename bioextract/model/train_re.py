"""Flan-T5 relationship extraction training for Approach B (split pipeline).

Stage 2: Seq2seq with LoRA, conditioned on entity mentions.
Input format: "extract relationships: [TYPE: text] [TYPE: text] ... Text: {abstract}"
Output format: JSON {"relationships": [...]}

Designed to run on Google Colab (T4 GPU). Called by train_progressive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

FLAN_T5_BASE = "google/flan-t5-base"


def format_re_input(text: str, entities: list[dict]) -> str:
    """Build RE input with entity markers.

    Format: "extract relationships: [GENE: TREM2] [DISEASE: AD] Text: {abstract}"
    Deduplicate entities by (text, type) to avoid repetition.
    """
    seen = set()
    markers = []
    for ent in entities:
        key = (ent["text"], ent["type"])
        if key not in seen:
            seen.add(key)
            markers.append(f"[{ent['type']}: {ent['text']}]")

    marker_str = " ".join(markers)
    return f"extract relationships: {marker_str} Text: {text}"


def _format_re_output(relationships: list[dict]) -> str:
    """Format relationships as JSON output string."""
    clean_rels = []
    for r in relationships:
        clean_rels.append({
            "subject": r["subject"],
            "object": r["object"],
            "type": r["type"],
            "direction": r.get("direction", "neutral"),
        })
    return json.dumps({"relationships": clean_rels}, ensure_ascii=False)


def train_re(
    examples: list[dict],
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_input_length: int = 512,
    max_output_length: int = 512,
):
    """Fine-tune Flan-T5-base with LoRA for relationship extraction.

    Trained with gold entities as input markers.

    Args:
        examples: Training examples with text, entities, relationships.
        output_dir: Where to save the model.
        epochs: Training epochs.
        batch_size: Per-device batch size (capped at 4 on CPU).
        learning_rate: Learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        max_input_length: Max input tokens.
        max_output_length: Max output tokens (512 — shorter than full pipeline since no entities).
    """
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType

    device_is_gpu = torch.cuda.is_available()
    if not device_is_gpu:
        batch_size = min(batch_size, 4)

    logger.info("Training RE: %d examples, %d epochs, batch_size=%d, gpu=%s",
                len(examples), epochs, batch_size, device_is_gpu)

    logger.info("Loading Flan-T5-base tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_T5_BASE)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format data
    formatted = []
    for ex in examples:
        input_text = format_re_input(ex["text"], ex.get("entities", []))
        output_text = _format_re_output(ex.get("relationships", []))
        formatted.append({"input": input_text, "output": output_text})

    # Train/eval split
    if len(formatted) > 2:
        split_idx = max(1, int(len(formatted) * 0.9))
        train_data = formatted[:split_idx]
        eval_data = formatted[split_idx:]
    else:
        train_data = formatted
        eval_data = None

    def tokenize(batch):
        inputs = tokenizer(
            batch["input"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            batch["output"],
            max_length=max_output_length,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    train_ds = Dataset.from_list(train_data).map(
        tokenize, batched=True, remove_columns=["input", "output"]
    )
    eval_ds = None
    if eval_data:
        eval_ds = Dataset.from_list(eval_data).map(
            tokenize, batched=True, remove_columns=["input", "output"]
        )

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        generation_max_length=max_output_length,
        logging_steps=max(1, len(train_ds) // batch_size // 5),
        fp16=device_is_gpu,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    logger.info("Starting RE training...")
    trainer.train()
    logger.info("RE training complete. Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("RE model saved to %s", output_dir)


def load_re_model(model_dir: str):
    """Load a trained RE model and tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer
    from peft import AutoPeftModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_dir)
    except Exception:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_relationships(
    text: str,
    entities: list[dict],
    model_dir: str = None,
    max_input_length: int = 512,
    max_output_length: int = 512,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """Run RE inference with entity-conditioned input.

    Returns list of {"subject": str, "object": str, "type": str, "direction": str}.

    Pass model and tokenizer to avoid reloading on every call.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_re_model(model_dir)

    input_text = format_re_input(text, entities)
    inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output_length,
            num_beams=2,
            early_stopping=True,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        data = json.loads(decoded)
        return data.get("relationships", [])
    except json.JSONDecodeError:
        logger.warning("RE model produced invalid JSON: %s", decoded[:200])
        return []
