"""Training script for the BioExtract student model.

Designed to run on Google Colab free tier (T4 GPU, 16GB VRAM).
Fine-tunes SciFive-base on a mix of gold-standard (BioRED, BC5CDR, ChemProt)
and silver-standard (Claude-labeled) training data.

Usage (Colab):
    !pip install transformers datasets peft accelerate
    from bioextract.model.train import train
    train("data/training.jsonl", "models/bioextract_v1")
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model choices in order of preference
MODELS = {
    "scifive": "razent/SciFive-base-Pubmed_PMC",
    "flan_t5": "google/flan-t5-base",
}

DEFAULT_MODEL = "scifive"


def load_training_data(data_path: str) -> list[dict]:
    """Load JSONL training data.

    Expected format per line:
    {"text": "abstract...", "entities": [...], "relationships": [...], "source": "biored|bc5cdr|chemprot|claude"}
    """
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info("Loaded %d training examples from %s", len(examples), data_path)
    return examples


def format_for_seq2seq(examples: list[dict]) -> list[dict]:
    """Convert training examples to seq2seq input/output pairs.

    Input: "extract entities and relationships: <abstract text>"
    Output: JSON string of {entities: [...], relationships: [...]}
    """
    formatted = []
    for ex in examples:
        input_text = f"extract entities and relationships: {ex['text']}"
        output_text = json.dumps({
            "entities": ex.get("entities", []),
            "relationships": ex.get("relationships", []),
        }, ensure_ascii=False)
        formatted.append({"input": input_text, "output": output_text})
    return formatted


def train(
    data_path: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_input_length: int = 512,
    max_output_length: int = 1024,
    use_lora: bool = True,
):
    """Fine-tune the student model.

    Args:
        data_path: Path to JSONL training data.
        output_dir: Where to save the trained model.
        model_name: Key from MODELS dict or a HuggingFace model ID.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate.
        lora_r: LoRA rank (lower = fewer params, faster training).
        lora_alpha: LoRA alpha scaling.
        max_input_length: Max input token length.
        max_output_length: Max output token length.
        use_lora: Whether to use LoRA (recommended for memory efficiency).
    """
    # Deferred imports — only needed during training
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset

    # Resolve model ID
    model_id = MODELS.get(model_name, model_name)
    logger.info("Loading model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Apply LoRA if requested
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q", "v"],  # T5 attention projections
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load and format data
    examples = load_training_data(data_path)
    formatted = format_for_seq2seq(examples)

    # Split: 90% train, 10% eval
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

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

    train_dataset = Dataset.from_list(train_data).map(
        tokenize, batched=True, remove_columns=["input", "output"]
    )
    eval_dataset = Dataset.from_list(eval_data).map(
        tokenize, batched=True, remove_columns=["input", "output"]
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=max_output_length,
        logging_steps=50,
        fp16=True,  # Works on T4
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    logger.info("Starting training: %d train, %d eval examples", len(train_data), len(eval_data))
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)


def evaluate_on_biored(model_dir: str, test_path: str) -> dict:
    """Evaluate the student model on BioRED hold-out set.

    Returns dict with entity_f1, relationship_f1, direction_accuracy.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    test_data = load_training_data(test_path)

    entity_tp = entity_fp = entity_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    direction_correct = direction_total = 0

    for example in test_data:
        prompt = f"extract entities and relationships: {example['text']}"
        inputs = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=1024, num_beams=2)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            predicted = json.loads(decoded)
        except json.JSONDecodeError:
            # Count all gold as false negatives
            entity_fn += len(example.get("entities", []))
            rel_fn += len(example.get("relationships", []))
            continue

        # Entity evaluation (text + type match)
        gold_entities = {(e["text"].lower(), e["type"]) for e in example.get("entities", [])}
        pred_entities = {(e["text"].lower(), e["type"]) for e in predicted.get("entities", [])}
        entity_tp += len(gold_entities & pred_entities)
        entity_fp += len(pred_entities - gold_entities)
        entity_fn += len(gold_entities - pred_entities)

        # Relationship evaluation (subject + object + type match)
        gold_rels = {
            (r["subject"].lower(), r["object"].lower(), r["type"])
            for r in example.get("relationships", [])
        }
        pred_rels = {
            (r["subject"].lower(), r["object"].lower(), r["type"])
            for r in predicted.get("relationships", [])
        }
        rel_tp += len(gold_rels & pred_rels)
        rel_fp += len(pred_rels - gold_rels)
        rel_fn += len(gold_rels - pred_rels)

        # Direction accuracy on matching relationships
        gold_dir = {
            (r["subject"].lower(), r["object"].lower(), r["type"]): r.get("direction", "neutral")
            for r in example.get("relationships", [])
        }
        for r in predicted.get("relationships", []):
            key = (r["subject"].lower(), r["object"].lower(), r["type"])
            if key in gold_dir:
                direction_total += 1
                if r.get("direction") == gold_dir[key]:
                    direction_correct += 1

    def f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "entity_f1": round(f1(entity_tp, entity_fp, entity_fn), 4),
        "relationship_f1": round(f1(rel_tp, rel_fp, rel_fn), 4),
        "direction_accuracy": round(
            direction_correct / direction_total if direction_total > 0 else 0, 4
        ),
        "entity_counts": {"tp": entity_tp, "fp": entity_fp, "fn": entity_fn},
        "relationship_counts": {"tp": rel_tp, "fp": rel_fp, "fn": rel_fn},
    }
