"""Progressive student model training harness.

Compares two approaches at progressive scales (1, 10, 100, 400 examples):
  - Approach A: Single generative model (Flan-T5-base, seq2seq with LoRA)
  - Approach B: Split pipeline (PubMedBERT NER + Flan-T5-base RE with LoRA)

Designed to run on Google Colab (T4 GPU). See notebooks/train_progressive.ipynb.

Usage (Colab):
    from bioextract.model.train_progressive import run_progressive
    results = run_progressive(scales=[1, 10], approaches=["A", "B"])
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .data_utils import compute_metrics, compute_ner_metrics, diverse_sample, load_data

# Defaults
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_PATH = DATA_DIR / "biored_train.jsonl"
TEST_PATH = DATA_DIR / "biored_test.jsonl"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

SCALES = {
    1: 50,
    10: 20,
    100: 10,
    400: 5,
}

# Gold label sets
GOLD_ENTITY_TYPES = {"GENE", "DISEASE", "CHEMICAL", "VARIANT", "ORGANISM", "CELL_TYPE"}
GOLD_REL_TYPES = {"associated_with", "binds", "upregulates", "downregulates", "regulates", "interacts_with"}


def _log(msg: str):
    """Print with flush so output appears immediately in Colab."""
    print(msg, flush=True)


def filter_to_gold_labels(examples: list[dict]) -> list[dict]:
    """Filter entities and relationships to the gold label set only."""
    filtered = []
    for ex in examples:
        new_ex = {**ex}
        new_ex["entities"] = [
            e for e in ex.get("entities", [])
            if e["type"] in GOLD_ENTITY_TYPES
        ]
        new_ex["relationships"] = [
            r for r in ex.get("relationships", [])
            if r["type"] in GOLD_REL_TYPES
        ]
        filtered.append(new_ex)
    return filtered


def _format_seq2seq_output(example: dict) -> str:
    """Format gold output for Approach A (no start/end, gold label set only)."""
    entities = [{"text": e["text"], "type": e["type"]} for e in example.get("entities", [])]
    relationships = [
        {
            "subject": r["subject"],
            "object": r["object"],
            "type": r["type"],
            "direction": r.get("direction", "neutral"),
        }
        for r in example.get("relationships", [])
    ]
    return json.dumps({"entities": entities, "relationships": relationships}, ensure_ascii=False)


def _recover_spans(text: str, entities: list[dict]) -> list[dict]:
    """Post-process Approach A output: recover start/end via string matching."""
    result = []
    for ent in entities:
        ent_text = ent.get("text", "")
        ent_type = ent.get("type", "")
        # Case-insensitive search in original text
        lower_text = text.lower()
        lower_ent = ent_text.lower()
        idx = lower_text.find(lower_ent)
        if idx >= 0:
            result.append({
                "text": text[idx:idx + len(ent_text)],
                "type": ent_type,
                "start": idx,
                "end": idx + len(ent_text),
            })
        else:
            # Keep without span if not found
            result.append({"text": ent_text, "type": ent_type, "start": 0, "end": 0})
    return result


def train_approach_a(subset: list[dict], output_dir: str, epochs: int):
    """Train Approach A: Single Flan-T5-base generative model with LoRA."""
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType

    import torch
    device_is_gpu = torch.cuda.is_available()
    batch_size = 4 if device_is_gpu else min(4, max(1, len(subset)))

    _log(f"[Approach A] Loading Flan-T5-base model... (gpu={device_is_gpu})")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)

    # Format data
    formatted = []
    for ex in subset:
        input_text = f"extract entities and relationships: {ex['text']}"
        output_text = _format_seq2seq_output(ex)
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
            batch["input"], max_length=512, truncation=True, padding="max_length",
        )
        targets = tokenizer(
            batch["output"], max_length=1024, truncation=True, padding="max_length",
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
        learning_rate=3e-4,
        weight_decay=0.01,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        predict_with_generate=True,
        generation_max_length=1024,
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

    _log(f"[Approach A] Starting training: {len(train_data)} train examples, {epochs} epochs...")
    trainer.train()
    _log(f"[Approach A] Training complete. Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate_approach_a(model_dir: str, test_data: list[dict]) -> dict:
    """Evaluate Approach A on test data."""
    import torch
    from transformers import AutoTokenizer
    from peft import AutoPeftModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"[Approach A] Loading model for evaluation (device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_dir)
    except Exception:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device).eval()

    n_total = len(test_data)
    _log(f"[Approach A] Evaluating on {n_total} test examples...")
    predictions = []
    parse_failures = 0
    for i, ex in enumerate(test_data):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"  Approach A eval: {i + 1}/{n_total}")
        prompt = f"extract entities and relationships: {ex['text']}"
        inputs = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, num_beams=1)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            data = json.loads(decoded)
            # Recover spans for entities
            data["entities"] = _recover_spans(ex["text"], data.get("entities", []))
            predictions.append(data)
        except json.JSONDecodeError:
            parse_failures += 1
            predictions.append(None)

    _log(f"  Approach A eval done. JSON parse failures: {parse_failures}/{n_total}")
    return compute_metrics(predictions, test_data)


def train_and_evaluate_approach_b(
    subset: list[dict],
    test_data: list[dict],
    ner_dir: str,
    re_dir: str,
    epochs: int,
) -> dict:
    """Train and evaluate Approach B: NER + RE split pipeline.

    Returns dict with end_to_end metrics, ner_only metrics, and re_with_gold metrics.
    """
    from .train_ner import train_ner, predict_entities, load_ner_model
    from .train_re import train_re, predict_relationships, load_re_model

    # Stage 1: Train NER
    _log(f"[Approach B] Stage 1: Training NER ({len(subset)} examples, {epochs} epochs)...")
    train_ner(subset, ner_dir, epochs=epochs)
    _log(f"[Approach B] NER training complete.")

    # Stage 2: Train RE (with gold entities)
    _log(f"[Approach B] Stage 2: Training RE ({len(subset)} examples, {epochs} epochs)...")
    train_re(subset, re_dir, epochs=epochs)
    _log(f"[Approach B] RE training complete.")

    # --- Evaluation (load models once) ---

    _log(f"[Approach B] Loading NER model for evaluation...")
    ner_model, ner_tokenizer = load_ner_model(ner_dir)
    _log(f"[Approach B] Loading RE model for evaluation...")
    re_model, re_tokenizer = load_re_model(re_dir)

    n_total = len(test_data)

    # NER-only evaluation
    _log(f"[Approach B] Evaluating NER on {n_total} test examples...")
    ner_predictions = []
    for i, ex in enumerate(test_data):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"  NER eval: {i + 1}/{n_total}")
        pred_entities = predict_entities(ex["text"], model=ner_model, tokenizer=ner_tokenizer)
        ner_predictions.append(pred_entities)

    gold_entity_lists = [ex.get("entities", []) for ex in test_data]
    ner_metrics = compute_ner_metrics(ner_predictions, gold_entity_lists)
    _log(f"  NER eval done. Entity F1: {ner_metrics['entity_f1']:.4f}")

    # RE with gold entities (diagnostic)
    _log(f"[Approach B] Evaluating RE (gold entities) on {n_total} test examples...")
    re_gold_predictions = []
    for i, ex in enumerate(test_data):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"  RE (gold entities) eval: {i + 1}/{n_total}")
        pred_rels = predict_relationships(ex["text"], ex.get("entities", []),
                                          model=re_model, tokenizer=re_tokenizer)
        re_gold_predictions.append({"entities": ex.get("entities", []), "relationships": pred_rels})

    re_gold_metrics = compute_metrics(re_gold_predictions, test_data)
    _log(f"  RE (gold entities) eval done. Rel F1: {re_gold_metrics['relationship_f1']:.4f}")

    # End-to-end evaluation (NER errors propagate to RE)
    _log(f"[Approach B] Evaluating end-to-end pipeline on {n_total} test examples...")
    e2e_predictions = []
    for i, (ex, pred_entities) in enumerate(zip(test_data, ner_predictions)):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"  E2E eval: {i + 1}/{n_total}")
        pred_rels = predict_relationships(ex["text"], pred_entities,
                                          model=re_model, tokenizer=re_tokenizer)
        e2e_predictions.append({"entities": pred_entities, "relationships": pred_rels})

    e2e_metrics = compute_metrics(e2e_predictions, test_data)
    _log(f"  E2E eval done. Entity F1: {e2e_metrics['entity_f1']:.4f}, Rel F1: {e2e_metrics['relationship_f1']:.4f}")

    return {
        "end_to_end": e2e_metrics,
        "ner_only": ner_metrics,
        "re_with_gold_entities": re_gold_metrics,
    }


def run_progressive(
    scales: list[int] | None = None,
    approaches: list[str] | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    models_dir: str | None = None,
):
    """Run the full progressive training experiment.

    Args:
        scales: List of training sizes to evaluate (default: [1, 10, 100, 400]).
        approaches: Which approaches to run: ["A"], ["B"], or ["A", "B"] (default: both).
        train_path: Path to training JSONL.
        test_path: Path to test JSONL.
        models_dir: Base directory for saving models.
    """
    scales = scales or list(SCALES.keys())
    approaches = approaches or ["A", "B"]
    train_path = train_path or str(TRAIN_PATH)
    test_path = test_path or str(TEST_PATH)
    models_dir = Path(models_dir or MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    _log(f"Loading training data from {train_path}")
    all_train = load_data(train_path)
    _log(f"Loading test data from {test_path}")
    test_data = load_data(test_path)

    # Filter to gold label set
    all_train = filter_to_gold_labels(all_train)
    test_data = filter_to_gold_labels(test_data)
    _log(f"Data loaded: {len(all_train)} train, {len(test_data)} test (filtered to gold labels)")

    results = {}

    for n in scales:
        epochs = SCALES.get(n, 5)
        _log(f"\n{'=' * 60}")
        _log(f"SCALE N={n}, epochs={epochs}")
        _log(f"{'=' * 60}")

        # Diverse sample
        subset = diverse_sample(all_train, n)
        scale_results = {"n": n, "epochs": epochs, "n_actual": len(subset)}

        if "A" in approaches:
            _log(f"\n--- Approach A: Single Generative Model (N={n}) ---")
            a_dir = str(models_dir / f"flan_t5_n{n}")
            t0 = time.time()
            train_approach_a(subset, a_dir, epochs)
            train_time = time.time() - t0
            _log(f"[Approach A] Training done in {train_time:.1f}s. Starting evaluation...")

            t0 = time.time()
            a_metrics = evaluate_approach_a(a_dir, test_data)
            eval_time = time.time() - t0

            a_metrics["train_time_sec"] = round(train_time, 1)
            a_metrics["eval_time_sec"] = round(eval_time, 1)
            scale_results["approach_a"] = a_metrics
            _log(f"\n>>> Approach A DONE (N={n}): entity_f1={a_metrics['entity_f1']:.4f}, "
                 f"rel_f1={a_metrics['relationship_f1']:.4f}, parse_rate={a_metrics['json_parse_rate']:.4f}, "
                 f"train={train_time:.0f}s, eval={eval_time:.0f}s")

        if "B" in approaches:
            _log(f"\n--- Approach B: Split Pipeline (N={n}) ---")
            ner_dir = str(models_dir / f"ner_n{n}")
            re_dir = str(models_dir / f"re_n{n}")
            t0 = time.time()
            b_metrics = train_and_evaluate_approach_b(subset, test_data, ner_dir, re_dir, epochs)
            total_time = time.time() - t0

            b_metrics["total_time_sec"] = round(total_time, 1)
            scale_results["approach_b"] = b_metrics
            e2e = b_metrics["end_to_end"]
            ner = b_metrics["ner_only"]
            _log(f"\n>>> Approach B DONE (N={n}): e2e_entity_f1={e2e['entity_f1']:.4f}, "
                 f"e2e_rel_f1={e2e['relationship_f1']:.4f}, ner_f1={ner['entity_f1']:.4f}, "
                 f"total={total_time:.0f}s")

        results[f"n{n}"] = scale_results

    # Save results
    results_path = models_dir / "progressive_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"\nResults saved to {results_path}")

    # Print summary table
    _print_summary(results)

    return results


def _print_summary(results: dict):
    """Print a comparison summary table."""
    print("\n" + "=" * 80)
    print("PROGRESSIVE TRAINING RESULTS")
    print("=" * 80)
    print(f"{'N':>5} | {'Approach A':^40} | {'Approach B (E2E)':^30}")
    print(f"{'':>5} | {'Ent F1':>8} {'Rel F1':>8} {'Dir Acc':>8} {'Parse%':>8} | {'Ent F1':>8} {'Rel F1':>8} {'NER F1':>8}")
    print("-" * 80)

    for key in sorted(results.keys(), key=lambda k: results[k]["n"]):
        r = results[key]
        n = r["n"]

        a = r.get("approach_a", {})
        b_e2e = r.get("approach_b", {}).get("end_to_end", {})
        b_ner = r.get("approach_b", {}).get("ner_only", {})

        a_str = f"{a.get('entity_f1', '-'):>8} {a.get('relationship_f1', '-'):>8} {a.get('direction_accuracy', '-'):>8} {a.get('json_parse_rate', '-'):>8}"
        b_str = f"{b_e2e.get('entity_f1', '-'):>8} {b_e2e.get('relationship_f1', '-'):>8} {b_ner.get('entity_f1', '-'):>8}"

        print(f"{n:>5} | {a_str} | {b_str}")

    print("=" * 80, flush=True)
