"""Progressive student model training harness.

Trains the split pipeline (BioLinkBERT-large NER + BioLinkBERT-large RE pair classifier)
at progressive scales (1, 10, 100, 400 examples) to measure learning curves.

NER uses BIOES tagging with sliding window for full abstract coverage.
RE uses typed entity markers with dual classification heads.

Designed to run on Google Colab (T4 GPU). See notebooks/train_progressive.ipynb.

Usage (Colab):
    from bioextract.model.train_progressive import run_progressive
    results = run_progressive(scales=[1, 10])
"""

from __future__ import annotations

import json
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


def recover_spans(text: str, entities: list[dict]) -> list[dict]:
    """Recover start/end character offsets via string matching.

    For entities without spans (e.g., from LLM output), find them in the
    original text using case-insensitive search.
    """
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


def train_and_evaluate(
    subset: list[dict],
    test_data: list[dict],
    ner_dir: str,
    re_dir: str,
    epochs: int,
) -> dict:
    """Train and evaluate the split pipeline (NER + RE).

    Returns dict with end_to_end metrics, ner_only metrics, and re_with_gold metrics.
    """
    from .train_ner import train_ner, predict_entities, load_ner_model
    from .train_re import train_re, predict_relationships, load_re_model

    # Stage 1: Train NER
    _log(f"  Stage 1: Training NER ({len(subset)} examples, {epochs} epochs)...")
    train_ner(subset, ner_dir, epochs=epochs)
    _log(f"  NER training complete.")

    # Stage 2: Train RE (with gold entities)
    _log(f"  Stage 2: Training RE ({len(subset)} examples, {epochs} epochs)...")
    train_re(subset, re_dir, epochs=epochs)
    _log(f"  RE training complete.")

    # --- Evaluation (load models once) ---

    _log(f"  Loading NER model for evaluation...")
    ner_model, ner_tokenizer = load_ner_model(ner_dir)
    _log(f"  Loading RE model for evaluation...")
    re_model, re_tokenizer = load_re_model(re_dir)

    n_total = len(test_data)

    # NER-only evaluation
    _log(f"  Evaluating NER on {n_total} test examples...")
    ner_predictions = []
    for i, ex in enumerate(test_data):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"    NER eval: {i + 1}/{n_total}")
        pred_entities = predict_entities(ex["text"], model=ner_model, tokenizer=ner_tokenizer)
        ner_predictions.append(pred_entities)

    gold_entity_lists = [ex.get("entities", []) for ex in test_data]
    ner_metrics = compute_ner_metrics(ner_predictions, gold_entity_lists)
    _log(f"  NER eval done. Entity F1: {ner_metrics['entity_f1']:.4f}")

    # RE with gold entities (diagnostic)
    _log(f"  Evaluating RE (gold entities) on {n_total} test examples...")
    re_gold_predictions = []
    for i, ex in enumerate(test_data):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"    RE (gold entities) eval: {i + 1}/{n_total}")
        pred_rels = predict_relationships(ex["text"], ex.get("entities", []),
                                          model=re_model, tokenizer=re_tokenizer)
        re_gold_predictions.append({"entities": ex.get("entities", []), "relationships": pred_rels})

    re_gold_metrics = compute_metrics(re_gold_predictions, test_data)
    _log(f"  RE (gold entities) eval done. Rel F1: {re_gold_metrics['relationship_f1']:.4f}")

    # End-to-end evaluation (NER errors propagate to RE)
    _log(f"  Evaluating end-to-end pipeline on {n_total} test examples...")
    e2e_predictions = []
    for i, (ex, pred_entities) in enumerate(zip(test_data, ner_predictions)):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"    E2E eval: {i + 1}/{n_total}")
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
    train_path: str | None = None,
    test_path: str | None = None,
    models_dir: str | None = None,
):
    """Run the full progressive training experiment.

    Trains the split pipeline (BioLinkBERT-large NER + RE pair classifier) at each
    scale and evaluates on the test set.

    Args:
        scales: List of training sizes to evaluate (default: [1, 10, 100, 400]).
        train_path: Path to training JSONL.
        test_path: Path to test JSONL.
        models_dir: Base directory for saving models.
    """
    scales = scales or list(SCALES.keys())
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
        ner_dir = str(models_dir / f"ner_n{n}")
        re_dir = str(models_dir / f"re_n{n}")

        t0 = time.time()
        metrics = train_and_evaluate(subset, test_data, ner_dir, re_dir, epochs)
        total_time = time.time() - t0

        metrics["total_time_sec"] = round(total_time, 1)
        e2e = metrics["end_to_end"]
        ner = metrics["ner_only"]
        _log(f"\n>>> N={n} DONE: e2e_entity_f1={e2e['entity_f1']:.4f}, "
             f"e2e_rel_f1={e2e['relationship_f1']:.4f}, ner_f1={ner['entity_f1']:.4f}, "
             f"total={total_time:.0f}s")

        results[f"n{n}"] = {
            "n": n,
            "epochs": epochs,
            "n_actual": len(subset),
            **metrics,
        }

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
    print("PROGRESSIVE TRAINING RESULTS — Split Pipeline (BioLinkBERT-large NER + RE)")
    print("=" * 80)
    print(f"{'N':>5} | {'NER F1':>8} {'E2E Ent F1':>11} {'E2E Rel F1':>11} {'RE(gold) F1':>12} {'Dir Acc':>8} | {'Time':>6}")
    print("-" * 80)

    for key in sorted(results.keys(), key=lambda k: results[k]["n"]):
        r = results[key]
        n = r["n"]

        ner = r.get("ner_only", {})
        e2e = r.get("end_to_end", {})
        re_gold = r.get("re_with_gold_entities", {})

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

        time_str = f"{r.get('total_time_sec', 0):.0f}s"

        print(f"{n:>5} | {fmt(ner.get('entity_f1', '-')):>8} "
              f"{fmt(e2e.get('entity_f1', '-')):>11} "
              f"{fmt(e2e.get('relationship_f1', '-')):>11} "
              f"{fmt(re_gold.get('relationship_f1', '-')):>12} "
              f"{fmt(e2e.get('direction_accuracy', '-')):>8} | "
              f"{time_str:>6}")

    print("=" * 80, flush=True)
