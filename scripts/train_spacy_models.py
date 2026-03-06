#!/usr/bin/env python
"""
spaCy NER Model Training Script

Trains multiple spaCy NER models with different configurations:
1. Different base models: blank, sm, md, lg
2. Different dataset sizes: 1000, 5000, 7000 (full)

"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.models.spacy_ner import (
    SpacyNERTrainer,
    SpacyNERModel,
    train_model_experiment,
)
from src.nlp.evaluator import NLPEvaluator, load_test_data

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def evaluate_model(model_path: str, test_path: str) -> Dict[str, float]:
    """Evaluate a trained spaCy model."""
    model = SpacyNERModel(model_path)
    evaluator = NLPEvaluator(verbose=False)
    test_data = load_test_data(test_path)

    result = evaluator.evaluate(model, test_data)

    return {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "origin_f1": result.origin_f1,
        "destination_f1": result.destination_f1,
        "validity_accuracy": result.validity_accuracy,
        "avg_inference_time_ms": result.avg_inference_time_ms,
    }


def run_single_experiment(
    base_model: str,
    train_size: int | None,
    n_iter: int = 30,
) -> Dict[str, Any]:
    """Run a single training experiment and evaluate."""

    # Train model
    model_path, train_info = train_model_experiment(
        base_model=base_model,
        train_limit=train_size,
        n_iter=n_iter,
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model_path, "data/training/test.csv")

    # Combine results
    results = {**train_info, **metrics}

    print(f"\nResults for {train_info['model_name']}:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  F1 Score:  {metrics['f1_score']:.2%}")
    print(f"  Inference: {metrics['avg_inference_time_ms']:.2f} ms")

    return results


def run_all_experiments(n_iter: int = 30) -> List[Dict[str, Any]]:
    """
    Run all experiments:
    - Different base models with full dataset
    - Different dataset sizes with sm model
    """
    all_results = []

    # Experiment 1: Different dataset sizes with blank model
    print("\n" + "=" * 70)
    print("EXPERIMENT SET 1: Impact of Dataset Size (blank model)")
    print("=" * 70)

    dataset_sizes = [1000, 3000, 5000, None]  # None = full dataset

    for size in dataset_sizes:
        try:
            results = run_single_experiment("blank", size, n_iter)
            all_results.append(results)
        except Exception as e:
            print(f"Error training with size {size}: {e}")

    # Experiment 2: Different base models with full dataset
    print("\n" + "=" * 70)
    print("EXPERIMENT SET 2: Impact of Base Model (full dataset)")
    print("=" * 70)

    base_models = ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"]

    for base_model in base_models:
        try:
            results = run_single_experiment(base_model, None, n_iter)
            all_results.append(results)
        except Exception as e:
            print(f"Error training with {base_model}: {e}")

    return all_results


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print comparison table of all experiments."""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS COMPARISON")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<25} {'Samples':>8} {'Accuracy':>10} {'F1':>10} "
          f"{'Origin F1':>10} {'Dest F1':>10} {'Time(s)':>10}")
    print("-" * 100)

    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)

    for r in sorted_results:
        print(f"{r['model_name']:<25} {r['train_samples']:>8} "
              f"{r['accuracy']:>9.2%} {r['f1_score']:>9.2%} "
              f"{r['origin_f1']:>9.2%} {r['destination_f1']:>9.2%} "
              f"{r['training_time_s']:>9.1f}")

    print("-" * 100)

    # Best model
    best = sorted_results[0]
    print(f"\nBest model: {best['model_name']}")
    print(f"  - F1 Score: {best['f1_score']:.2%}")
    print(f"  - Accuracy: {best['accuracy']:.2%}")


def plot_loss_curves(results: List[Dict[str, Any]], output_dir: str = "outputs/evaluation"):
    """Plot and save loss curves for all experiments."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping loss curve generation")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        loss_history = r.get("loss_history", [])
        if loss_history:
            ax.plot(range(1, len(loss_history) + 1), loss_history, label=r["model_name"])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("NER Loss")
    ax.set_title("spaCy NER Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / "spacy_loss_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved to {output_path}")


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save experiment results to JSON, accumulating with existing results."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if file exists
    existing_results = []
    if Path(output_path).exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []

    # Create a dict of existing results by model_name for easy lookup
    existing_dict = {r['model_name']: r for r in existing_results}

    # Add or update results
    for result in results:
        model_name = result['model_name']
        if model_name in existing_dict:
            print(f"  Updating existing results for {model_name}")
        else:
            print(f"  Adding new results for {model_name}")
        existing_dict[model_name] = result

    # Convert back to list and sort by model_name for consistency
    all_results = sorted(existing_dict.values(), key=lambda x: x['model_name'])

    # Save all results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nTotal {len(all_results)} model(s) saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train spaCy NER models")
    parser.add_argument(
        "--base-model",
        type=str,
        default="blank",
        choices=["blank", "fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"],
        help="Base spaCy model to use",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training samples (default: all)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Run all predefined experiments",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/spacy_experiments.json",
        help="Path to save experiment results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SPACY NER MODEL TRAINING")
    print("=" * 70)

    if args.all_experiments:
        # Run all experiments
        results = run_all_experiments(n_iter=args.n_iter)
        print_comparison_table(results)
        plot_loss_curves(results)
        save_results(results, args.output)
    else:
        # Run single experiment
        results = run_single_experiment(
            base_model=args.base_model,
            train_size=args.train_size,
            n_iter=args.n_iter,
        )
        save_results([results], args.output)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
