#!/usr/bin/env python
"""
CamemBERT NER Training Script

Fine-tunes CamemBERT for origin/destination extraction with different dataset sizes.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.models.camembert_ner import CamembertNERTrainer, CamembertNERModel
from src.nlp.evaluator import NLPEvaluator, load_test_data

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def evaluate_model(model_path: str, test_path: str) -> Dict[str, float]:
    """Evaluate trained CamemBERT model."""
    model = CamembertNERModel(model_path)
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


def run_experiment(
    train_limit: int | None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    lr_scheduler_type: str = "linear",
    model_name: str = "camembert-base",
    output_name: str | None = None,
) -> Dict[str, Any]:
    """Run training experiment and evaluate."""
    trainer = CamembertNERTrainer(model_name=model_name)

    # Train
    model_path, train_info = trainer.train(
        train_csv="data/training/train.csv",
        val_csv="data/training/val.csv",
        train_limit=train_limit,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        output_name=output_name,
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model_path, "data/training/test.csv")

    results = {**train_info, **metrics}

    print(f"\nResults for {train_info['model_name']}:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  F1 Score:  {metrics['f1_score']:.2%}")
    print(f"  Inference: {metrics['avg_inference_time_ms']:.2f} ms")

    return results


def plot_training_curves(results: List[Dict[str, Any]], output_dir: str = "outputs/evaluation"):
    """Plot and save training/eval loss curves for all experiments."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping training curve generation")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Train loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for r in results:
        train_losses = r.get("train_losses", [])
        eval_losses = r.get("eval_losses", [])
        name = r["model_name"]

        if train_losses:
            ax1.plot(range(1, len(train_losses) + 1), train_losses, label=name, alpha=0.8)
        if eval_losses:
            ax2.plot(range(1, len(eval_losses) + 1), eval_losses, label=name, marker="o")

    ax1.set_xlabel("Logging Step")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("CamemBERT Train Loss")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Eval Loss")
    ax2.set_title("CamemBERT Eval Loss")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    output_path = Path(output_dir) / "camembert_loss_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {output_path}")


def run_all_experiments(
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
) -> List[Dict[str, Any]]:
    """
    Run all experiments: dataset sizes, LR variations, batch sizes,
    scheduler types, and DistilCamemBERT.
    """
    all_results = []

    # Experiment 1: Dataset sizes
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Impact of Dataset Size (CamemBERT)")
    print("=" * 70)

    dataset_sizes = [1000, 3000, 5000, None]

    for size in dataset_sizes:
        try:
            results = run_experiment(size, num_epochs, batch_size, learning_rate)
            all_results.append(results)
        except Exception as e:
            print(f"Error training with size {size}: {e}")

    # Experiment 2: Learning rate variations
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Learning Rate Variations")
    print("=" * 70)

    for lr in [1e-5, 3e-5, 5e-5]:
        try:
            name = f"camembert_lr{lr:.0e}"
            results = run_experiment(
                None, num_epochs, batch_size, lr,
                output_name=name,
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error with LR {lr}: {e}")

    # Experiment 3: Batch size variations
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Batch Size Variations")
    print("=" * 70)

    for bs in [8, 16, 32]:
        try:
            name = f"camembert_bs{bs}"
            results = run_experiment(
                None, num_epochs, bs, learning_rate,
                output_name=name,
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error with batch size {bs}: {e}")

    # Experiment 4: Scheduler variations
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: LR Scheduler Variations")
    print("=" * 70)

    for scheduler in ["linear", "cosine"]:
        try:
            name = f"camembert_sched_{scheduler}"
            results = run_experiment(
                None, num_epochs, batch_size, learning_rate,
                lr_scheduler_type=scheduler,
                output_name=name,
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error with scheduler {scheduler}: {e}")

    # Experiment 5: DistilCamemBERT
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: DistilCamemBERT")
    print("=" * 70)

    try:
        results = run_experiment(
            None, num_epochs, batch_size, learning_rate,
            model_name="cmarkea/distilcamembert-base",
            output_name="distilcamembert_ner_full",
        )
        all_results.append(results)
    except Exception as e:
        print(f"Error training DistilCamemBERT: {e}")

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
    parser = argparse.ArgumentParser(description="Train CamemBERT NER model")
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training samples (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant"],
        help="LR scheduler type (default: linear)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="camembert-base",
        help="Pretrained model name (default: camembert-base)",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Run all predefined experiments",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/camembert_experiments.json",
        help="Path to save results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CAMEMBERT NER MODEL TRAINING")
    print("=" * 70)

    if args.all_experiments:
        # Run all experiments
        results = run_all_experiments(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print_comparison_table(results)
        plot_training_curves(results)
        save_results(results, args.output)
    else:
        # Run single experiment
        results = run_experiment(
            train_limit=args.train_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler,
            model_name=args.model_name,
        )
        save_results([results], args.output)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
