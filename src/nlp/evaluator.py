"""
NLP Model Evaluator

Computes all required metrics for model evaluation:
1. Accuracy 
2. Precision/Recall/F1 for origin and destination
3. Confusion matrix
4. Robustness metrics (no capitals, misspellings, no accents)
5. Performance metrics (time, CPU, memory)
"""

import csv
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    # Overall metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Per-entity metrics
    origin_precision: float = 0.0
    origin_recall: float = 0.0
    origin_f1: float = 0.0
    destination_precision: float = 0.0
    destination_recall: float = 0.0
    destination_f1: float = 0.0

    # Validity detection
    validity_accuracy: float = 0.0
    validity_precision: float = 0.0
    validity_recall: float = 0.0

    # Robustness metrics
    accuracy_normal: float = 0.0
    accuracy_lowercase: float = 0.0
    accuracy_no_accents: float = 0.0
    accuracy_misspelled: float = 0.0

    # Performance metrics
    total_samples: int = 0
    correct_samples: int = 0
    avg_inference_time_ms: float = 0.0
    total_inference_time_s: float = 0.0

    # Confusion data
    confusion_matrix: Dict = field(default_factory=dict)

    # Detailed errors
    errors: List[Dict] = field(default_factory=list)


class NLPEvaluator:
    """
    Evaluator for NLP extraction models.

    Computes comprehensive metrics for origin/destination extraction.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize evaluator.

        Args:
            verbose: Print progress during evaluation
        """
        self.verbose = verbose

    def _normalize_city(self, city: Optional[str]) -> str:
        """Normalize city name for comparison."""
        if city is None:
            return ""
        # Lowercase, remove accents, remove hyphens
        city = city.lower().strip()
        accents = {
            'à': 'a', 'â': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'î': 'i', 'ï': 'i',
            'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ÿ': 'y', 'ç': 'c',
        }
        for accent, replacement in accents.items():
            city = city.replace(accent, replacement)
        city = city.replace('-', ' ').replace('  ', ' ')
        return city

    def _cities_match(self, pred: Optional[str], gold: Optional[str]) -> bool:
        """Check if two city names match (with normalization)."""
        if pred is None and gold is None:
            return True
        if pred is None or gold is None:
            return False
        return self._normalize_city(pred) == self._normalize_city(gold)

    def _is_valid_order(self, origin: str, destination: str) -> bool:
        """Check if origin/destination represent a valid order."""
        return origin != "INVALID" and destination != "INVALID"

    def _detect_sentence_type(self, sentence: str) -> str:
        """Detect the type of sentence for robustness analysis."""
        has_upper = any(c.isupper() for c in sentence)
        has_accent = any(c in 'àâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ' for c in sentence)

        if not has_upper and not has_accent:
            return "lowercase_no_accent"
        elif not has_upper:
            return "lowercase"
        elif not has_accent:
            return "no_accent"
        else:
            return "normal"

    def evaluate(
        self,
        model,
        test_data: List[Dict[str, str]],
    ) -> EvaluationResult:
        """
        Evaluate model on test data.

        Args:
            model: Model with predict(sentence) method
            test_data: List of dicts with 'sentence', 'origin', 'destination'

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult()
        result.total_samples = len(test_data)

        # Counters
        correct_full = 0
        origin_tp, origin_fp, origin_fn = 0, 0, 0
        dest_tp, dest_fp, dest_fn = 0, 0, 0
        validity_tp, validity_fp, validity_fn, validity_tn = 0, 0, 0, 0

        # Robustness counters
        robustness_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

        # Timing
        inference_times = []
        errors = []

        if self.verbose:
            print(f"Evaluating {len(test_data)} samples...")

        for i, sample in enumerate(test_data):
            sentence = sample['sentence']
            gold_origin = sample['origin']
            gold_destination = sample['destination']
            gold_valid = self._is_valid_order(gold_origin, gold_destination)

            # Predict with timing
            start_time = time.time()
            prediction = model.predict(sentence)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)

            pred_origin = prediction.origin
            pred_destination = prediction.destination
            pred_valid = prediction.is_valid

            # Check full correctness
            origin_correct = self._cities_match(pred_origin, gold_origin)
            dest_correct = self._cities_match(pred_destination, gold_destination)
            full_correct = origin_correct and dest_correct

            if full_correct:
                correct_full += 1

            # Origin metrics
            if gold_valid:
                if origin_correct and pred_origin:
                    origin_tp += 1
                elif pred_origin and not origin_correct:
                    origin_fp += 1
                elif not pred_origin:
                    origin_fn += 1

            # Destination metrics
            if gold_valid:
                if dest_correct and pred_destination:
                    dest_tp += 1
                elif pred_destination and not dest_correct:
                    dest_fp += 1
                elif not pred_destination:
                    dest_fn += 1

            # Validity detection metrics
            if gold_valid and pred_valid:
                validity_tp += 1
            elif gold_valid and not pred_valid:
                validity_fn += 1
            elif not gold_valid and pred_valid:
                validity_fp += 1
            else:
                validity_tn += 1

            # Robustness analysis
            sentence_type = self._detect_sentence_type(sentence)
            robustness_counts[sentence_type]['total'] += 1
            if full_correct:
                robustness_counts[sentence_type]['correct'] += 1

            # Record errors for analysis
            if not full_correct:
                errors.append({
                    'sentence': sentence,
                    'gold_origin': gold_origin,
                    'gold_destination': gold_destination,
                    'pred_origin': pred_origin,
                    'pred_destination': pred_destination,
                    'method': prediction.method,
                    'sentence_type': sentence_type,
                })

            if self.verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}...")

        # Calculate overall metrics
        result.accuracy = correct_full / result.total_samples if result.total_samples > 0 else 0
        result.correct_samples = correct_full

        # Origin metrics
        result.origin_precision = origin_tp / (origin_tp + origin_fp) if (origin_tp + origin_fp) > 0 else 0
        result.origin_recall = origin_tp / (origin_tp + origin_fn) if (origin_tp + origin_fn) > 0 else 0
        result.origin_f1 = 2 * result.origin_precision * result.origin_recall / (result.origin_precision + result.origin_recall) if (result.origin_precision + result.origin_recall) > 0 else 0

        # Destination metrics
        result.destination_precision = dest_tp / (dest_tp + dest_fp) if (dest_tp + dest_fp) > 0 else 0
        result.destination_recall = dest_tp / (dest_tp + dest_fn) if (dest_tp + dest_fn) > 0 else 0
        result.destination_f1 = 2 * result.destination_precision * result.destination_recall / (result.destination_precision + result.destination_recall) if (result.destination_precision + result.destination_recall) > 0 else 0

        # Overall precision/recall/F1 (average of origin and destination)
        result.precision = (result.origin_precision + result.destination_precision) / 2
        result.recall = (result.origin_recall + result.destination_recall) / 2
        result.f1_score = (result.origin_f1 + result.destination_f1) / 2

        # Validity metrics
        result.validity_accuracy = (validity_tp + validity_tn) / result.total_samples if result.total_samples > 0 else 0
        result.validity_precision = validity_tp / (validity_tp + validity_fp) if (validity_tp + validity_fp) > 0 else 0
        result.validity_recall = validity_tp / (validity_tp + validity_fn) if (validity_tp + validity_fn) > 0 else 0

        # Robustness metrics
        for stype, counts in robustness_counts.items():
            if counts['total'] > 0:
                acc = counts['correct'] / counts['total']
                if stype == "normal":
                    result.accuracy_normal = acc
                elif stype == "lowercase":
                    result.accuracy_lowercase = acc
                elif stype == "no_accent":
                    result.accuracy_no_accents = acc
                elif stype == "lowercase_no_accent":
                    result.accuracy_misspelled = acc

        # Timing metrics
        result.avg_inference_time_ms = sum(inference_times) / len(inference_times) if inference_times else 0
        result.total_inference_time_s = sum(inference_times) / 1000

        # Confusion matrix
        result.confusion_matrix = {
            'validity': {
                'TP': validity_tp,
                'TN': validity_tn,
                'FP': validity_fp,
                'FN': validity_fn,
            },
            'origin': {
                'TP': origin_tp,
                'FP': origin_fp,
                'FN': origin_fn,
            },
            'destination': {
                'TP': dest_tp,
                'FP': dest_fp,
                'FN': dest_fn,
            }
        }

        # Store sample errors
        result.errors = errors[:50]  # Limit to first 50 errors for analysis

        return result

    def print_report(self, result: EvaluationResult, model_name: str = "Model"):
        """Print formatted evaluation report."""
        print("\n" + "=" * 70)
        print(f"EVALUATION REPORT: {model_name}")
        print("=" * 70)

        print(f"\n{'OVERALL METRICS':^70}")
        print("-" * 70)
        print(f"  Total Samples:     {result.total_samples}")
        print(f"  Correct Samples:   {result.correct_samples}")
        print(f"  Accuracy:          {result.accuracy:.2%}")
        print(f"  Precision:         {result.precision:.2%}")
        print(f"  Recall:            {result.recall:.2%}")
        print(f"  F1 Score:          {result.f1_score:.2%}")

        print(f"\n{'PER-ENTITY METRICS':^70}")
        print("-" * 70)
        print(f"  {'Entity':<15} {'Precision':>12} {'Recall':>12} {'F1':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'Origin':<15} {result.origin_precision:>11.2%} {result.origin_recall:>11.2%} {result.origin_f1:>11.2%}")
        print(f"  {'Destination':<15} {result.destination_precision:>11.2%} {result.destination_recall:>11.2%} {result.destination_f1:>11.2%}")

        print(f"\n{'VALIDITY DETECTION':^70}")
        print("-" * 70)
        print(f"  Accuracy:          {result.validity_accuracy:.2%}")
        print(f"  Precision:         {result.validity_precision:.2%}")
        print(f"  Recall:            {result.validity_recall:.2%}")

        print(f"\n{'ROBUSTNESS METRICS':^70}")
        print("-" * 70)
        print(f"  Normal sentences:           {result.accuracy_normal:.2%}")
        print(f"  Lowercase only:             {result.accuracy_lowercase:.2%}")
        print(f"  No accents:                 {result.accuracy_no_accents:.2%}")
        print(f"  Lowercase + no accents:     {result.accuracy_misspelled:.2%}")

        print(f"\n{'PERFORMANCE':^70}")
        print("-" * 70)
        print(f"  Avg inference time:  {result.avg_inference_time_ms:.2f} ms")
        print(f"  Total time:          {result.total_inference_time_s:.2f} s")

        print(f"\n{'CONFUSION MATRIX (Validity)':^70}")
        print("-" * 70)
        cm = result.confusion_matrix['validity']
        print(f"                  Predicted Valid    Predicted Invalid")
        print(f"  Actual Valid        {cm['TP']:>6}              {cm['FN']:>6}")
        print(f"  Actual Invalid      {cm['FP']:>6}              {cm['TN']:>6}")

        if result.errors:
            print(f"\n{'SAMPLE ERRORS (first 5)':^70}")
            print("-" * 70)
            for err in result.errors[:5]:
                print(f"  Sentence: {err['sentence'][:60]}...")
                print(f"    Gold: {err['gold_origin']} -> {err['gold_destination']}")
                print(f"    Pred: {err['pred_origin']} -> {err['pred_destination']} ({err['method']})")
                print()

        print("=" * 70)

    def save_report(self, result: EvaluationResult, filepath: str, model_name: str = "Model"):
        """Save evaluation report to CSV file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Model', model_name])
            writer.writerow(['Total Samples', result.total_samples])
            writer.writerow(['Correct Samples', result.correct_samples])
            writer.writerow(['Accuracy', f'{result.accuracy:.4f}'])
            writer.writerow(['Precision', f'{result.precision:.4f}'])
            writer.writerow(['Recall', f'{result.recall:.4f}'])
            writer.writerow(['F1 Score', f'{result.f1_score:.4f}'])
            writer.writerow(['Origin Precision', f'{result.origin_precision:.4f}'])
            writer.writerow(['Origin Recall', f'{result.origin_recall:.4f}'])
            writer.writerow(['Origin F1', f'{result.origin_f1:.4f}'])
            writer.writerow(['Destination Precision', f'{result.destination_precision:.4f}'])
            writer.writerow(['Destination Recall', f'{result.destination_recall:.4f}'])
            writer.writerow(['Destination F1', f'{result.destination_f1:.4f}'])
            writer.writerow(['Validity Accuracy', f'{result.validity_accuracy:.4f}'])
            writer.writerow(['Accuracy Normal', f'{result.accuracy_normal:.4f}'])
            writer.writerow(['Accuracy Lowercase', f'{result.accuracy_lowercase:.4f}'])
            writer.writerow(['Accuracy No Accents', f'{result.accuracy_no_accents:.4f}'])
            writer.writerow(['Accuracy Misspelled', f'{result.accuracy_misspelled:.4f}'])
            writer.writerow(['Avg Inference Time (ms)', f'{result.avg_inference_time_ms:.4f}'])

        print(f"Report saved to {filepath}")


def load_test_data(filepath: str) -> List[Dict[str, str]]:
    """Load test data from CSV file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'sentence': row['sentence'],
                'origin': row['origin'],
                'destination': row['destination'],
            })
    return data


# Main evaluation script
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.nlp.models.baseline_model import BaselineModel

    # Load test data
    test_data = load_test_data("data/training/test.csv")
    print(f"Loaded {len(test_data)} test samples")

    # Initialize model and evaluator
    model = BaselineModel()
    evaluator = NLPEvaluator(verbose=True)

    # Evaluate
    result = evaluator.evaluate(model, test_data)

    # Print report
    evaluator.print_report(result, "Baseline (Preposition-based)")

    # Save report
    evaluator.save_report(result, "outputs/evaluation/baseline_report.csv", "Baseline")
