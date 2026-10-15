#!/usr/bin/env python
"""
Model Comparison Script

Evaluates every NLP model on:
- the test set (data/training/test.csv): all its cities are also in the
  training set, so it measures performance on known cities
- an "unseen cities" test set: the same sentences with cities replaced by SNCF
  stations absent from the training data, to measure generalization

Each NER model is evaluated raw and with the gazetteer post-processing.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --models baseline models/spacy_ner/ner_blank_1000
    python scripts/evaluate_models.py --no-unseen
"""

import argparse
import contextlib
import csv
import io
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from src.nlp.evaluator import NLPEvaluator, load_test_data
from src.nlp.gazetteer import CityGazetteer, find_entity_span, normalize

TEST_PATH = "data/training/test.csv"
UNSEEN_PATH = "data/training/test_unseen_cities.csv"
OUTPUT_PATH = "outputs/evaluation/model_comparison.json"


def build_unseen_test_set(gazetteer: CityGazetteer, output_path: str, seed: int = 0) -> int:
    """
    Build a test set with cities unknown to the gazetteer.

    Valid sentences of the test set get their origin/destination replaced by
    SNCF stations that are not in the training data.
    """
    from src.api.sncf_client import SNCFClient

    client = SNCFClient()
    with contextlib.redirect_stdout(io.StringIO()):
        client.load_data(use_api=True)
    known = set(gazetteer.keys)
    unseen = sorted({
        s.city for s in client.get_all_stations()
        if normalize(s.city) not in known and len(normalize(s.city)) >= 3
        # Skip names sharing a word with a known city ("Albi Madeleine" / "Albi")
        and not any(k in normalize(s.city).split() or normalize(s.city) in k.split()
                    for k in known if len(k) >= 4)
    })
    if len(unseen) < 2:
        raise RuntimeError("Not enough unseen stations (is the SNCF API reachable?)")

    rng = random.Random(seed)
    rows = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["origin"] == "INVALID":
                continue
            span_o = find_entity_span(row["sentence"], row["origin"])
            span_d = find_entity_span(row["sentence"], row["destination"],
                                      exclude=[span_o] if span_o else [])
            if not span_o or not span_d:
                continue
            origin, destination = rng.sample(unseen, 2)
            sentence = row["sentence"]
            lower = sentence == sentence.lower()
            for (start, end), city in sorted([(span_o, origin), (span_d, destination)],
                                             key=lambda x: -x[0][0]):
                sentence = sentence[:start] + (city.lower() if lower else city) + sentence[end:]
            rows.append({"sentence_id": row["sentence_id"], "sentence": sentence,
                         "origin": origin, "destination": destination})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence_id", "sentence", "origin", "destination"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def categorize_error(evaluator: NLPEvaluator, error: dict) -> str:
    """Classify a prediction error."""
    n = evaluator._normalize_city
    go, gd = error["gold_origin"], error["gold_destination"]
    po, pd = error["pred_origin"], error["pred_destination"]
    if go == "INVALID":
        return "invalid_accepted"
    if po is None and pd is None:
        return "valid_rejected"
    if po and pd and n(po) == n(gd) and n(pd) == n(go):
        return "swapped"
    if po is None or pd is None:
        return "one_entity_missing"
    return "wrong_span"


def model_factories(spec: str, gazetteer: CityGazetteer):
    """Return [(name, factory)] for a model spec ("baseline" or a model path)."""
    if spec == "baseline":
        from src.nlp.models.baseline_model import BaselineModel
        return [("baseline", lambda: BaselineModel(gazetteer=gazetteer))]

    path = Path(spec)
    if (path / "config.json").exists():
        from src.nlp.models.camembert_ner import CamembertNERModel as Model
    else:
        from src.nlp.models.spacy_ner import SpacyNERModel as Model
    return [
        (f"{path.name} (raw)", lambda: Model(spec, use_gazetteer=False)),
        (f"{path.name} (+gazetteer)", lambda: Model(spec, gazetteer=gazetteer)),
    ]


def default_models():
    """Baseline + every trained model found in models/."""
    specs = ["baseline"]
    for folder in ("models/spacy_ner", "models/camembert_ner"):
        if Path(folder).exists():
            specs += sorted(str(p) for p in Path(folder).iterdir()
                            if p.is_dir() and ((p / "meta.json").exists() or (p / "config.json").exists()))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Compare NLP models")
    parser.add_argument("--models", nargs="+", help="'baseline' and/or model paths (default: all)")
    parser.add_argument("--no-unseen", action="store_true", help="Skip the unseen cities test set")
    parser.add_argument("--output", default=OUTPUT_PATH, help="JSON results file")
    args = parser.parse_args()

    gazetteer = CityGazetteer()
    evaluator = NLPEvaluator(verbose=False)

    test_sets = {"known": TEST_PATH}
    if not args.no_unseen:
        try:
            n = build_unseen_test_set(gazetteer, UNSEEN_PATH)
            print(f"Unseen cities test set: {n} sentences -> {UNSEEN_PATH}")
            test_sets["unseen"] = UNSEEN_PATH
        except Exception as e:
            print(f"Skipping unseen cities test set: {e}")
    data = {name: load_test_data(path) for name, path in test_sets.items()}

    results = []
    for spec in args.models or default_models():
        for name, factory in model_factories(spec, gazetteer):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = factory()
            entry = {"model": name, "path": spec}
            for set_name, samples in data.items():
                res = evaluator.evaluate(model, samples)
                entry[set_name] = {
                    "accuracy": round(res.accuracy * 100, 2),
                    "f1": round(res.f1_score * 100, 2),
                    "validity_accuracy": round(res.validity_accuracy * 100, 2),
                    "avg_inference_ms": round(res.avg_inference_time_ms, 2),
                    "errors": dict(Counter(categorize_error(evaluator, e) for e in res.errors)),
                }
            results.append(entry)
            line = f"{name:<40}"
            for set_name in data:
                line += f" {set_name}: acc={entry[set_name]['accuracy']:6.2f}% f1={entry[set_name]['f1']:6.2f}%"
            print(line + f"  {entry['known']['avg_inference_ms']:.1f} ms", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
