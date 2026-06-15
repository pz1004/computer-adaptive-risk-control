"""
Build a per-exit cache for a small real tabular cascade.

Confirmatory IEEE Access run:
  python -m adapters.tabular_cascade --dataset digits --out cache/digits_tabular_cascade.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from carc.chain import build_chain


def git_sha() -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return proc.stdout.strip()


def load_dataset(name: str):
    if name == "digits":
        return load_digits()
    if name == "wine":
        return load_wine()
    if name == "breast_cancer":
        return load_breast_cancer()
    raise ValueError(f"unknown dataset: {name}")


def make_exits(seed: int):
    return [
        (
            "gaussian_nb",
            GaussianNB(),
            1.0,
        ),
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, C=2.0, solver="lbfgs", random_state=seed),
            ),
            4.0,
        ),
        (
            "rbf_svc",
            make_pipeline(
                StandardScaler(),
                SVC(C=10.0, gamma="scale", probability=True, random_state=seed),
            ),
            12.0,
        ),
    ]


def build_cache(args) -> dict:
    dataset = load_dataset(args.dataset)
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target)
    X_train, X_cache, y_train, y_cache = train_test_split(
        X,
        y,
        test_size=args.cache_fraction,
        random_state=args.seed,
        stratify=y,
    )
    exits = make_exits(args.seed)
    scores = []
    correct = []
    predictions = []
    exit_names = []
    exit_costs = []
    exit_accuracies = []

    for name, model, cost in exits:
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_cache)
        pred = prob.argmax(axis=1)
        conf = np.minimum(prob.max(axis=1), np.nextafter(1.0, 0.0))
        corr = pred == y_cache
        exit_names.append(name)
        exit_costs.append(cost)
        exit_accuracies.append(float(corr.mean()))
        scores.append(conf)
        correct.append(corr.astype(float))
        predictions.append(pred)

    scores_arr = np.stack(scores, axis=1)
    correct_arr = np.stack(correct, axis=1)
    pred_arr = np.stack(predictions, axis=1)
    loss_arr = 1.0 - correct_arr
    exit_cost_arr = np.asarray(exit_costs, dtype=float)
    thresholds = np.linspace(0.0, 1.0, args.threshold_count)
    loss_matrix, cost_matrix = build_chain(scores_arr, loss_arr, exit_cost_arr, thresholds)
    meta = {
        "dataset": args.dataset,
        "dataset_source": "sklearn.datasets",
        "adapter": "adapters.tabular_cascade",
        "seed": int(args.seed),
        "git_sha": git_sha(),
        "train_size": int(X_train.shape[0]),
        "cache_size": int(X_cache.shape[0]),
        "num_features": int(X.shape[1]),
        "num_classes": int(len(np.unique(y))),
        "exit_names": exit_names,
        "exit_cost_kind": "relative_classifier_cost_proxy",
        "exit_costs": exit_cost_arr.tolist(),
        "exit_accuracies": exit_accuracies,
        "threshold_count": int(args.threshold_count),
        "chain": "scalar top-probability threshold shared across exits",
    }
    return {
        "scores": scores_arr,
        "predictions": pred_arr,
        "correct": correct_arr,
        "loss": loss_arr,
        "exit_cost": exit_cost_arr,
        "thresholds": thresholds,
        "loss_matrix": loss_matrix,
        "cost_matrix": cost_matrix,
        "meta_json": json.dumps(meta, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["digits", "wine", "breast_cancer"], default="digits")
    parser.add_argument("--out", default="cache/digits_tabular_cascade.npz")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--cache-fraction", type=float, default=0.5)
    parser.add_argument("--threshold-count", type=int, default=101)
    args = parser.parse_args()
    if not (0.1 <= args.cache_fraction <= 0.9):
        raise ValueError("--cache-fraction must be between 0.1 and 0.9")
    if args.threshold_count < 2:
        raise ValueError("--threshold-count must be at least 2")

    payload = build_cache(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    meta = json.loads(payload["meta_json"])
    print(f"wrote {out}")
    print(f"cache_size={meta['cache_size']} exit_accuracies={meta['exit_accuracies']}")
    print(f"exit_costs={meta['exit_costs']}")


if __name__ == "__main__":
    main()
