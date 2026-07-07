"""Inspect saved training artifacts (metrics, fairness slices, PR curve).

Run after ``make train``::

    make fairness
    make fairness MODEL=xgboost
    make pr-curve
    uv run python -m src.inspect --model random_forest
    uv run python -m src.inspect --pr-curve --model random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from . import data, preprocess
from .champion import DEFAULT_MODEL, load_metrics, load_pipeline, metrics_path, model_dir
from .train import split_train_val_test


def format_fairness_table(fairness: dict[str, dict]) -> str:
    """Pretty-print gender (or other) fairness slices."""
    lines = [
        f"{'Group':<8} {'n':>5} {'Recall':>8} {'Precision':>10} "
        f"{'Flag rate':>10} {'Churn rate':>11}",
    ]
    for group, stats in sorted(fairness.items()):
        lines.append(
            f"{group:<8} {stats['n']:>5} {stats['recall']:>8.1%} "
            f"{stats['precision']:>10.1%} {stats['predicted_positive_rate']:>10.1%} "
            f"{stats['actual_churn_rate']:>11.1%}"
        )
    return "\n".join(lines)


def print_fairness_report(model: str = DEFAULT_MODEL) -> None:
    """Load ``models/<model>/metrics.json`` and print fairness + test headline."""
    metrics = load_metrics(model)
    path = metrics_path(model)

    test = metrics.get("test", {})
    threshold = metrics.get("threshold", test.get("threshold"))

    print(f"Model:     {model}")
    print(f"Metrics:   {path}")
    if threshold is not None:
        print(f"Threshold: {threshold:.4f}")
    if test:
        print(
            "Test:      "
            f"recall={test.get('recall', 0):.1%}  "
            f"precision={test.get('precision', 0):.1%}  "
            f"f1={test.get('f1', 0):.3f}  "
            f"pr_auc={test.get('pr_auc', 0):.3f}"
        )
    print()

    fairness = metrics.get("fairness_by_gender")
    if not fairness:
        print("No fairness_by_gender block in metrics.json.")
        sys.exit(1)

    print("Fairness by gender (test set, at tuned threshold):")
    print(format_fairness_table(fairness))


def load_test_scores(
    model: str = DEFAULT_MODEL,
    *,
    feature_set: str = "baseline",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-score the held-out test split with the saved pipeline."""
    raw = data.load_customers()
    cleaned = preprocess.clean(raw)
    engineered = feature_set == "engineered"
    ds = preprocess.dataset_from_cleaned(cleaned, engineered=engineered)
    _, _, X_test, _, _, y_test, _, _, _ = split_train_val_test(
        ds.X, ds.y, ds.customer_id
    )
    pipe = load_pipeline(model)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()
    return y_true, y_proba, float(y_true.mean())


def pr_curve_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Precision-recall curve points and headline metrics."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = float(average_precision_score(y_true, y_proba))
    prevalence = float(y_true.mean())
    summary = {
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "lift_over_random": pr_auc / prevalence if prevalence else float("nan"),
        "precision": precision,
        "recall": recall,
    }
    if threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
        summary["operating_precision"] = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        summary["operating_recall"] = float(recall_score(y_true, y_pred))
    return summary


def save_pr_curve_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    *,
    model: str,
    threshold: float | None = None,
) -> dict[str, float]:
    """Plot PR curve with random baseline and optional operating point."""
    summary = pr_curve_summary(y_true, y_proba, threshold=threshold)
    precision = summary["precision"]
    recall = summary["recall"]
    pr_auc = summary["pr_auc"]
    prevalence = summary["prevalence"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"{model} (AP = {pr_auc:.3f})", linewidth=2)
    ax.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"Random baseline (prevalence = {prevalence:.1%})",
    )
    if threshold is not None:
        ax.scatter(
            [summary["operating_recall"]],
            [summary["operating_precision"]],
            color="crimson",
            s=60,
            zorder=5,
            label=f"Operating point (thr = {threshold:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall curve — {model} (test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return summary


def plot_pr_curve(
    model: str = DEFAULT_MODEL,
    *,
    out_path: Path | None = None,
    feature_set: str = "baseline",
) -> Path:
    """Re-score test data and save a PR curve PNG next to model artifacts."""
    metrics = load_metrics(model)
    threshold = metrics.get("threshold", metrics.get("test", {}).get("threshold"))
    y_true, y_proba, prevalence = load_test_scores(model, feature_set=feature_set)
    out = out_path or (model_dir(model) / "pr_curve.png")

    summary = save_pr_curve_plot(
        y_true,
        y_proba,
        out,
        model=model,
        threshold=float(threshold) if threshold is not None else None,
    )

    test = metrics.get("test", {})
    print(f"Model:              {model}")
    print(f"Test prevalence:    {prevalence:.1%} churn")
    print(f"PR-AUC (test):      {summary['pr_auc']:.3f}  (saved metrics: {test.get('pr_auc', 0):.3f})")
    print(f"Lift over random:   {summary['lift_over_random']:.2f}x")
    if threshold is not None:
        print(
            "Operating point:  "
            f"precision={summary['operating_precision']:.1%}  "
            f"recall={summary['operating_recall']:.1%}  "
            f"(threshold={float(threshold):.3f})"
        )
    print(f"Saved PR curve ->   {out}")

    sidecar = out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "model": model,
                "split": "test",
                "pr_auc": summary["pr_auc"],
                "prevalence": prevalence,
                "lift_over_random": summary["lift_over_random"],
                "threshold": threshold,
                "operating_precision": summary.get("operating_precision"),
                "operating_recall": summary.get("operating_recall"),
            },
            indent=2,
        )
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect saved metrics: fairness slices or PR curve."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model subdirectory under models/ (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pr-curve",
        action="store_true",
        help="plot precision-recall curve on the held-out test set",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG path (default: models/<model>/pr_curve.png)",
    )
    parser.add_argument(
        "--feature-set",
        choices=preprocess.FEATURE_SETS,
        default="baseline",
        help="must match the feature set used when training the model",
    )
    args = parser.parse_args(argv)
    if args.pr_curve:
        plot_pr_curve(args.model, out_path=args.out, feature_set=args.feature_set)
    else:
        print_fairness_report(args.model)


if __name__ == "__main__":
    main()
