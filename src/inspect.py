"""Inspect saved training artifacts (metrics, fairness slices).

Run after ``make train``::

    make fairness
    make fairness MODEL=xgboost
    uv run python -m src.inspect --model random_forest
"""

from __future__ import annotations

import argparse
import sys

from .champion import DEFAULT_MODEL, load_metrics, metrics_path


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print fairness slices from saved metrics.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model subdirectory under models/ (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)
    print_fairness_report(args.model)


if __name__ == "__main__":
    main()
