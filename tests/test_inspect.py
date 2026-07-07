"""Tests for src.inspect."""

import numpy as np

from src.inspect import format_fairness_table, pr_curve_summary, save_pr_curve_plot


def test_format_fairness_table_columns():
    table = format_fairness_table(
        {
            "Female": {
                "n": 521,
                "recall": 0.696551724137931,
                "precision": 0.543010752688172,
                "predicted_positive_rate": 0.3570057581573896,
                "actual_churn_rate": 0.2783109404990403,
            },
            "Male": {
                "n": 536,
                "recall": 0.725925925925926,
                "precision": 0.5568181818181818,
                "predicted_positive_rate": 0.3283582089552239,
                "actual_churn_rate": 0.251865671641791,
            },
        }
    )
    assert "Group" in table
    assert "Female" in table
    assert "69.7%" in table
    assert "72.6%" in table


def test_pr_curve_summary_lift_over_random():
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=200, p=[0.73, 0.27])
    y_proba = rng.random(200)
    summary = pr_curve_summary(y_true, y_proba, threshold=0.5)
    assert 0 <= summary["pr_auc"] <= 1
    assert abs(summary["prevalence"] - 0.27) < 0.1
    assert summary["lift_over_random"] > 0


def test_save_pr_curve_plot_writes_file(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    out = tmp_path / "pr_curve.png"
    summary = save_pr_curve_plot(
        y_true, y_proba, out, model="random_forest", threshold=0.5
    )
    assert out.exists()
    assert summary["pr_auc"] > 0
