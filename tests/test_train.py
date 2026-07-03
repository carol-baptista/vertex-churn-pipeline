"""Unit tests for src.train probe audit (no BigQuery required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.preprocess import make_dataset
from src.train import (
    best_threshold,
    build_random_forest,
    probe_kept_features,
    resolve_sklearn_scorer,
    resolve_training_metrics,
    run_probe_audit,
    _with_feature_selector,
)
from tests.test_preprocess import _row


def _synthetic_dataset(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        tenure = int(rng.integers(1, 72))
        monthly = float(rng.uniform(20, 120))
        churn = tenure < 12 or monthly > 90
        rows.append(
            _row(
                tenure=tenure,
                total_charges=monthly * tenure,
                customer_id=f"cust-{i}",
                MonthlyCharges=monthly,
                Churn=churn,
            )
        )
    ds = make_dataset(pd.DataFrame(rows))
    return ds.X, ds.y


def test_run_probe_audit_reports_probe_columns_and_importances():
    X, y = _synthetic_dataset()

    _, audit = run_probe_audit(X, y)

    assert audit["method"] == "feature_engine.ProbeFeatureSelection"
    assert audit["n_probes"] == 3
    assert len(audit["probe_columns"]) == 3
    assert audit["n_features_in"] == len(audit["feature_importances"])
    assert audit["n_selected"] + audit["n_dropped"] == audit["n_features_in"]
    assert all("probe" in name for name in audit["probe_columns"])
    assert audit["probe_threshold_value"] > 0


def test_probe_kept_features_excludes_dropped():
    audit = {
        "features_to_drop": ["cat__Partner_False"],
        "feature_importances": [
            {"feature": "num__tenure", "importance": 0.1},
            {"feature": "cat__Partner_False", "importance": 0.01},
            {"feature": "cat__Contract_Month-to-month", "importance": 0.08},
        ],
    }
    kept = probe_kept_features(audit)
    assert kept == ["num__tenure", "cat__Contract_Month-to-month"]


def test_with_feature_selector_reduces_encoded_width():
    X, y = _synthetic_dataset(n=80)
    _, audit = run_probe_audit(X, y)
    kept = probe_kept_features(audit)
    pipe = _with_feature_selector(
        build_random_forest(pos_weight=1.0), kept, audit["encoded_feature_names"]
    )
    pipe.fit(X, y)
    transformed = pipe.named_steps["select"].transform(
        pipe.named_steps["prep"].transform(X)
    )
    assert transformed.shape[1] == len(kept)


def test_resolve_sklearn_scorer_aliases():
    assert resolve_sklearn_scorer("pr_auc") == "average_precision"
    assert resolve_sklearn_scorer("f2") == "f2"
    assert resolve_sklearn_scorer("mcc") == "matthews_corrcoef"


def test_resolve_training_metrics_defaults_to_single_metric():
    grid, select = resolve_training_metrics("f2", None, None)
    assert grid == "f2"
    assert select == "f2"


def test_best_threshold_recall_floor_targets_precision_at_min_recall():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.95, 0.85, 0.75, 0.55, 0.45, 0.35, 0.25, 0.15, 0.10, 0.05])

    threshold = best_threshold(
        y_true, y_proba, strategy="recall_floor", recall_floor=0.75
    )
    y_pred = (y_proba >= threshold).astype(int)

    assert recall_score(y_true, y_pred) >= 0.75
    assert precision_score(y_true, y_pred, zero_division=0) >= 0.5
