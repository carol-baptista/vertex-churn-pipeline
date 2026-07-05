"""Tests for src.predict (no BigQuery required)."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.champion import load_threshold, score_rows
from tests.test_preprocess import _row


def test_score_rows_applies_threshold():
    pipe = Pipeline(
        [
            ("prep", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
    X = pd.DataFrame({"a": [0.0, 10.0]})
    y = pd.Series([0, 1])
    pipe.fit(X, y)

    rows = score_rows(pipe, X, threshold=0.5)
    assert len(rows) == 2
    assert rows[0]["churn_flag"] == 0
    assert rows[1]["churn_flag"] == 1


def test_load_threshold_reads_metrics(tmp_path, monkeypatch):
    model_dir = tmp_path / "random_forest"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text(json.dumps({"threshold": 0.441}))

    import src.champion as champion_module

    monkeypatch.setattr(champion_module, "MODELS_DIR", tmp_path)
    assert load_threshold() == 0.441


def test_make_dataset_row_shape_for_baseline():
    from src.preprocess import make_dataset

    ds = make_dataset(pd.DataFrame([_row(tenure=12, total_charges=600)]), engineered=False)
    assert ds.X.shape[0] == 1
    assert "customerID" not in ds.X.columns
