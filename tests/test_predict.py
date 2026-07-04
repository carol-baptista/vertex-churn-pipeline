"""Tests for src.predict (no BigQuery required)."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.predict import score_features, threshold_for_model
from tests.test_preprocess import _row


def test_score_features_applies_threshold():
    pipe = Pipeline(
        [
            ("prep", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
    X = pd.DataFrame({"a": [0.0, 10.0]})
    y = pd.Series([0, 1])
    pipe.fit(X, y)

    out = score_features(pipe, X, threshold=0.5)
    assert len(out["churn_probability"]) == 2
    assert out["churn_flag"] == [0, 1]
    assert out["threshold"] == 0.5


def test_threshold_for_model_reads_metrics(tmp_path, monkeypatch):
    model_dir = tmp_path / "random_forest"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text(json.dumps({"threshold": 0.441}))

    import src.inspect as inspect_module
    import src.predict as predict_module

    monkeypatch.setattr(inspect_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(predict_module, "MODELS_DIR", tmp_path)
    assert threshold_for_model("random_forest") == 0.441


def test_make_dataset_row_shape_for_baseline():
    from src.preprocess import make_dataset

    ds = make_dataset(pd.DataFrame([_row(tenure=12, total_charges=600)]), engineered=False)
    assert ds.X.shape[0] == 1
    assert "customerID" not in ds.X.columns
