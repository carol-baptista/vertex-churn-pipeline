"""Tests for src.batch (no live GCP calls)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.batch import parse_batch_output_jsonl, score_frame, scoring_features
from tests.test_preprocess import _row


def test_scoring_features_without_churn():
    raw = pd.DataFrame([_row(tenure=12, total_charges=600, customer_id="cust-1")])
    raw = raw.drop(columns=["Churn"])
    X, ids = scoring_features(raw)
    assert len(X) == 1
    assert ids.iloc[0] == "cust-1"
    assert "Churn" not in X.columns


def test_score_frame_matches_threshold(tmp_path, monkeypatch):
    import src.champion as champion_module

    model_dir = tmp_path / "random_forest"
    model_dir.mkdir()
    raw = pd.DataFrame(
        [
            _row(tenure=12, total_charges=600, Churn=False, customer_id="low-risk"),
            _row(tenure=2, total_charges=100, Churn=True, customer_id="high-risk"),
        ]
    )
    raw = raw.drop(columns=["Churn"])

    from src.preprocess import build_preprocessor, make_dataset

    train_raw = pd.DataFrame(
        [
            _row(tenure=12, total_charges=600, Churn=False, customer_id="a"),
            _row(tenure=2, total_charges=100, Churn=True, customer_id="b"),
        ]
    )
    ds = make_dataset(train_raw, engineered=False)
    prep = build_preprocessor(engineered=False, scale_numeric=False)
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=200))])
    pipe.fit(ds.X, ds.y)
    (model_dir / "metrics.json").write_text(json.dumps({"threshold": 0.5}))

    import joblib

    joblib.dump(pipe, model_dir / "model.joblib")
    monkeypatch.setattr(champion_module, "MODELS_DIR", tmp_path)

    scored = score_frame(raw)
    assert list(scored["customerID"]) == ["low-risk", "high-risk"]
    assert set(scored["churn_flag"]) <= {0, 1}


def test_parse_batch_output_jsonl():
    text = '\n'.join(
        [
            json.dumps({"predictions": [{"customerID": "a", "churn_probability": 0.2}]}),
            json.dumps({"predictions": [{"customerID": "b", "churn_flag": 1}]}),
        ]
    )
    rows = parse_batch_output_jsonl(text)
    assert len(rows) == 2
    assert rows[0]["customerID"] == "a"
