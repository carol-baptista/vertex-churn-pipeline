"""Tests for src.champion."""

from __future__ import annotations

import json

import pytest

from src.champion import (
    CHAMPION_MODEL,
    baseline_feature_columns,
    build_manifest,
    load_threshold,
    score_rows,
    threshold_from_metrics,
    validate_feature_frame,
)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_champion_model_is_random_forest():
    assert CHAMPION_MODEL == "random_forest"


def test_baseline_feature_columns_count():
    cols = baseline_feature_columns()
    assert len(cols) == 18
    assert "tenure" in cols
    assert "customerID" not in cols


def test_threshold_from_metrics():
    assert threshold_from_metrics({"threshold": 0.441}) == 0.441
    assert threshold_from_metrics({"test": {"threshold": 0.5}}) == 0.5


def test_load_threshold_from_tmp(tmp_path, monkeypatch):
    model_dir = tmp_path / "random_forest"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text(json.dumps({"threshold": 0.441}))

    import src.champion as champion_module

    monkeypatch.setattr(champion_module, "MODELS_DIR", tmp_path)
    assert load_threshold() == 0.441


def test_score_rows_applies_threshold():
    pipe = Pipeline([("prep", StandardScaler()), ("model", LogisticRegression())])
    X = pd.DataFrame({"a": [0.0, 10.0]})
    y = pd.Series([0, 1])
    pipe.fit(X, y)

    rows = score_rows(pipe, X, threshold=0.5)
    assert len(rows) == 2
    assert rows[0]["churn_flag"] == 0
    assert rows[1]["churn_flag"] == 1


def test_build_manifest_includes_release_and_metrics(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "random_forest"
    model_dir.mkdir(parents=True)
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": 0.441,
                "select_metric": "f1",
                "cv_score": 0.63,
                "validation": {"f1": 0.635, "recall": 0.73, "precision": 0.56},
                "test": {
                    "recall": 0.711,
                    "precision": 0.55,
                    "f1": 0.62,
                    "pr_auc": 0.683,
                    "roc_auc": 0.842,
                },
            }
        )
    )

    import src.champion as champion_module

    monkeypatch.setattr(champion_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(champion_module, "SUMMARY_PATH", tmp_path / "missing-summary.json")

    manifest = build_manifest()
    assert manifest["release_id"] == "v1"
    assert manifest["model_id"] == "churn-rf"
    assert manifest["model"] == "random_forest"
    assert manifest["headline_metrics"]["f1"] == 0.62
    assert manifest["provenance"]["packaged_at"]


def test_validate_feature_frame_rejects_missing():
    with pytest.raises(ValueError, match="Missing feature columns"):
        validate_feature_frame(pd.DataFrame({"a": [1]}), columns=["a", "b"])
