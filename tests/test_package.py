"""Tests for src.package."""

from __future__ import annotations

import json
import shutil

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.champion import baseline_feature_columns
from src.package import bundle_paths, package_champion


def _fake_pipeline():
    pipe = Pipeline([("prep", StandardScaler()), ("model", LogisticRegression())])
    import pandas as pd

    pipe.fit(pd.DataFrame({"a": [0.0, 1.0]}), [0, 1])
    return pipe


def test_package_champion_writes_files(tmp_path, monkeypatch):
    import src.champion as champion_module

    model_dir = tmp_path / "models" / "random_forest"
    model_dir.mkdir(parents=True)
    out_dir = tmp_path / "bundle"
    template_dir = tmp_path / "serving" / "churn-rf" / "v1"
    template_dir.mkdir(parents=True)
    (template_dir / "predictor.py").write_text("# stub predictor\n")
    (template_dir / "requirements.txt").write_text("scikit-learn\n")

    joblib.dump(_fake_pipeline(), model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": 0.441,
                "test": {"recall": 0.71, "precision": 0.55, "f1": 0.62, "pr_auc": 0.68},
            }
        )
    )

    monkeypatch.setattr(champion_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(champion_module, "SERVING_DIR", out_dir)
    import src.package as package_module

    monkeypatch.setattr(package_module, "PREDICTOR_SRC", template_dir / "predictor.py")
    monkeypatch.setattr(package_module, "REQUIREMENTS_SRC", template_dir / "requirements.txt")

    root = package_champion(out_dir=out_dir)
    paths = bundle_paths(out_dir)
    assert paths["model"].exists()
    assert paths["threshold"].exists()
    assert paths["manifest"].exists()
    assert paths["feature_schema"].exists()
    meta = json.loads(paths["threshold"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    assert meta["threshold"] == 0.441
    assert manifest["release_id"] == "v1"
    assert manifest["headline_metrics"]["f1"] == 0.62
    assert manifest["feature_columns"] == baseline_feature_columns()
    assert root == out_dir
