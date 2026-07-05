"""Tests for serving bundle predictor parity."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.champion import score_rows
from src.package import load_predictor_from_dir, package_champion, predict_with_bundle
from tests.test_preprocess import _row


@pytest.fixture
def bundled_champion(tmp_path, monkeypatch):
    import src.champion as champion_module
    import src.package as package_module

    model_dir = tmp_path / "models" / "random_forest"
    model_dir.mkdir(parents=True)
    out_dir = tmp_path / "bundle"
    template_predictor = Path(__file__).resolve().parents[1] / "serving" / "churn-rf" / "v1" / "predictor.py"
    template_requirements = Path(__file__).resolve().parents[1] / "serving" / "churn-rf" / "v1" / "requirements.txt"

    raw = pd.DataFrame(
        [
            _row(tenure=12, total_charges=600, Churn=False, customer_id="cust-1"),
            _row(tenure=2, total_charges=100, Churn=True, customer_id="cust-2"),
        ]
    )
    from src.preprocess import build_preprocessor, make_dataset

    ds = make_dataset(raw, engineered=False)
    prep = build_preprocessor(engineered=False, scale_numeric=False)
    pipe = Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=10, random_state=0))])
    pipe.fit(ds.X, ds.y)
    joblib.dump(pipe, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps({"threshold": 0.441}))

    monkeypatch.setattr(champion_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(champion_module, "SERVING_DIR", out_dir)
    monkeypatch.setattr(package_module, "PREDICTOR_SRC", template_predictor)
    monkeypatch.setattr(package_module, "REQUIREMENTS_SRC", template_requirements)
    package_champion(out_dir=out_dir)
    return out_dir, ds


def test_predictor_loads_from_bundle(bundled_champion):
    bundle_dir, ds = bundled_champion
    predictor = load_predictor_from_dir(bundle_dir)
    frame = predictor.preprocess({"instances": [ds.X.iloc[0].to_dict()]})
    proba = predictor.predict(frame)
    result = predictor.postprocess(proba)
    assert "predictions" in result
    assert "churn_probability" in result["predictions"][0]


def test_predictor_passes_through_customer_id(bundled_champion):
    bundle_dir, ds = bundled_champion
    instance = ds.X.iloc[0].to_dict()
    instance["customerID"] = "cust-pass-through"
    actual = predict_with_bundle(bundle_dir, [instance])[0]
    assert actual["customerID"] == "cust-pass-through"


def test_bundle_matches_score_rows(bundled_champion):
    bundle_dir, ds = bundled_champion
    instance = ds.X.iloc[0].to_dict()
    pipe = joblib.load(bundle_dir / "model.joblib")
    expected = score_rows(pipe, ds.X, 0.441)[0]
    actual = predict_with_bundle(bundle_dir, [instance])[0]
    assert abs(expected["churn_probability"] - actual["churn_probability"]) < 1e-6
    assert expected["churn_flag"] == actual["churn_flag"]