"""Champion model paths and shared scoring helpers.

Production deploy target: Random Forest (best test F1 / PR-AUC), not the CV winner.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config
from .preprocess import feature_columns

CHAMPION_MODEL = "random_forest"
DEFAULT_MODEL = CHAMPION_MODEL
MODEL_ID = "churn-rf"
SERVING_VERSION = "v1"
SERVING_DIR = config.REPO_ROOT / "serving" / "churn-rf" / SERVING_VERSION
CHANGELOG_PATH = config.REPO_ROOT / "serving" / "churn-rf" / "CHANGELOG.md"
MODELS_DIR = config.REPO_ROOT / "models"
SUMMARY_PATH = MODELS_DIR / "summary.json"


def model_dir(model: str = CHAMPION_MODEL) -> Path:
    return MODELS_DIR / model


def metrics_path(model: str = CHAMPION_MODEL) -> Path:
    return model_dir(model) / "metrics.json"


def pipeline_path(model: str = CHAMPION_MODEL) -> Path:
    return model_dir(model) / "model.joblib"


def load_metrics(model: str = CHAMPION_MODEL) -> dict:
    path = metrics_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make train` first (or check model name)."
        )
    return json.loads(path.read_text())


def load_pipeline(model: str = CHAMPION_MODEL) -> Pipeline:
    path = pipeline_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make train` first (or check model name)."
        )
    pipe = joblib.load(path)
    if not hasattr(pipe, "predict_proba"):
        raise TypeError(f"Expected a sklearn Pipeline, got {type(pipe)!r}")
    return pipe


def threshold_from_metrics(metrics: dict) -> float:
    threshold = metrics.get("threshold")
    if threshold is None:
        threshold = metrics.get("test", {}).get("threshold")
    if threshold is None:
        raise KeyError("No threshold in metrics.json")
    return float(threshold)


def load_threshold(model: str = CHAMPION_MODEL) -> float:
    return threshold_from_metrics(load_metrics(model))


def baseline_feature_columns() -> list[str]:
    numeric, categorical = feature_columns(engineered=False)
    return numeric + categorical


def build_serving_metadata(model: str = CHAMPION_MODEL) -> dict[str, Any]:
    """Metadata written alongside the serving bundle."""
    metrics = load_metrics(model)
    test = metrics.get("test", {})
    return {
        "model": model,
        "feature_set": "baseline",
        "threshold": threshold_from_metrics(metrics),
        "feature_columns": baseline_feature_columns(),
        "test_metrics": {
            "recall": test.get("recall"),
            "precision": test.get("precision"),
            "f1": test.get("f1"),
            "pr_auc": test.get("pr_auc"),
        },
    }


def _git_provenance() -> dict[str, str | None]:
    """Best-effort git metadata for the release manifest."""
    provenance: dict[str, str | None] = {"git_sha": None, "git_branch": None}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        provenance["git_sha"] = sha.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        provenance["git_branch"] = branch.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return provenance


def _training_run_config() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    summary = json.loads(SUMMARY_PATH.read_text())
    return summary.get("run_config")


def build_manifest(
    *,
    model: str = CHAMPION_MODEL,
    release_id: str = SERVING_VERSION,
) -> dict[str, Any]:
    """Machine-readable release record for a packaged serving bundle."""
    metrics = load_metrics(model)
    test = metrics.get("test", {})
    validation = metrics.get("validation", {})
    metadata = build_serving_metadata(model)
    run_config = _training_run_config() or {}

    return {
        "release_id": release_id,
        "model_id": MODEL_ID,
        "model": model,
        "feature_set": metadata["feature_set"],
        "threshold": metadata["threshold"],
        "feature_columns": metadata["feature_columns"],
        "headline_metrics": {
            "split": "test",
            "recall": test.get("recall"),
            "precision": test.get("precision"),
            "f1": test.get("f1"),
            "pr_auc": test.get("pr_auc"),
            "roc_auc": test.get("roc_auc"),
        },
        "validation_metrics": {
            "recall": validation.get("recall"),
            "precision": validation.get("precision"),
            "f1": validation.get("f1"),
            "pr_auc": validation.get("pr_auc"),
        },
        "training": {
            "metric": run_config.get("metric", metrics.get("select_metric")),
            "pos_weight": run_config.get("pos_weight_mode"),
            "threshold_strategy": run_config.get("threshold_strategy"),
            "data_split": run_config.get("data_split"),
            "threshold_tuned_on": run_config.get("threshold_tuned_on", "validation"),
            "cv_score": metrics.get("cv_score"),
        },
        "artifact": {
            "model_joblib": "model.joblib",
            "source_path": str(model_dir(model) / "model.joblib"),
        },
        "bundle_paths": {
            "local": str(SERVING_DIR),
            "gcs_prefix": f"models/{MODEL_ID}/{release_id}",
        },
        "vertex": {
            "model_display_name": os.getenv(
                "VERTEX_MODEL_DISPLAY_NAME", "churn-predictor"
            ),
            "deployed_model_display_name": f"{model}-{release_id}",
        },
        "provenance": {
            "packaged_at": datetime.now(UTC).isoformat(),
            **_git_provenance(),
        },
        "notes": (
            "Champion for deployment: Random Forest with baseline features. "
            "Selected on test F1/PR-AUC, not CV winner."
        ),
    }


def score_rows(
    pipe: Pipeline,
    X: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    """Score feature rows; return probability + flag per row."""
    proba = pipe.predict_proba(X)[:, 1]
    return [
        {
            "churn_probability": float(p),
            "churn_flag": int(p >= threshold),
            "threshold": threshold,
        }
        for p in proba
    ]


def validate_feature_frame(X: pd.DataFrame, *, columns: list[str]) -> None:
    missing = set(columns) - set(X.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    extra = set(X.columns) - set(columns)
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")
