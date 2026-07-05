"""Vertex CPR predictor for the churn Random Forest champion.

Self-contained: no imports from ``src/`` (runs inside the Vertex serving container).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor


class ChurnPredictor(Predictor):
    """Load RF pipeline + threshold; score raw baseline feature rows."""

    def load(self, artifacts_uri: str) -> None:
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._pipe = joblib.load("model.joblib")
        with Path("threshold.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        self._threshold = float(meta["threshold"])
        with Path("feature_schema.json").open(encoding="utf-8") as f:
            schema = json.load(f)
        self._feature_columns: list[str] = list(schema["feature_columns"])

    def preprocess(self, prediction_input: dict) -> pd.DataFrame:
        instances = prediction_input.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("prediction_input must contain a non-empty 'instances' list")
        frame = pd.DataFrame(instances)
        missing = set(self._feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        return frame[self._feature_columns]

    def predict(self, instances: pd.DataFrame) -> np.ndarray:
        return self._pipe.predict_proba(instances)[:, 1]

    def postprocess(self, prediction_results: np.ndarray) -> dict[str, Any]:
        predictions = []
        for proba in prediction_results:
            p = float(proba)
            predictions.append(
                {
                    "churn_probability": p,
                    "churn_flag": int(p >= self._threshold),
                    "threshold": self._threshold,
                }
            )
        return {"predictions": predictions}
