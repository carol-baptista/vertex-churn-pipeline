"""Score saved model artifacts on rows from BigQuery.

Run after ``make train``::

    make predict
    make predict CUSTOMER_ID=7590-VHVEG
    uv run python -m src.predict --model random_forest --row 0
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import pandas as pd
from google.cloud import bigquery
from sklearn.pipeline import Pipeline

from . import config, data
from .inspect import DEFAULT_MODEL, MODELS_DIR, load_metrics
from .preprocess import ID_COL, TARGET, make_dataset


def load_pipeline(model: str) -> Pipeline:
    path = MODELS_DIR / model / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make train` first (or check --model=...)."
        )
    pipe = joblib.load(path)
    if not hasattr(pipe, "predict_proba"):
        raise TypeError(f"Expected a sklearn Pipeline, got {type(pipe)!r}")
    return pipe


def threshold_for_model(model: str) -> float:
    metrics = load_metrics(model)
    threshold = metrics.get("threshold")
    if threshold is None:
        threshold = metrics.get("test", {}).get("threshold")
    if threshold is None:
        raise KeyError(f"No threshold in {metrics_path(model)}")
    return float(threshold)


def metrics_path(model: str):
    return MODELS_DIR / model / "metrics.json"


def score_features(pipe: Pipeline, X: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Return probability + binary flag for one or more feature rows."""
    proba = pipe.predict_proba(X)[:, 1]
    flags = (proba >= threshold).astype(int)
    return {
        "churn_probability": proba.tolist(),
        "churn_flag": flags.tolist(),
        "threshold": threshold,
    }


def load_customer_row(*, customer_id: str | None, row: int) -> pd.DataFrame:
    """Fetch one raw customer row from BigQuery."""
    if customer_id:
        query = f"""
            SELECT * FROM `{config.TABLE_ID}`
            WHERE {ID_COL} = @customer_id
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)
            ]
        )
        df = data.get_client().query(query, job_config=job_config).to_dataframe()
        if df.empty:
            raise ValueError(f"No row found for {ID_COL}={customer_id!r}")
        return df

    df = data.load_customers(limit=max(row + 1, 1))
    if row >= len(df):
        raise IndexError(f"--row {row} out of range (loaded {len(df)} row(s))")
    return df.iloc[[row]]


def predict_one(
    *,
    model: str = DEFAULT_MODEL,
    customer_id: str | None = None,
    row: int = 0,
    engineered: bool = False,
) -> dict[str, Any]:
    """Load artifact, fetch one BQ row, score at the saved validation threshold."""
    pipe = load_pipeline(model)
    threshold = threshold_for_model(model)
    raw = load_customer_row(customer_id=customer_id, row=row)
    ds = make_dataset(raw, engineered=engineered)

    scores = score_features(pipe, ds.X, threshold)
    actual = ds.y.iloc[0]
    return {
        "model": model,
        "customer_id": ds.customer_id.iloc[0],
        "actual_churn": int(actual),
        "churn_probability": scores["churn_probability"][0],
        "churn_flag": scores["churn_flag"][0],
        "threshold": threshold,
    }


def print_prediction(result: dict[str, Any]) -> None:
    print(f"Model:              {result['model']}")
    print(f"Customer:           {result['customer_id']}")
    print(f"Actual churn (BQ):  {result['actual_churn']}")
    print(f"Threshold:          {result['threshold']:.4f}")
    print(f"Churn probability:  {result['churn_probability']:.4f}")
    print(f"Churn flag:         {result['churn_flag']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score one BigQuery customer with a saved model.joblib artifact."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model subdirectory under models/ (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--customer-id",
        default=None,
        help=f"score this {ID_COL} from BigQuery (overrides --row)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="when no --customer-id, score the Nth row from a small BQ sample (default: 0)",
    )
    parser.add_argument(
        "--feature-set",
        choices=("baseline", "engineered"),
        default="baseline",
        help="must match the feature set used when training the artifact (default: baseline)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print result as JSON instead of a human-readable table",
    )
    args = parser.parse_args(argv)

    result = predict_one(
        model=args.model,
        customer_id=args.customer_id,
        row=args.row,
        engineered=args.feature_set == "engineered",
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_prediction(result)


if __name__ == "__main__":
    main()
