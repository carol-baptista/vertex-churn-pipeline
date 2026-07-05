"""Batch scoring: seed a live-like BQ population and write predictions back to BigQuery.

Typical flow (local scoring, free aside from BQ storage)::

    make seed-scoring          # customers_scoring from a random sample (no Churn label)
    make score-local           # champion model -> churn_ml.predictions

Production-style flow (Vertex BatchPredictionJob)::

    make score-vertex          # export GCS -> batch job -> load predictions table

Run after ``make train`` and ``make deploy REGISTER_ONLY=1`` (for score-vertex).
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from google.cloud import aiplatform, bigquery, storage

from . import config, data
from .champion import (
    CHAMPION_MODEL,
    MODEL_ID,
    SERVING_VERSION,
    baseline_feature_columns,
    load_pipeline,
    load_threshold,
    score_rows,
)
from .preprocess import ID_COL, TARGET, clean, feature_columns

MODEL_DISPLAY_NAME = os.getenv("VERTEX_MODEL_DISPLAY_NAME", "churn-predictor")
BATCH_GCS_PREFIX = os.getenv("BATCH_GCS_PREFIX", "batch/churn-rf")


def _require_bucket() -> str:
    bucket = config.GCS_BUCKET or os.getenv("GCS_BUCKET", "")
    if not bucket:
        raise ValueError("GCS_BUCKET is not set in .env")
    return bucket


def predictions_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("customerID", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("churn_probability", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("churn_flag", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("threshold", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("source_table", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("scored_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def ensure_predictions_table(client: bigquery.Client | None = None) -> None:
    client = client or data.get_client()
    table = bigquery.Table(config.PREDICTIONS_TABLE_ID, schema=predictions_schema())
    table.time_partitioning = bigquery.TimePartitioning(field="scored_at")
    client.create_table(table, exists_ok=True)


def seed_scoring_table(*, limit: int = 500, keep_churn: bool = False) -> None:
    """Create ``customers_scoring``: random sample without labels (live-like)."""
    client = data.get_client()
    except_clause = "" if keep_churn else f" EXCEPT({TARGET})"
    query = f"""
        CREATE OR REPLACE TABLE `{config.SCORING_TABLE_ID}` AS
        SELECT
            *{except_clause},
            CURRENT_DATE() AS as_of_date
        FROM `{config.TABLE_ID}`
        ORDER BY RAND()
        LIMIT {int(limit)}
    """
    client.query(query).result()
    count = client.query(
        f"SELECT COUNT(*) AS n FROM `{config.SCORING_TABLE_ID}`"
    ).to_dataframe()["n"].iloc[0]
    print(f"Seeded scoring table -> {config.SCORING_TABLE_ID} ({count} rows)")
    if not keep_churn:
        print("  (Churn label dropped — simulates production scoring population)")


def load_scoring_frame(*, limit: int | None = None) -> pd.DataFrame:
    query = f"SELECT * FROM `{config.SCORING_TABLE_ID}`"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return data.get_client().query(query).to_dataframe()


def scoring_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw scoring rows and return baseline feature matrix + customer IDs."""
    cleaned = clean(raw)
    numeric, categorical = feature_columns(engineered=False)
    X = cleaned[numeric + categorical]
    return X, cleaned[ID_COL]


def score_frame(
    raw: pd.DataFrame,
    *,
    model: str = CHAMPION_MODEL,
) -> pd.DataFrame:
    """Score cleaned feature rows with the saved champion pipeline."""
    X, customer_ids = scoring_features(raw)
    validate_cols = baseline_feature_columns()
    missing = set(validate_cols) - set(X.columns)
    if missing:
        raise ValueError(f"Scoring frame missing feature columns: {sorted(missing)}")

    pipe = load_pipeline(model)
    threshold = load_threshold(model)
    rows = score_rows(pipe, X[validate_cols], threshold)
    return pd.DataFrame(
        {
            "customerID": customer_ids.astype(str).values,
            "churn_probability": [r["churn_probability"] for r in rows],
            "churn_flag": [r["churn_flag"] for r in rows],
            "threshold": threshold,
            "model": model,
            "model_version": f"{MODEL_ID}/{SERVING_VERSION}",
        }
    )


def write_predictions(
    scored: pd.DataFrame,
    *,
    run_id: str,
    source_table: str,
    scored_at: datetime | None = None,
) -> None:
    """Append one scoring run to ``churn_ml.predictions``."""
    client = data.get_client()
    ensure_predictions_table(client)

    out = scored.copy()
    out["run_id"] = run_id
    out["source_table"] = source_table
    out["scored_at"] = scored_at or datetime.now(UTC)

    columns = [field.name for field in predictions_schema()]
    job = client.load_table_from_dataframe(
        out[columns],
        config.PREDICTIONS_TABLE_ID,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ),
    )
    job.result()
    print(f"Wrote {len(out)} rows -> {config.PREDICTIONS_TABLE_ID} (run_id={run_id})")


def run_local_score(*, limit: int | None = None, model: str = CHAMPION_MODEL) -> str:
    """Score ``customers_scoring`` with the local artifact and append to predictions."""
    raw = load_scoring_frame(limit=limit)
    if raw.empty:
        raise ValueError(
            f"{config.SCORING_TABLE_ID} is empty. Run `make seed-scoring` first."
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    scored = score_frame(raw, model=model)
    write_predictions(scored, run_id=run_id, source_table=config.SCORING_TABLE_ID)

    flagged = int(scored["churn_flag"].sum())
    print(
        f"  scored={len(scored)}  flagged={flagged}  "
        f"flag_rate={flagged / len(scored):.1%}  threshold={scored['threshold'].iloc[0]:.4f}"
    )
    return run_id


def export_batch_jsonl(
    raw: pd.DataFrame,
    *,
    gcs_uri: str,
) -> int:
    """Write Vertex batch input JSONL (one request per line, with customerID passthrough)."""
    X, customer_ids = scoring_features(raw)
    feature_cols = baseline_feature_columns()

    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    blob_path = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])

    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    lines: list[str] = []
    for i in range(len(X)):
        instance = X.iloc[i][feature_cols].to_dict()
        instance[ID_COL] = str(customer_ids.iloc[i])
        lines.append(json.dumps({"instances": [instance]}))

    blob.upload_from_string("\n".join(lines) + "\n", content_type="application/jsonl")
    return len(lines)


def parse_batch_output_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        preds = payload.get("predictions")
        if isinstance(preds, list):
            rows.extend(preds)
        else:
            rows.append(payload)
    return rows


def load_batch_output(*, gcs_prefix: str) -> pd.DataFrame:
    """Read Vertex batch prediction JSONL output under a GCS prefix."""
    bucket_name = gcs_prefix.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_prefix.replace("gs://", "").split("/")[1:])
    if not prefix.endswith("/"):
        prefix += "/"

    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(bucket_name)

    rows: list[dict[str, Any]] = []
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        rows.extend(parse_batch_output_jsonl(blob.download_as_text()))

    if not rows:
        raise FileNotFoundError(f"No batch output JSONL under {gcs_prefix}")

    frame = pd.DataFrame(rows)
    if "customerID" not in frame.columns:
        raise ValueError("Batch output missing customerID — redeploy predictor with passthrough")
    return frame


def get_registered_model() -> aiplatform.Model:
    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        raise RuntimeError(
            f"No model named {MODEL_DISPLAY_NAME!r} in {config.REGION}. "
            "Run `make deploy REGISTER_ONLY=1` first."
        )
    return models[0]


def run_vertex_batch(*, limit: int | None = None, model: str = CHAMPION_MODEL) -> str:
    """Export scoring rows, run Vertex BatchPredictionJob, load results to BigQuery."""
    bucket = _require_bucket()
    raw = load_scoring_frame(limit=limit)
    if raw.empty:
        raise ValueError(
            f"{config.SCORING_TABLE_ID} is empty. Run `make seed-scoring` first."
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    input_uri = f"gs://{bucket}/{BATCH_GCS_PREFIX}/{run_id}/input/instances.jsonl"
    output_prefix = f"gs://{bucket}/{BATCH_GCS_PREFIX}/{run_id}/output/"

    n = export_batch_jsonl(raw, gcs_uri=input_uri)
    print(f"Exported {n} instances -> {input_uri}")

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
    registered = get_registered_model()
    print(f"Using registered model -> {registered.resource_name}")

    batch_job = registered.batch_predict(
        job_display_name=f"churn-batch-{run_id}",
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=input_uri,
        gcs_destination_prefix=output_prefix,
        machine_type=os.getenv("VERTEX_BATCH_MACHINE_TYPE", "n1-standard-2"),
        sync=True,
    )
    print(f"Batch job complete -> {batch_job.resource_name}")

    output = load_batch_output(gcs_prefix=output_prefix)
    scored = pd.DataFrame(
        {
            "customerID": output["customerID"].astype(str),
            "churn_probability": output["churn_probability"].astype(float),
            "churn_flag": output["churn_flag"].astype(int),
            "threshold": output["threshold"].astype(float),
            "model": model,
            "model_version": f"{MODEL_ID}/{SERVING_VERSION}",
        }
    )
    write_predictions(scored, run_id=run_id, source_table=config.SCORING_TABLE_ID)

    flagged = int(scored["churn_flag"].sum())
    print(
        f"  scored={len(scored)}  flagged={flagged}  "
        f"flag_rate={flagged / len(scored):.1%}"
    )
    return run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Seed scoring data and write batch predictions to BigQuery."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="create customers_scoring from a random sample")
    seed.add_argument("--limit", type=int, default=500, help="rows to sample (default: 500)")
    seed.add_argument(
        "--keep-churn",
        action="store_true",
        help="keep Churn label in scoring table (for backtesting only)",
    )

    local = sub.add_parser("score-local", help="score with local model.joblib -> BQ")
    local.add_argument("--limit", type=int, default=None, help="optional row cap")
    local.add_argument("--model", default=CHAMPION_MODEL)

    vertex = sub.add_parser("score-vertex", help="score via Vertex BatchPredictionJob -> BQ")
    vertex.add_argument("--limit", type=int, default=None, help="optional row cap")
    vertex.add_argument("--model", default=CHAMPION_MODEL)

    args = parser.parse_args(argv)

    if args.command == "seed":
        seed_scoring_table(limit=args.limit, keep_churn=args.keep_churn)
    elif args.command == "score-local":
        run_local_score(limit=args.limit, model=args.model)
    elif args.command == "score-vertex":
        run_vertex_batch(limit=args.limit, model=args.model)


if __name__ == "__main__":
    main()
