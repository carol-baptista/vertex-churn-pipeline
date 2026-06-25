"""Central config loaded from the repo's .env file.

Importing this module makes the GCP/BigQuery settings available to both the
EDA notebook and the training script, so they never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load .env from the repo root regardless of where the process is started.
load_dotenv(REPO_ROOT / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "churn-predictor-ml-2026")
REGION = os.getenv("GCP_REGION", "us-west1")
BQ_DATASET = os.getenv("BQ_DATASET", "churn_ml")
BQ_TABLE = os.getenv("BQ_TABLE", "customers")

# Fully-qualified table id, e.g. churn-predictor-ml-2026.churn_ml.customers
TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"


def summary() -> str:
    """Human-readable view of the resolved settings (handy in the notebook)."""
    return (
        f"project:  {PROJECT_ID}\n"
        f"region:   {REGION}\n"
        f"table:    {TABLE_ID}"
    )
