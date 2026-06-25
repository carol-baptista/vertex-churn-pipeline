"""Load the churn dataset from BigQuery into a pandas DataFrame.

Authentication uses Application Default Credentials (ADC). You set these up
in Phase 0 with:

    gcloud auth application-default login

No service-account key file is needed for local development.
"""

from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from . import config


def get_client() -> bigquery.Client:
    """Return a BigQuery client bound to the configured project."""
    return bigquery.Client(project=config.PROJECT_ID)


def load_customers(limit: int | None = None) -> pd.DataFrame:
    """Read churn_ml.customers into a DataFrame.

    Args:
        limit: optional row cap, useful for quick checks in the notebook.

    Returns:
        The full customers table as a pandas DataFrame.
    """
    query = f"SELECT * FROM `{config.TABLE_ID}`"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    client = get_client()
    # to_dataframe() requires the db-dtypes package (see requirements.txt).
    return client.query(query).to_dataframe()


if __name__ == "__main__":
    df = load_customers(limit=5)
    print(config.summary())
    print(df.head())
