"""Preprocessing: clean the raw churn table and build the feature pipeline.

The decisions here come straight from the EDA "Takeaways" cell in
``notebooks/01_eda.ipynb``:

- ``TotalCharges`` is loaded as STRING; coerce to numeric. New customers
  (``tenure == 0``) have not been billed yet, so blanks become 0. Any other blank
  is treated as a data error: the row is dropped and logged.
- Drop ``customerID`` (pure identifier) and ``gender`` (protected attribute with
  ~0 correlation). ``gender`` is returned separately for a fairness audit and is
  never used as a model feature.
- Collapse ``"No internet service"`` / ``"No phone service"`` levels to ``"No"``.
  This is lossless because ``InternetService`` / ``PhoneService`` already carry the
  no-service signal, and it removes columns that are perfectly collinear with them.
- One-hot encode categoricals and standardize numerics. ``handle_unknown="ignore"``
  keeps the fitted pipeline safe if an unseen category shows up at serving time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data

logger = logging.getLogger(__name__)

TARGET = "Churn"
ID_COL = "customerID"
PROTECTED_COL = "gender"

# Internet add-ons whose "No internet service" level duplicates InternetService == "No". Information that can be derived from the InternetService column.
NO_INTERNET_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


@dataclass
class Dataset:
    """Cleaned modelling inputs."""

    X: pd.DataFrame  # feature columns only (numeric + categorical)
    y: pd.Series  # 0/1 churn target
    gender: pd.Series  # kept aside for fairness slicing, never a feature


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the EDA cleaning decisions and return a tidy copy."""
    df = df.copy()

    # TotalCharges arrives as STRING from the CSV load; coerce to numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # New customers (tenure == 0) simply have not been billed yet -> 0.
    new_customer = df["tenure"] == 0
    df.loc[new_customer, "TotalCharges"] = df.loc[new_customer, "TotalCharges"].fillna(0)

    # Any remaining blank with tenure != 0 is a likely data error -> drop + log.
    bad = df["TotalCharges"].isna()
    if bad.any():
        logger.warning(
            "Dropping %d row(s) with blank TotalCharges and tenure != 0 (data error).",
            int(bad.sum()),
        )
        df = df[~bad].copy()

    # Collapse redundant "no service" levels (lossless; see Takeaways).
    df[NO_INTERNET_COLS] = df[NO_INTERNET_COLS].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Normalise dtypes: nullable Int64/boolean columns can trip up sklearn transformers.
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)

    return df


def make_dataset(df: pd.DataFrame | None = None) -> Dataset:
    """Clean the data and split into features, target, and the protected attribute.

    Args:
        df: raw customers table. If ``None``, it is loaded from BigQuery.

    Returns:
        A :class:`Dataset` with features ``X``, target ``y``, and ``gender``.
    """
    if df is None:
        df = data.load_customers()

    df = clean(df)

    y = df[TARGET].astype(int)
    gender = df[PROTECTED_COL].astype(str)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    return Dataset(X=X, y=y, gender=gender)


def build_preprocessor(drop_first: bool = False) -> ColumnTransformer:
    """Build the feature-engineering ColumnTransformer.

    One-hot encodes the categoricals and standardizes the numerics. Standardizing
    is required by Logistic Regression and harmless for tree models (a monotonic
    transform does not change tree splits), so a single builder serves both.

    Args:
        drop_first: drop the first dummy level per categorical. Use ``True`` for
            Logistic Regression (avoids the dummy-variable trap and gives clean
            baseline-relative coefficients); ``False`` for tree models such as
            XGBoost, which are not hurt by collinearity.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first" if drop_first else None,
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = make_dataset()
    print(f"rows: {len(ds.X)}  features: {ds.X.shape[1]}")
    print(f"churn rate: {ds.y.mean():.3f}")
    print(f"gender values: {ds.gender.value_counts().to_dict()}")
