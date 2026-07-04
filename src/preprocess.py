"""Preprocessing: clean the raw churn table and build the feature pipeline.

The decisions here come straight from the EDA "Takeaways" cell in
``notebooks/01_eda.ipynb``:

- ``TotalCharges`` is loaded as STRING; coerce to numeric. New customers
  (``tenure == 0``) have not been billed yet, so blanks become 0. Any other blank
  is treated as a data error: the row is dropped and logged.
- Drop ``customerID`` and protected attributes (e.g. ``gender``) from model features.
  Keep ``customerID`` as a join key only; re-attach protected columns from the
  source table when running fairness audits after scoring.
- One-hot encode categoricals, **drop** redundant ``*No internet service`` / ``*No phone service``
  dummies (identical to ``InternetService_No`` / ``PhoneService``). Standardize numerics
  for linear models only (Logistic Regression); tree models skip scaling.
- Optional **engineered** features (default for training): ``avg_monthly_charge``,
  ``tenure_bucket``, ``addon_count``, ``month_to_month_electronic``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data

logger = logging.getLogger(__name__)

TARGET = "Churn"
ID_COL = "customerID"
PROTECTED_COLS = ["gender"]  # excluded from X; joined back on ID_COL for fairness audits

# One-hot columns containing these markers are identical to InternetService_No /
# PhoneService and are dropped after encoding (see CategoricalEncoder).
REDUNDANT_DUMMY_MARKERS = ("No internet service", "No phone service")

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

ENGINEERED_NUMERIC_FEATURES = [
    "avg_monthly_charge",
    "addon_count",
    "month_to_month_electronic",
]

ENGINEERED_CATEGORICAL_FEATURES = ["tenure_bucket"]

ADDON_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

FEATURE_SETS = ("baseline", "engineered")


def feature_columns(*, engineered: bool) -> tuple[list[str], list[str]]:
    """Return numeric and categorical column lists for a feature set."""
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    if engineered:
        numeric.extend(ENGINEERED_NUMERIC_FEATURES)
        categorical.extend(ENGINEERED_CATEGORICAL_FEATURES)
    return numeric, categorical


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EDA-driven features on top of the cleaned raw columns."""
    df = df.copy()
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype(str)
    df["addon_count"] = (df[ADDON_COLUMNS] == "Yes").sum(axis=1).astype(float)
    df["month_to_month_electronic"] = (
        (df["Contract"] == "Month-to-month")
        & (df["PaymentMethod"] == "Electronic check")
    ).astype(float)
    return df


@dataclass
class Dataset:
    """Cleaned modelling inputs."""

    X: pd.DataFrame  # feature columns only (numeric + categorical)
    y: pd.Series  # 0/1 churn target
    customer_id: pd.Series  # join key for eval / fairness audits, never a feature


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

    # Normalise dtypes: nullable Int64/boolean columns can trip up sklearn transformers.
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)

    return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode categoricals and drop redundant *No service* dummies.

    Levels like ``OnlineSecurity_No internet service`` are structurally identical to
    ``InternetService_No``. Dropping those columns after encoding keeps add-on
    ``_No`` / ``_Yes`` dummies semantically clean: ``OnlineSecurity_No`` means
    "declined the add-on", not "has no internet".
    """

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            drop="first" if self.drop_first else None,
            sparse_output=False,
        )
        self.encoder_.fit(X)
        all_names = self.encoder_.get_feature_names_out()
        self.keep_mask_ = np.array(
            [
                not any(marker in name for marker in REDUNDANT_DUMMY_MARKERS)
                for name in all_names
            ]
        )
        self.feature_names_out_ = all_names[self.keep_mask_]
        return self

    def transform(self, X):
        encoded = self.encoder_.transform(X)
        return encoded[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def demographics_table(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Protected attributes keyed by ``customerID`` for post-scoring fairness joins."""
    return cleaned_df[[ID_COL, *PROTECTED_COLS]].copy()


def dataset_from_cleaned(
    cleaned_df: pd.DataFrame, *, engineered: bool = True
) -> Dataset:
    """Build modelling inputs from an already-cleaned customers table."""
    frame = engineer_features(cleaned_df) if engineered else cleaned_df
    numeric, categorical = feature_columns(engineered=engineered)
    y = frame[TARGET].astype(int)
    X = frame[numeric + categorical]
    return Dataset(X=X, y=y, customer_id=frame[ID_COL])


def make_dataset(df: pd.DataFrame | None = None, *, engineered: bool = True) -> Dataset:
    """Clean the data and split into features, target, and join key.

    Args:
        df: raw customers table. If ``None``, it is loaded from BigQuery.

    Returns:
        A :class:`Dataset` with features ``X``, target ``y``, and ``customer_id``.
        Protected columns are *not* included; use :func:`demographics_table` on the
        cleaned frame and join on ``customerID`` when auditing fairness.
    """
    if df is None:
        df = data.load_customers()

    return dataset_from_cleaned(clean(df), engineered=engineered)


def build_preprocessor(
    drop_first: bool = False,
    scale_numeric: bool = True,
    *,
    engineered: bool = True,
) -> ColumnTransformer:
    """Build the feature-engineering ColumnTransformer.

    One-hot encodes the categoricals (dropping redundant *No service* dummies).
    Numerics are optionally standardized (required for Logistic Regression; unnecessary
    for tree models such as XGBoost).

    Args:
        drop_first: drop the first dummy level per categorical. Use ``True`` for
            Logistic Regression (avoids the dummy-variable trap and gives clean
            baseline-relative coefficients); ``False`` for tree models such as
            XGBoost, which are not hurt by collinearity.
        scale_numeric: apply ``StandardScaler`` to numeric columns. Use ``True`` for
            Logistic Regression; ``False`` for tree models.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    num_transformer: StandardScaler | str = (
        StandardScaler() if scale_numeric else "passthrough"
    )
    numeric, categorical = feature_columns(engineered=engineered)
    encoder = CategoricalEncoder(drop_first=drop_first)
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric),
            ("cat", encoder, categorical),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = make_dataset()
    print(f"rows: {len(ds.X)}  features: {ds.X.shape[1]}")
    print(f"churn rate: {ds.y.mean():.3f}")
    print(f"sample customer_id: {ds.customer_id.iloc[0]}")
