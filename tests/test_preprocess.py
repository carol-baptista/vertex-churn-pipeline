"""Unit tests for src.preprocess cleaning rules (no BigQuery required)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess import build_preprocessor, clean, engineer_features, make_dataset


def _row(*, tenure: int, total_charges, customer_id: str = "cust-1", **overrides) -> dict:
    """Minimal valid customers-table row for preprocessing tests."""
    row = {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": True,
        "Dependents": False,
        "tenure": tenure,
        "PhoneService": True,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": True,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": total_charges,
        "Churn": False,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("blank_value", ["", " ", pd.NA])
def test_tenure_zero_fills_blank_total_charges(blank_value):
    df = pd.DataFrame([_row(tenure=0, total_charges=blank_value)])

    out = clean(df)

    assert len(out) == 1
    assert out.loc[0, "TotalCharges"] == 0.0


@pytest.mark.parametrize("blank_value", ["", " ", pd.NA])
def test_tenure_positive_drops_blank_total_charges(blank_value):
    df = pd.DataFrame([_row(tenure=12, total_charges=blank_value)])

    out = clean(df)

    assert out.empty


def test_make_dataset_drops_bad_total_charges_keeps_new_customers():
    df = pd.DataFrame(
        [
            _row(tenure=0, total_charges="", customer_id="new"),
            _row(tenure=12, total_charges="", customer_id="bad"),
            _row(tenure=24, total_charges="1200.50", customer_id="ok"),
        ]
    )

    ds = make_dataset(df)

    assert len(ds.X) == 2
    assert set(ds.X.index) == {0, 2}
    assert ds.X.loc[0, "TotalCharges"] == 0.0
    assert ds.X.loc[2, "TotalCharges"] == 1200.50


def test_no_internet_does_not_encode_add_on_as_declined():
    df = pd.DataFrame(
        [
            _row(
                tenure=12,
                total_charges=600,
                customer_id="no-internet",
                InternetService="No",
                OnlineSecurity="No internet service",
                OnlineBackup="No internet service",
                DeviceProtection="No internet service",
                TechSupport="No internet service",
                StreamingTV="No internet service",
                StreamingMovies="No internet service",
            )
        ]
    )
    ds = make_dataset(df)
    prep = build_preprocessor(drop_first=False)
    matrix = prep.fit_transform(ds.X)
    names = prep.get_feature_names_out()
    values = dict(zip(names, matrix[0]))

    assert values["cat__InternetService_No"] == 1.0
    assert values.get("cat__OnlineSecurity_No", 0.0) == 0.0
    assert "cat__OnlineSecurity_No internet service" not in values


def test_dsl_declined_add_on_encodes_as_no():
    df = pd.DataFrame(
        [
            _row(
                tenure=12,
                total_charges=600,
                customer_id="dsl-declined",
                InternetService="DSL",
                OnlineSecurity="No",
            )
        ]
    )
    ds = make_dataset(df)
    prep = build_preprocessor(drop_first=False)
    matrix = prep.fit_transform(ds.X)
    names = prep.get_feature_names_out()
    values = dict(zip(names, matrix[0]))

    assert values["cat__InternetService_DSL"] == 1.0
    assert values["cat__OnlineSecurity_No"] == 1.0


def test_tree_preprocessor_does_not_scale_numerics():
    df = pd.DataFrame([_row(tenure=12, total_charges=600, MonthlyCharges=99.5)])
    ds = make_dataset(df, engineered=False)
    prep = build_preprocessor(drop_first=False, scale_numeric=False, engineered=False)
    matrix = prep.fit_transform(ds.X)
    names = prep.get_feature_names_out()
    values = dict(zip(names, matrix[0]))

    assert values["num__tenure"] == 12.0
    assert values["num__MonthlyCharges"] == 99.5
    assert values["num__TotalCharges"] == 600.0


def test_engineer_features_adds_eda_columns():
    df = pd.DataFrame(
        [
            _row(
                tenure=24,
                total_charges=1200,
                Contract="Month-to-month",
                PaymentMethod="Electronic check",
                OnlineSecurity="Yes",
                StreamingTV="Yes",
            )
        ]
    )
    out = engineer_features(clean(df))

    assert out["avg_monthly_charge"].iloc[0] == 50.0
    assert out["tenure_bucket"].iloc[0] == "13-24"
    assert out["addon_count"].iloc[0] == 2.0
    assert out["month_to_month_electronic"].iloc[0] == 1.0


def test_make_dataset_engineered_has_more_features_than_baseline():
    df = pd.DataFrame([_row(tenure=12, total_charges=600)])

    baseline = make_dataset(df, engineered=False)
    engineered = make_dataset(df, engineered=True)

    assert engineered.X.shape[1] == baseline.X.shape[1] + 4
    assert "avg_monthly_charge" in engineered.X.columns
    assert "tenure_bucket" in engineered.X.columns
