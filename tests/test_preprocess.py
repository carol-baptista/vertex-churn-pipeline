"""Unit tests for src.preprocess cleaning rules (no BigQuery required)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess import clean, make_dataset


def _row(*, tenure: int, total_charges, customer_id: str = "cust-1") -> dict:
    """Minimal valid customers-table row for preprocessing tests."""
    return {
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
