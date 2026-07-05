"""Tests for src.deploy (no live GCP calls)."""

from __future__ import annotations

import os

import pytest

from src.deploy import GCS_PREFIX, gcs_uri


def test_gcs_prefix():
    assert GCS_PREFIX == "models/churn-rf/v1"


def test_gcs_uri_requires_bucket(monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    with pytest.raises(ValueError, match="GCS_BUCKET"):
        gcs_uri()


def test_gcs_uri_format(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "my-bucket")
    assert gcs_uri() == f"gs://my-bucket/{GCS_PREFIX}"
