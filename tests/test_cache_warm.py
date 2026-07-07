"""Tests for src.cache_warm (no live BQ)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cache_warm import load_cache, lookup, write_cache


def test_write_load_lookup_roundtrip(tmp_path: Path):
    records = [
        {
            "customerID": "7590-VHVEG",
            "churn_probability": 0.72,
            "churn_flag": 1,
            "threshold": 0.441,
            "model": "random_forest",
            "model_version": "v1",
            "run_id": "test-run",
            "scored_at": "2026-07-01T06:00:00+00:00",
        },
        {
            "customerID": "1234-ABCDE",
            "churn_probability": 0.15,
            "churn_flag": 0,
            "threshold": 0.441,
            "model": "random_forest",
            "model_version": "v1",
            "run_id": "test-run",
            "scored_at": "2026-07-01T06:00:00+00:00",
        },
    ]
    path = tmp_path / "cache.jsonl"
    write_cache(records, path)

    loaded = load_cache(path)
    assert set(loaded) == {"7590-VHVEG", "1234-ABCDE"}

    hit = lookup("7590-VHVEG", path=path)
    assert hit["churn_probability"] == 0.72
    assert hit["churn_flag"] == 1


def test_lookup_missing_customer(tmp_path: Path):
    path = tmp_path / "cache.jsonl"
    write_cache(
        [{"customerID": "x", "churn_probability": 0.5, "churn_flag": 0,
          "threshold": 0.441, "model": "rf", "model_version": "v1",
          "run_id": "r", "scored_at": "2026-01-01"}],
        path,
    )
    with pytest.raises(KeyError, match="not-there"):
        lookup("not-there", path=path)


def test_cache_jsonl_one_object_per_line(tmp_path: Path):
    path = tmp_path / "cache.jsonl"
    write_cache([{"customerID": "a", "churn_probability": 0.1}], path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["customerID"] == "a"
