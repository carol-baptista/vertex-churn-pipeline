"""Post-batch cache warm — batch + cache hybrid showcase.

After ``make score-local`` or ``make score-vertex``, predictions live in
BigQuery (source of truth). This module exports the latest score per customer
to a local JSONL file as a stand-in for Redis / Memorystore populated from
``predictions_latest`` (see sql/03_predictions_latest.sql).

Production: same query, written by a Cloud Run Job triggered when the batch
job succeeds — not on every app request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import config, data

CACHE_FIELDS = (
    "customerID",
    "churn_probability",
    "churn_flag",
    "threshold",
    "model",
    "model_version",
    "run_id",
    "scored_at",
)

DEFAULT_CACHE_PATH = config.REPO_ROOT / "data" / "cache" / "churn_scores.jsonl"

LATEST_SCORES_SQL = f"""
SELECT
  customerID,
  churn_probability,
  churn_flag,
  threshold,
  model,
  model_version,
  run_id,
  scored_at
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY customerID
      ORDER BY scored_at DESC, run_id DESC
    ) AS rn
  FROM `{config.PREDICTIONS_TABLE_ID}`
)
WHERE rn = 1
"""


def fetch_latest_scores() -> list[dict[str, Any]]:
    """Latest prediction row per customer from BigQuery."""
    df = data.get_client().query(LATEST_SCORES_SQL).to_dataframe()
    if df.empty:
        raise ValueError(
            f"No rows in {config.PREDICTIONS_TABLE_ID}. "
            "Run make score-local or make score-vertex first."
        )
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        record = {field: getattr(row, field) for field in CACHE_FIELDS}
        record["scored_at"] = str(record["scored_at"])
        records.append(record)
    return records


def write_cache(records: list[dict[str, Any]], path: Path = DEFAULT_CACHE_PATH) -> Path:
    """Write one JSON object per line (easy to load into Redis HSET/MSET)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return path


def load_cache(path: Path = DEFAULT_CACHE_PATH) -> dict[str, dict[str, Any]]:
    """Load cache file into a customerID → score dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found at {path}. Run make warm-cache after scoring."
        )
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[str(record["customerID"])] = record
    return out


def lookup(customer_id: str, *, path: Path = DEFAULT_CACHE_PATH) -> dict[str, Any]:
    """Simulate a low-latency app read from cache (no model, no BQ hot path)."""
    cache = load_cache(path)
    if customer_id not in cache:
        raise KeyError(f"No cached score for customerID={customer_id!r}")
    return cache[customer_id]


def warm_cache(*, path: Path = DEFAULT_CACHE_PATH) -> Path:
    records = fetch_latest_scores()
    out = write_cache(records, path)
    print(f"Warmed cache -> {out} ({len(records)} customers)")
    print("  (Production: same rows → Redis/Memorystore after each batch run)")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export latest BQ predictions to a read cache (hybrid pattern demo)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    warm = sub.add_parser("warm", help="export predictions_latest → JSONL cache file")
    warm.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"cache file path (default: {DEFAULT_CACHE_PATH})",
    )

    get = sub.add_parser("lookup", help="read one customer from cache (app hot path)")
    get.add_argument("--customer-id", required=True)
    get.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"cache file path (default: {DEFAULT_CACHE_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "warm":
        warm_cache(path=args.out)
    elif args.command == "lookup":
        result = lookup(args.customer_id, path=args.cache)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
