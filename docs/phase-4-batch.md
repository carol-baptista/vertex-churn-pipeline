# Phase 4 — Batch scoring to BigQuery

Score a **live-like population** from BigQuery and persist predictions back to BigQuery for analytics and downstream use.

Vertex Model Registry holds the model; **BigQuery holds the scores** — that is what analysts and retention workflows query.

## Prerequisites

- Phase 1: `customers` table loaded (`./scripts/load_to_bq.sh`)
- Phase 2: `make train`
- Phase 3: `make deploy REGISTER_ONLY=1` (required only for `make score-vertex`)
- `.env` with `GCS_BUCKET`, `GCP_REGION=us-west1`

## Tables

| Table | Purpose |
|---|---|
| `churn_ml.customers` | Full historical dataset (with `Churn` label) |
| `churn_ml.customers_scoring` | Random sample **without labels** — simulates production scoring population |
| `churn_ml.predictions` | Append-only scored output (partitioned by `scored_at`) |

## Commands (recommended order)

### 1. Seed fake “live” scoring data (free)

```bash
make seed-scoring              # 500 random customers, Churn label dropped
make seed-scoring LIMIT=100    # smaller sample
```

This creates `customers_scoring` with an `as_of_date` column — as if you exported this month’s active accounts without labels.

### 2. Score locally → write predictions to BQ (free)

Uses the same `models/random_forest/model.joblib` as training — good for testing the BQ integration before paying for Vertex batch compute:

```bash
make score-local
```

Verify in BigQuery:

```sql
SELECT customerID, churn_probability, churn_flag, run_id, scored_at
FROM `churn-predictor-ml-2026.churn_ml.predictions`
ORDER BY scored_at DESC
LIMIT 20;
```

More queries: [sql/02_predictions.sql](../sql/02_predictions.sql)

### 3. Score via Vertex batch job (production path)

Uses the registered `churn-predictor` model and CPR container:

```bash
make score-vertex
```

Flow:

```text
customers_scoring (BQ)
  → export JSONL to GCS (with customerID passthrough)
  → Vertex BatchPredictionJob
  → load JSONL output → churn_ml.predictions
```

**Note:** CPR models need `/predict` and `/health` routes on upload for batch prediction. If `make score-vertex` fails with a PredictRoute/HealthRoute error, re-register:

```bash
make deploy REGISTER_ONLY=1
make score-vertex
```

## Local vs Vertex scoring

| | `make score-local` | `make score-vertex` |
|---|---|---|
| Model source | Local `model.joblib` | Vertex Registry + CPR image |
| Cost | BQ storage only | Batch job compute (~minutes) |
| Use case | Dev / BQ pipeline test | Production-like inference |
| Output | Same `predictions` table | Same `predictions` table |

## Monthly schedule (not automated yet)

In production you would trigger `score-vertex` (or a Cloud Run wrapper) on a cron, e.g. Cloud Scheduler on the **1st of each month at 6am**. Phase 4 stops at the Makefile commands; Scheduler wiring is a small follow-up once batch scoring is verified.

**Cadence rationale:** contracts here are at least month-to-month (many longer). Feature values (`tenure`, `MonthlyCharges`, `TotalCharges`, contract type) change on billing cycles, not weekly — so monthly scoring is enough and avoids redundant batch jobs.

## Interview one-liner

> "Registry versions the model; monthly batch scoring writes probabilities and flags to BigQuery with `customerID`, `run_id`, and `scored_at` so analytics and retention never call Vertex directly."
