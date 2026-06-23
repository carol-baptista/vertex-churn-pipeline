# Phase 1 — Data in BigQuery

Goal: get the Telco churn data into BigQuery as `churn_ml.customers` and explore it with SQL.

## Prerequisites (from Phase 0)

- Project `churn-predictor-ml-2026` set and authenticated
- Dataset `churn_ml` created in `us-west1`
- CSV uploaded to the bucket: `gs://churn-predictor-ml-artifacts/telco-customer-churn.csv`

## Step 1 — Load the CSV into BigQuery

```bash
./scripts/load_to_bq.sh
```

This runs `bq load` from GCS into `churn_ml.customers` with:

- `--autodetect` — infers the schema from the header row
- `--replace` — safe to re-run; overwrites the table
- prints the row count when done (expect ~7,043 rows)

## Step 2 — Explore with SQL

Open the BigQuery console (or use `bq query`) and run the queries in
[`sql/01_explore.sql`](../sql/01_explore.sql). They cover:

- Overall churn rate (~26% churn — the dataset is imbalanced)
- Duplicate check (rows vs distinct `customerID`)
- Churn by contract type and tenure (strong predictors)
- Monthly charges by churn

Example (run one query directly):

```bash
bq query --use_legacy_sql=false \
  "SELECT Churn, COUNT(*) c FROM \`churn-predictor-ml-2026.churn_ml.customers\` GROUP BY Churn"
```

## Known data quirk: `TotalCharges`

A handful of rows (new customers with `tenure = 0`) have a blank `TotalCharges`.
Because of those blanks, autodetect loads the column as **STRING**, not a number.

We don't fix it here — it's handled in **Phase 2 preprocessing** (cast to float,
impute the blanks). Query 6 in `01_explore.sql` counts them.

## Dataset columns (reference)

| Column | Type | Notes |
|--------|------|-------|
| `customerID` | STRING | Unique ID (drop before training) |
| `gender`, `Partner`, `Dependents` | STRING | Demographics |
| `SeniorCitizen` | INT | 0/1 |
| `tenure` | INT | Months as customer |
| `PhoneService`, `InternetService`, ... | STRING | Services subscribed |
| `Contract` | STRING | Month-to-month / One year / Two year |
| `MonthlyCharges` | FLOAT | Current monthly charge |
| `TotalCharges` | STRING* | Cast to float in Phase 2 |
| `Churn` | STRING | **Target**: Yes / No |

## What's next (Phase 2)

- Read `churn_ml.customers` into a training script
- Preprocess (cast `TotalCharges`, encode categoricals, split)
- Train + evaluate a baseline model locally (no Vertex cost yet)
