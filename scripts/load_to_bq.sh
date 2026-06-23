#!/usr/bin/env bash
# Phase 1: load the Telco churn CSV from GCS into BigQuery.
# Usage:
#   ./scripts/load_to_bq.sh
# Values are read from .env if present; override by exporting before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/../.env"
  set +a
fi

: "${GCP_PROJECT_ID:=churn-predictor-ml-2026}"
: "${GCP_REGION:=us-west1}"
: "${GCS_BUCKET:=churn-predictor-ml-artifacts}"
: "${BQ_DATASET:=churn_ml}"
: "${BQ_TABLE:=customers}"
: "${GCS_CSV_OBJECT:=telco-customer-churn.csv}"

SOURCE_URI="gs://${GCS_BUCKET}/${GCS_CSV_OBJECT}"
DEST_TABLE="${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE}"

echo "Source: ${SOURCE_URI}"
echo "Dest:   ${DEST_TABLE}"

# --autodetect infers the schema from the CSV header.
# --replace makes this re-runnable (overwrites the table each time).
# Note: TotalCharges has a few blank values, so autodetect types it as STRING.
#       We cast it to a number during preprocessing (Phase 2).
bq --location="${GCP_REGION}" load \
  --autodetect \
  --source_format=CSV \
  --skip_leading_rows=1 \
  --replace \
  "${DEST_TABLE}" \
  "${SOURCE_URI}"

echo ""
echo "Loaded. Row count:"
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) AS row_count FROM \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\`"

echo ""
echo "Done. Next: run the exploration queries in sql/01_explore.sql"
