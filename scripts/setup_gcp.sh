#!/usr/bin/env bash
# Phase 0: enable APIs and create core GCP resources.
# Usage:
#   ./scripts/setup_gcp.sh
# Values are read from .env if present; otherwise the defaults below are used.
# Override any value by exporting it before running, e.g.:
#   GCP_PROJECT_ID=other-project ./scripts/setup_gcp.sh

set -euo pipefail

# Load .env if present (so you don't have to export vars manually).
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

echo "Project:  ${GCP_PROJECT_ID}"
echo "Region:   ${GCP_REGION}"
echo "Bucket:   gs://${GCS_BUCKET}"
echo "Dataset:  ${BQ_DATASET}"

gcloud config set project "${GCP_PROJECT_ID}"

echo "Enabling APIs (one-time, may take a minute)..."
gcloud services enable \
  bigquery.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudresourcemanager.googleapis.com

echo "Creating GCS bucket (skip if it already exists)..."
if ! gcloud storage buckets describe "gs://${GCS_BUCKET}" &>/dev/null; then
  gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --location="${GCP_REGION}" \
    --uniform-bucket-level-access
fi

echo "Creating BigQuery dataset (skip if it already exists)..."
if ! bq show --dataset "${GCP_PROJECT_ID}:${BQ_DATASET}" &>/dev/null; then
  bq --location="${GCP_REGION}" mk --dataset "${GCP_PROJECT_ID}:${BQ_DATASET}"
fi

: "${VERTEX_ARTIFACT_REPO:=vertex-churn}"
echo "Creating Artifact Registry repo (skip if it already exists)..."
if ! gcloud artifacts repositories describe "${VERTEX_ARTIFACT_REPO}" \
  --location="${GCP_REGION}" &>/dev/null; then
  gcloud artifacts repositories create "${VERTEX_ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${GCP_REGION}"
fi

echo ""
echo "Done. Next steps:"
echo "  1. Copy .env.example to .env and fill in values"
echo "  2. Run: gcloud auth application-default login"
echo "  3. Set a billing budget alert in Cloud Console"
echo "  4. Continue with Phase 1: load churn data into BigQuery"
