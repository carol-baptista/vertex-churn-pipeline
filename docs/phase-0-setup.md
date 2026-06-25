# Phase 0 — GCP & local setup

## Can you use Vertex AI for free?

**Short answer:** not completely free forever, but you can keep this portfolio project **very cheap** or covered by **$300 trial credit**.

| Service | Always free? | Notes for this project |
|---------|--------------|------------------------|
| **BigQuery** | Partially | ~10 GiB storage + 1 TiB queries/month on free tier ([details](https://cloud.google.com/free)) |
| **Cloud Storage** | Partially | 5 GB-month in `us-east1` / `us-west1` / `us-central1` only |
| **Vertex AI Training** | No | Billed per compute hour when you run a training job |
| **Vertex AI Endpoints** | No | Billed while a model is **deployed** (even idle) — **undeploy when not demoing** |
| **Model Registry** | Cheap | Storing model artifacts on GCS; registry metadata is low cost |
| **Vertex Express Mode** | Limited | Gen AI only (Gemini) — **not** for tabular churn ML |

### New Google Cloud accounts

- **$300 credit for 90 days** on a free trial ([Google Cloud free program](https://cloud.google.com/free))
- Credit card required for identity verification; you are **not** auto-charged when the trial ends unless you upgrade to paid
- This portfolio fits comfortably in $300 if you **undeploy endpoints** and **train locally** first

### Cost-saving strategy (recommended)

1. **Train on your laptop** (Phases 1–2) — $0 Vertex compute
2. **Use BigQuery free tier** for a small Telco churn dataset (~7k rows)
3. **Upload to Model Registry** only when you want to demo versioning
4. **Deploy endpoint briefly** for screenshots/demo, then **undeploy**
5. **Set a budget alert** at $10–20 so you get emailed before surprises

> **Rule of thumb:** training + registry = pennies to a few dollars. **Endpoints left running 24/7** are what burn credit.

---

## Step 1 — Create a GCP project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project, e.g. `vertex-churn-portfolio`
3. Link a billing account (trial or paid)
4. Note your **Project ID** (not display name)

---

## Step 2 — Install tools (macOS)

### Google Cloud SDK (`gcloud` + `bq`)

```bash
brew install --cask google-cloud-sdk
```

Restart your terminal, then:

```bash
gcloud init
gcloud auth login
gcloud auth application-default login
```

`application-default login` lets Python libraries (`google-cloud-*`) use your credentials locally.

### Python env with uv (recommended)

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible
environments. The exact dependency versions are pinned in `uv.lock`.

```bash
brew install uv

# Creates .venv (Python 3.12) and installs everything from uv.lock
uv sync
source .venv/bin/activate
```

macOS only: `xgboost` needs the OpenMP runtime, so also run:

```bash
brew install libomp
```

> `pyproject.toml` holds the direct dependencies; `uv.lock` is the fully
> pinned graph (committed to git). Anyone cloning the repo gets the identical
> environment with `uv sync`. A `requirements.txt` is also kept for users
> without uv (`pip install -r requirements.txt`).

---

## Step 3 — Configure environment

```bash
cp .env.example .env
# Edit .env with your project ID, region, bucket name
```

Suggested region for western Canada: `us-west1` (Oregon). Keeps bucket, BigQuery, and Vertex AI in one region.

---

## Step 4 — Enable APIs & create resources

```bash
export GCP_PROJECT_ID=churn-predictor-ml-2026
export GCP_REGION=us-west1
export GCS_BUCKET=churn-predictor-ml-artifacts

chmod +x scripts/setup_gcp.sh
./scripts/setup_gcp.sh
```

This enables BigQuery, Vertex AI, Cloud Storage, and creates:

- GCS bucket for model artifacts
- BigQuery dataset `churn_ml`

---

## Step 5 — Budget alert (important)

1. Console → **Billing** → **Budgets & alerts**
2. Create budget: e.g. **$20/month**
3. Alert at 50%, 90%, 100%

---

## Step 6 — Verify access

```bash
gcloud config get-value project
bq ls
gcloud ai models list --region="${GCP_REGION}" 2>/dev/null || echo "Vertex OK (empty list is fine)"
python -c "from google.cloud import bigquery; print('BigQuery client OK')"
```

---

## IAM (if using your user account)

Your Google user needs these roles on the project (Owner or a combo):

- `roles/bigquery.admin` (or dataEditor + jobUser)
- `roles/storage.admin`
- `roles/aiplatform.user`

For a portfolio, **Owner** on a dedicated project is simplest.

---

## What’s next (Phase 1)

- Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset
- Load into `churn_ml.customers` in BigQuery
- Explore with SQL
