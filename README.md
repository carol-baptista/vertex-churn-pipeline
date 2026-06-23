# vertex-churn-pipeline

Churn prediction portfolio project on **Google Cloud**: BigQuery → training → **Vertex AI Model Registry** → endpoint deployment.

Designed to demonstrate end-to-end ML on GCP, including model versioning and gated promotion (later phases).

## Architecture (target)

```text
BigQuery (features + labels)
    → train (local or Vertex)
    → Model Registry (v1, v2, …)
    → Vertex Endpoint (deploy / undeploy for demos)
    → prediction log (BigQuery, later)
```

## Cost note

Vertex AI is **not** always-free. This project is built to stay cheap:

- Train locally first (no Vertex compute)
- BigQuery free tier covers a small dataset
- Deploy endpoints only for demos, then **undeploy**

New GCP accounts get **$300 credit for 90 days**. See [docs/phase-0-setup.md](docs/phase-0-setup.md) for details.

## Phase 0 — Setup (current)

### Prerequisites

- Google Cloud account with billing (trial OK)
- `gcloud` CLI, `bq`, Python 3.10+

### Quick start

```bash
# 1. Clone and enter repo
cd vertex-churn-pipeline

# 2. Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your GCP_PROJECT_ID, GCS_BUCKET, etc.

# 4. Authenticate (one-time)
gcloud auth login
gcloud auth application-default login

# 5. Provision GCP resources
export GCP_PROJECT_ID=churn-predictor-ml-2026
export GCP_REGION=us-west1
export GCS_BUCKET=churn-predictor-ml-artifacts
./scripts/setup_gcp.sh
```

Full walkthrough: **[docs/phase-0-setup.md](docs/phase-0-setup.md)**

## Project phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | **Done** | GCP project, APIs, bucket, BQ dataset, local env |
| 1 | **In progress** | Load Telco churn data into BigQuery |
| 2 | Planned | Train & evaluate locally |
| 3 | Planned | Register model + deploy to Vertex endpoint |
| 4 | Planned | Second version + gated pipeline |

See [docs/phase-1-data.md](docs/phase-1-data.md) for the data loading walkthrough.

## Repo structure

```text
vertex-churn-pipeline/
├── configs/           # non-secret config
├── docs/              # setup & phase guides
├── scripts/           # setup_gcp.sh, load_to_bq.sh
├── sql/               # BigQuery exploration queries
├── src/               # training code (Phase 2)
├── requirements.txt
└── .env.example
```

## License

Portfolio / educational use.
