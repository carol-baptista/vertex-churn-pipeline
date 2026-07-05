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
- `gcloud` CLI, `bq`
- [`uv`](https://docs.astral.sh/uv/) (manages Python 3.10-3.12 + dependencies)

### Quick start

```bash
# 1. Clone and enter repo
cd vertex-churn-pipeline

# 2. Python env (uv — reproducible from uv.lock)
uv sync
source .venv/bin/activate

# macOS only: xgboost needs the OpenMP runtime
brew install libomp

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
| 2 | **Done** | EDA, train & evaluate locally |
| 3 | **In progress** | Register RF champion + deploy to Vertex endpoint — see [docs/phase-3-deploy.md](docs/phase-3-deploy.md) |
| 4 | Planned | Second version + gated pipeline; log predictions with `customerID` for ongoing monitoring |

See [docs/phase-1-data.md](docs/phase-1-data.md) for the data loading walkthrough.

## Repo structure

```text
vertex-churn-pipeline/
├── configs/           # non-secret config
├── docs/              # setup & phase guides
├── experiments/       # baseline vs engineered comparisons
├── models/            # trained artifacts (local, gitignored)
├── notebooks/         # EDA (01_eda.ipynb)
├── scripts/           # setup_gcp.sh, load_to_bq.sh
├── serving/           # CPR bundle (churn-rf/v1/, CHANGELOG.md)
├── sql/               # BigQuery exploration queries
├── src/               # pipeline library + CLI entrypoints (see below)
├── tests/
├── pyproject.toml     # dependencies (source of truth)
├── uv.lock            # pinned, reproducible env (committed)
├── requirements.txt   # kept in sync for non-uv users
└── .env.example
```

### Source layout (`src/`)

The pipeline code lives in a flat `src/` package (~9 modules). That matches the scope of this project: one dataset, one champion model line, and a small set of Makefile-driven commands.

| Module | Role |
|--------|------|
| `config.py`, `data.py`, `preprocess.py` | BigQuery load, feature prep, train/test split |
| `train.py` | Train all model types, tune threshold, write `metrics.json` (incl. fairness slices) |
| `champion.py` | Champion paths, artifact loading, serving manifest metadata |
| `predict.py` | Score a customer row locally (parity check before deploy) |
| `inspect.py` | Print saved fairness slices from `metrics.json` (`make fairness`) |
| `package.py` | Build `serving/churn-rf/v1/` CPR bundle |
| `deploy.py` | Upload bundle to GCS, register in Vertex, deploy endpoint |

Boundaries that matter more than folder depth:

- **Train** writes to `models/`; **serve** reads from `serving/churn-rf/vN/` (threshold applied outside the sklearn pipeline).
- **Notebooks** are for exploration; reusable logic stays in `src/`.
- **Serving** has its own `requirements.txt` — Vertex builds a minimal prediction image, not the full training env.

### How this would scale

For a larger system (batch scoring, monitoring, multiple model lines, shared feature store), I would split on **domain boundaries**, not file count:

```text
src/churn/
├── data.py, preprocess.py     # ingestion + features
├── train.py, champion.py      # training + artifact contract
└── cli/
    predict.py, inspect.py
    package.py, deploy.py      # thin entrypoints; library code stays importable
```

Further growth might add `training/`, `serving/`, and `monitoring/` packages once modules stop fitting in one directory or teams own separate areas. This repo stays flat until that pain shows up — avoiding structure for its own sake keeps the demo easy to walk through in an interview while still showing where the seams are.

## License

Portfolio / educational use.
