# vertex-churn-pipeline

Churn prediction portfolio project on **Google Cloud**: BigQuery → local training → **Vertex Model Registry** → batch predictions back to BigQuery.

Designed to demonstrate end-to-end ML on GCP: train locally, register the champion in Vertex, batch-score into BigQuery for analytics.

## Pipelines at a glance

Two flows share the same **preprocessing logic** (`src/preprocess.py`) but differ after the model is fit: training writes artifacts to disk; scoring reads a BQ population and writes predictions back to BQ.

### Training pipeline

Historical data with labels → fit models → save artifacts + metrics.

```mermaid
flowchart LR
  subgraph ingest["Phase 1 — Data"]
    CSV["GCS CSV\ntelco-customer-churn.csv"]
    BQc["BigQuery\nchurn_ml.customers\n(+ Churn label)"]
    CSV -->|"load_to_bq.sh"| BQc
  end

  subgraph train["Phase 2 — make train"]
    Load["data.load_customers()"]
    Clean["preprocess.clean()\nTotalCharges, dtypes"]
    Split["make_dataset()\n18 baseline features\n+ customerID join key"]
    Hold["Stratified split\n70% train / 15% val / 15% test"]
    Prep["build_preprocessor()\nOHE + optional scale"]
    Fit["sklearn Pipeline\nLogReg · RF · XGB · LGBM"]
    Tune["CV on train\nthreshold on val"]
    Test["Test metrics\n+ fairness_by_gender"]
    Art["models/{model}/\nmodel.joblib\nmetrics.json"]
    BQc --> Load --> Clean --> Split --> Hold --> Prep --> Fit --> Tune --> Test --> Art
  end
```

**Commands:** `./scripts/load_to_bq.sh` → `make train` → `make fairness`

**Outputs:** `models/random_forest/model.joblib` (champion), `metrics.json` (threshold ~0.441, test F1/PR-AUC, fairness slices). Threshold is stored **outside** the sklearn pipeline and applied at scoring time.

Details: [docs/phase-2-modeling.md](docs/phase-2-modeling.md)

### Batch scoring pipeline

Score an **active customer population** (features only, no churn label) and append predictions to BigQuery for analytics and retention workflows.

```mermaid
flowchart LR
  subgraph demo["This repo — demo seed step"]
    BQfull["churn_ml.customers\n(static telco snapshot)"]
    BQscore["churn_ml.customers_scoring\nrandom sample · no Churn\n+ as_of_date"]
    BQfull -->|"make seed-scoring"| BQscore
  end

  subgraph prep["Shared preprocessing"]
    Read["load_scoring_frame()"]
    Clean2["preprocess.clean()\n18 raw feature cols"]
    Read --> Clean2
    BQscore --> Read
  end

  subgraph paths["Scoring path"]
    Local["make score-local\nlocal model.joblib"]
    Vertex["make score-vertex\nVertex BatchPredictionJob"]
    Clean2 --> Local
    Clean2 --> Vertex
  end

  subgraph infer["Inference"]
    Proba["predict_proba → threshold\nchurn_probability + churn_flag"]
    Local --> Proba
    GCSin["GCS input JSONL\n+customerID passthrough"]
    CPR["CPR container\n/predict"]
    GCSout["GCS output JSONL"]
    Vertex --> GCSin --> CPR --> GCSout --> Proba
  end

  subgraph out["Predictions table"]
    Pred["churn_ml.predictions\npartitioned by scored_at\nrun_id · model_version"]
    Proba -->|"write_predictions()"| Pred
  end
```

#### What is `make seed-scoring`?

**Seed scoring does not run the model.** It only creates (or replaces) the input table `churn_ml.customers_scoring` — the population you intend to score.

| | What it does | What it does *not* do |
|---|---|---|
| **`make seed-scoring`** | Copies a random sample from `customers`, drops the `Churn` label, adds `as_of_date` | Score anyone, change feature values, or write to `predictions` |
| **`make score-local` / `make score-vertex`** | Reads `customers_scoring`, preprocesses, runs the model, appends rows to `predictions` | Create the scoring population |

We drop `Churn` on purpose: in production you score **before** you know who cancelled. The training table (`customers`) keeps labels; the scoring table does not.

The sample is **not synthetic data** — it is real rows from the telco snapshot, unchanged except for removing the label. It stands in for “this period’s accounts to score” because this portfolio uses a static CSV, not a live CRM feed.

#### How it would work in production (no fake seed)

You would **skip `make seed-scoring` entirely**. An upstream pipeline would already maintain the scoring population in BigQuery:

```mermaid
flowchart LR
  subgraph prod["Production — no seed step"]
    CRM["CRM / billing / product DBs"]
    ETL["Daily ETL · monthly scoring cadence\ndbt · Dataflow · scheduled queries"]
    BQlive["BigQuery\nactive_customers\nor features_current"]
    CRM --> ETL --> BQlive
  end

  subgraph score["Same scoring steps as this repo"]
    Job["Cloud Scheduler → Cloud Run Job\nor Workflow"]
    Score["make score-vertex\nequivalent"]
    Pred["churn_ml.predictions\nappend-only by run_id"]
    BQlive --> Job --> Score --> Pred
  end

  subgraph consumers["Downstream — never call Vertex"]
    Ana["Analytics dashboards"]
    Ret["Retention campaigns / CRM export"]
    Mon["Monitoring · drift · fairness over time"]
    Pred --> Ana
    Pred --> Ret
    Pred --> Mon
  end
```

| Demo (this repo) | Production |
|---|---|
| `customers` = static telco CSV in BQ | Warehouse tables refreshed by ETL (tenure, charges, contract, etc.) |
| `make seed-scoring` = random sample, no label | `active_customers` (or similar) = all accounts due for scoring; no label column |
| Manual `make score-*` | Cloud Scheduler triggers batch job **monthly** (e.g. 1st of month, 6am) |
| Same `predictions` table shape | Same pattern: `customerID`, proba, flag, `scored_at`, `run_id`, `model_version` |

**Why monthly, not weekly?** The shortest contract in this dataset is month-to-month — tenure, charges, and contract status typically move on a **billing cycle**, not a weekly one. Scoring every week would mostly re-read unchanged rows. A monthly batch aligns with when features actually update and matches how retention teams often run outreach campaigns.

Vertex **Model Registry** holds *which model version* scored the batch. **BigQuery `predictions`** is what marketing, analytics, and ops actually query.

#### Commands (demo)

```bash
make seed-scoring          # 1. build customers_scoring (skip in production)
make score-local           # 2a. score with local artifact → BQ (free)
make score-vertex          # 2b. score via registered model → BQ (batch job cost)
```

Verify predictions:

```sql
SELECT customerID, churn_probability, churn_flag, run_id, scored_at
FROM `churn-predictor-ml-2026.churn_ml.predictions`
ORDER BY scored_at DESC
LIMIT 20;
```

Details: [docs/phase-4-batch.md](docs/phase-4-batch.md)

### Model registration (between train and Vertex scoring)

```mermaid
flowchart LR
  Art["models/random_forest/\nmodel.joblib"]
  Pkg["make package\nserving/churn-rf/v1/"]
  GCS["GCS artifacts\n+ CPR Docker image"]
  Reg["Vertex Model Registry\nchurn-predictor"]
  Art --> Pkg --> GCS -->|"make deploy REGISTER_ONLY=1"| Reg
  Reg -->|"make score-vertex"| Batch["BatchPredictionJob"]
```

Details: [docs/phase-3-deploy.md](docs/phase-3-deploy.md)

## Architecture (summary)

```text
GCS CSV → BigQuery customers
       → train → models/ (local artifacts)
       → package + register → Vertex Model Registry
       → batch score → BigQuery predictions  ← analytics / production consumers
```

Optional: `make deploy` (without `REGISTER_ONLY`) attaches the model to an **online endpoint** for real-time demos; monthly batch scoring does not require a running endpoint.

## Cost note

Vertex AI is **not** always-free. This project is built to stay cheap:

- Train locally first (no Vertex compute)
- BigQuery free tier covers a small dataset
- Deploy endpoints only for demos, then **undeploy**

New GCP accounts get **$300 credit for 90 days**. See [docs/phase-0-setup.md](docs/phase-0-setup.md) for details.

## Setup

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

End-to-end flow: **load data → train → register model → batch score to BQ**. Optional online endpoint for demos only.

| Phase | Status | What you get | Key commands / docs |
|-------|--------|--------------|---------------------|
| **0** | Done | GCP project, APIs, GCS bucket, BQ dataset, local Python env | [phase-0-setup.md](docs/phase-0-setup.md) · `./scripts/setup_gcp.sh` |
| **1** | Done | Telco CSV in BigQuery (`churn_ml.customers`) | [phase-1-data.md](docs/phase-1-data.md) · `./scripts/load_to_bq.sh` |
| **2** | Done | Trained models, threshold tuning, fairness slices, local artifacts | [phase-2-modeling.md](docs/phase-2-modeling.md) · `make train` · `make fairness` |
| **3** | Done | RF champion packaged; CPR image + **Model Registry** (`churn-predictor`, us-west1) | [phase-3-deploy.md](docs/phase-3-deploy.md) · `make package` · `make deploy REGISTER_ONLY=1` |
| **4** | Done | Batch scoring → **`churn_ml.predictions`** (local + Vertex batch paths) | [phase-4-batch.md](docs/phase-4-batch.md) · `make seed-scoring` · `make score-local` · `make score-vertex` |
| **5** | Planned | **Monthly** Cloud Scheduler + monitoring / second model version | Automate `score-vertex` (cron `0 6 1 * *`); prediction drift dashboards |

### Phase 4 note (demo vs production)

| Step | This repo (demo) | Production |
|------|------------------|------------|
| Scoring population | `make seed-scoring` → `customers_scoring` | ETL-maintained table (e.g. `active_customers`) — **no seed step** |
| Inference | `make score-local` or `make score-vertex` | Same batch job, triggered monthly |
| Consumers | Query `predictions` in BQ | Analytics, retention CRM, monitoring |

Phase 4 code is complete; Phase 5 is wiring the **monthly schedule** and optional observability (see [Batch scoring pipeline](#batch-scoring-pipeline) above).

### Suggested run order (first time)

```bash
./scripts/setup_gcp.sh && ./scripts/load_to_bq.sh   # phases 0–1
make train && make fairness                          # phase 2
make package-test && make deploy REGISTER_ONLY=1     # phase 3
make seed-scoring && make score-local                # phase 4 (demo)
make score-vertex                                    # phase 4 (Vertex batch — after re-register if needed)
```

**Walkthrough guide:** [docs/presentation-walkthrough.md](docs/presentation-walkthrough.md) — timed narrative, code paths, and Q&A prep.

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
| `batch.py` | Seed `customers_scoring`, score to `predictions` (local or Vertex batch) |

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

Further growth might add `training/`, `serving/`, and `monitoring/` packages once modules stop fitting in one directory or teams own separate areas. This repo stays flat until that pain shows up — avoiding structure for its own sake keeps the walkthrough simple while still showing where the seams are.

## License

Portfolio / educational use.
