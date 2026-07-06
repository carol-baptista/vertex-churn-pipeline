# Project walkthrough

Extended notes for navigating this repo. Start with the README
[Project walkthrough](../README.md#project-walkthrough) section for linked
folders; use this doc for talking points and timing.

**Suggested flow:** README [System overview](../README.md#system-overview) →
topic tables below → open linked files → optional BigQuery Console.

---

## Opening (≈1 minute)

**Problem**

- Telco customer churn (~27% positive class).
- Retention teams need ranked risk scores, not raw model output.
- Goal: end-to-end pipeline on **Google Cloud** with sensible MLOps boundaries.

**Outcome**

- Train locally → register **Random Forest** champion in **Vertex Model Registry**.
- Batch-score into **`churn_ml.predictions`** in BigQuery for downstream use.
- Vertex holds the model; **BigQuery holds the scores**.

---

## Architecture (≈1 minute)

Use README [System overview](../README.md#system-overview) (vertical) and
[Pipelines at a glance](../README.md#pipelines-at-a-glance) (training + scoring).

| Component | Role |
|-----------|------|
| [Vertex Model Registry](docs/phase-3-deploy.md) | Which model version ran inference |
| BigQuery `predictions` | What analysts and retention workflows query |
| `make seed-scoring` | **Demo only** — simulates a label-free scoring population |

Production skips seed; ETL fills the scoring table (README: **How it would work in production**).

---

## Five topics to show in code (≈5–6 minutes)

### 1. Champion selection — deploy judgment, not autopilot

**Say:** XGBoost wins CV slightly; **Random Forest** wins **test** F1 and PR-AUC.

**Open**

- [serving/churn-rf/CHANGELOG.md](../serving/churn-rf/CHANGELOG.md)
- [models/random_forest/metrics.json](../models/random_forest/metrics.json)

| Metric | Test (RF) |
|--------|-----------|
| F1 | ~0.62 |
| Recall | ~71% |
| Precision | ~55% |
| Threshold | ~0.441 |

**Doc:** [phase-2-modeling.md](phase-2-modeling.md)

---

### 2. Evaluation discipline — what each split is for

**Say:** CV picks hyperparameters. Validation tunes threshold only. **Test** is the headline metric.

**Open**

- [phase-2-modeling.md](phase-2-modeling.md) — Metrics reporting
- [models/random_forest/metrics.json](../models/random_forest/metrics.json) — `validation` vs `test`

---

### 3. Fairness — exclude, then audit

**Say:** `gender` is never a model feature. Join back on `customerID` after scoring.

**Open**

- [src/preprocess.py](../src/preprocess.py) — `PROTECTED_COLS`
- [src/train.py](../src/train.py) — `fairness_audit()`
- [src/inspect.py](../src/inspect.py) — `make fairness`

---

### 4. Serving boundary — threshold outside the pipeline

**Say:** Pipeline ends at `predict_proba`; threshold in CPR postprocess / metadata.

**Open**

- [serving/churn-rf/v1/predictor.py](../serving/churn-rf/v1/predictor.py)
- [src/champion.py](../src/champion.py) — manifest
- [phase-3-deploy.md](phase-3-deploy.md)

---

### 5. Batch scoring → BigQuery — the production handoff

**Say:** Registry runs inference; predictions append to BQ with `run_id`, `scored_at`.

**Open**

- [src/batch.py](../src/batch.py)
- [phase-4-batch.md](phase-4-batch.md)
- [sql/02_predictions.sql](../sql/02_predictions.sql)

| Command | Use |
|---------|-----|
| `make score-local` | Dev / fast validation |
| `make score-vertex` | Production-like batch job |

**Cadence:** monthly (`0 6 1 * *`) — month-to-month contracts; features move on billing cycles.

---

## Optional live demo (≈2 minutes)

```bash
make fairness MODEL=random_forest
make predict CUSTOMER_ID=7590-VHVEG
```

Or screenshots: Model Registry (**us-west1**), BQ `predictions`, one batch job.

Avoid live `make deploy` or `make score-vertex` unless pre-tested (batch cold start is slow).

---

## Scope summary

**Complete (Phases 0–4):** data, train, fairness, Registry, predictions table.

**Planned (Phase 5):** Cloud Scheduler, monitoring, second model line.

**Cost choices:** local train; `REGISTER_ONLY=1`; batch not 24/7 endpoint.

---

## Anticipated questions

| Question | Short answer |
|----------|--------------|
| Why local train? | Faster iteration, lower cost; cloud proof is Registry + batch |
| Why batch not endpoint? | Monthly retention campaigns; batch is cheaper |
| Why RF over XGBoost? | Better **test** F1/PR-AUC |
| Train/serve skew? | Same `preprocess.clean` + 18 baseline cols |
| Why threshold outside pipeline? | Change policy without retrain |
| Why monthly schedule? | Billing-cycle feature updates |
| What's `seed-scoring`? | Demo substitute for production ETL table |

---

## Reviewer setup

```bash
uv sync
cp .env.example .env
make train
make test
```

**GCP region:** `us-west1`

| Path | Content |
|------|---------|
| [README.md](../README.md) | System diagram, walkthrough, phases |
| [docs/](.) | Phase guides |
| [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) | EDA |
| [tests/](../tests/) | Test suite |

---

## Timing cheat sheet

| Segment | Minutes |
|---------|---------|
| Opening + problem | 1 |
| README architecture | 1 |
| Five code topics | 5–6 |
| Demo or screenshots | 1–2 |
| Close + scope | 1 |
| **Total** | **8–10** |
