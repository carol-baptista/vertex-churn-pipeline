# Project walkthrough

Extended notes for navigating this repo. Start with the README
[Project walkthrough](../README.md#project-walkthrough) section for linked
folders; use this doc for topic summaries and file pointers.

**Suggested flow:** README [System overview](../README.md#system-overview) →
topics below → open linked files → optional BigQuery Console.

---

## Problem and outcome

**Problem**

- Telco customer churn (~27% positive class).
- Retention teams need ranked risk scores, not raw model output.
- Goal: end-to-end pipeline on **Google Cloud** with clear separation between training, serving, and the predictions table in BigQuery.

**Outcome**

- Train locally → register **Random Forest** champion in **Vertex Model Registry**.
- Batch-score into **`churn_ml.predictions`** in BigQuery for downstream use.
- Vertex holds the model; **BigQuery holds the scores**.

---

## Architecture

Use README [System overview](../README.md#system-overview) (vertical) and
[Pipelines at a glance](../README.md#pipelines-at-a-glance) (training + scoring).

| Component | Role |
|-----------|------|
| [Vertex Model Registry](docs/phase-3-deploy.md) | Which model version ran inference |
| BigQuery `predictions` | What analysts and retention workflows query |
| `make seed-scoring` | **Demo only** — simulates a label-free scoring population |

Production skips seed; ETL fills the scoring table (README: **How it would work in production**).

---

## Key topics

### 1. Champion selection — deploy judgment, not autopilot

XGBoost wins CV slightly; **Random Forest** wins **test** F1 and PR-AUC. Deploy champion chosen on held-out test performance.

**Open**

- [serving/churn-rf/CHANGELOG.md](../serving/churn-rf/CHANGELOG.md)
- [models/random_forest/metrics.json](../models/random_forest/metrics.json)

| Metric | Test (RF) |
|--------|-----------|
| F1 | ~0.62 |
| Recall | ~71% |
| Precision | ~55% |
| Threshold | ~0.441 |

**Doc:** [phase-2-modeling.md](phase-2-modeling.md) · [train-code-map.md](train-code-map.md)

---

### 2. Evaluation discipline — what each split is for

CV picks hyperparameters. Validation tunes threshold only. **Test** is the headline metric for stakeholders.

**Open**

- [phase-2-modeling.md](phase-2-modeling.md) — Metrics reporting
- [train-code-map.md](train-code-map.md) — `split_train_val_test`, `best_threshold`, `train_one`
- [models/random_forest/metrics.json](../models/random_forest/metrics.json) — `validation` vs `test`

---

### 3. Training loop — `train.py`

One orchestrator; the main path is `main()` → `train_model_suite()` → `train_one()`.

**Open**

- [train-code-map.md](train-code-map.md) — full section guide
- [src/train.py](../src/train.py) — `main()` ~1201, `train_one()` ~785

---

### 4. Fairness — exclude, then audit

`gender` is never a model feature. Join back on `customerID` after scoring.

**Open**

- [src/preprocess.py](../src/preprocess.py) — `PROTECTED_COLS`
- [src/train.py](../src/train.py) — `fairness_audit()`
- [src/inspect.py](../src/inspect.py) — `make fairness`

---

### 5. Serving boundary — threshold outside the pipeline

Pipeline ends at `predict_proba`; threshold applied in CPR postprocess / metadata.

**Open**

- [serving/churn-rf/v1/predictor.py](../serving/churn-rf/v1/predictor.py)
- [src/champion.py](../src/champion.py) — manifest
- [phase-3-deploy.md](phase-3-deploy.md)

---

### 6. Batch scoring → BigQuery — the production handoff

Registry runs inference; predictions append to BQ with `run_id`, `scored_at`.

**Open**

- [src/batch.py](../src/batch.py)
- [phase-4-batch.md](phase-4-batch.md)
- [sql/02_predictions.sql](../sql/02_predictions.sql)

| Command | Use |
|---------|-----|
| `make score-local` | Dev / fast validation |
| `make score-vertex` | Production-like batch job |
| `make warm-cache` | Post-batch cache export (hybrid pattern) |
| `make cache-lookup CUSTOMER_ID=…` | Simulated app read from cache |

**Cadence:** monthly (`0 6 1 * *`) — month-to-month contracts; features move on billing cycles.

**In-app reads:** batch + cache hybrid — [inference-patterns.md](inference-patterns.md). BQ stays source of truth; cache serves hot lookups without a 24/7 endpoint.

---

## Demo commands

```bash
make fairness MODEL=random_forest
make predict CUSTOMER_ID=7590-VHVEG          # debug: re-runs model
make score-local && make warm-cache
make cache-lookup CUSTOMER_ID=7590-VHVEG       # hybrid: read from cache
```

Screenshots alternative: Model Registry (**us-west1**), BQ `predictions`, one batch job.

`make deploy` and `make score-vertex` have long cold starts — run only if pre-tested.

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
| In-app churn risk? | Batch monthly + cache read; not a 24/7 endpoint |
| Why batch not endpoint? | Monthly cadence; batch + cache for in-app reads |
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
