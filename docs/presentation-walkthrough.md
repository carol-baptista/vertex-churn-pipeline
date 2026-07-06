# Project walkthrough guide

Structured notes for a **5–10 minute code presentation** of this repo. Use the
README diagrams for architecture, then jump into code for five design decisions.

**Suggested screen flow:** README → one or two source files per topic → BigQuery
Console (optional) → back to README phase table for scope.

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

**One-line summary**

> BigQuery → train → Registry → batch predict → BigQuery predictions.

---

## Architecture (≈1 minute)

Open README **Pipelines at a glance** and walk both diagrams:

1. **Training** — `customers` (with label) → preprocess → 70/15/15 split → four
   models → champion artifacts in `models/`.
2. **Scoring** — `customers_scoring` (no label) → same preprocess → score →
   append `predictions`.

Land this distinction:

| Component | Role |
|-----------|------|
| Vertex Model Registry | Which model version ran inference |
| BigQuery `predictions` | What analysts and retention workflows query |
| `make seed-scoring` | **Demo only** — simulates a label-free scoring population |

Production would skip seed and read from an ETL-maintained table (README:
**How it would work in production**).

---

## Five topics to show in code (≈5–6 minutes)

Pick these — they map to problem framing, ML rigour, fairness, serving, and
cloud delivery.

### 1. Champion selection — deploy judgment, not autopilot

**Say:** XGBoost wins CV slightly; **Random Forest** wins **test** F1 and PR-AUC.
I chose the deploy champion on held-out test performance.

**Open**

- `serving/churn-rf/CHANGELOG.md` — why v1 is RF
- `models/random_forest/metrics.json` — test vs validation vs `cv_score`

**Numbers to know**

| Metric | Test (RF) |
|--------|-----------|
| F1 | ~0.62 |
| Recall | ~71% |
| Precision | ~55% |
| Threshold | ~0.441 |

**Doc:** [phase-2-modeling.md](phase-2-modeling.md)

---

### 2. Evaluation discipline — what each split is for

**Say:** CV picks hyperparameters. Validation tunes threshold only. **Test** is
the headline metric for stakeholders.

**Open**

- `docs/phase-2-modeling.md` — Metrics reporting table
- `models/random_forest/metrics.json` — `validation` vs `test` blocks

**Key line**

> “I’d report test recall/precision/F1 — not validation, not CV.”

---

### 3. Fairness — exclude, then audit

**Say:** `gender` is never a model feature. After scoring, I join it back on
`customerID` and slice metrics by group.

**Open**

- `src/preprocess.py` — `PROTECTED_COLS`, `ID_COL`
- `src/train.py` — `fairness_audit()` (briefly)
- Terminal: `make fairness MODEL=random_forest`

**Why it matters:** avoids direct use of protected attributes while still checking
disparate impact via proxies (contract type, tenure, etc.).

---

### 4. Serving boundary — threshold outside the pipeline

**Say:** The sklearn pipeline ends at `predict_proba`. Threshold is applied in
serving metadata / CPR postprocess so policy can change without retraining.

**Open**

- `serving/churn-rf/v1/predictor.py` — `preprocess` → `predict` → `postprocess`
- `serving/churn-rf/v1/threshold.json` (after `make package`)
- `src/champion.py` — `build_manifest()`, paths

**CPR point:** bundle is self-contained (no `src/` imports in the container).

**Doc:** [phase-3-deploy.md](phase-3-deploy.md)

---

### 5. Batch scoring → BigQuery — the production handoff

**Say:** Registry runs inference; predictions land in BQ with `customerID`,
`run_id`, `scored_at`. Downstream never calls Vertex.

**Open**

- `src/batch.py` — `seed_scoring_table`, `score_frame`, `write_predictions`
- README — demo vs production table
- BQ query:

```sql
SELECT customerID, churn_probability, churn_flag, run_id, scored_at
FROM `churn-predictor-ml-2026.churn_ml.predictions`
ORDER BY scored_at DESC
LIMIT 10;
```

**Cadence:** monthly batch (`0 6 1 * *`) — shortest contract is month-to-month;
weekly scoring would mostly re-read unchanged features.

**Paths**

| Command | Use |
|---------|-----|
| `make score-local` | Dev / fast validation (local `model.joblib`) |
| `make score-vertex` | Production-like (Vertex BatchPredictionJob) |

**Doc:** [phase-4-batch.md](phase-4-batch.md)

---

## Optional live demo (≈2 minutes)

If time and GCP access allow:

```bash
make fairness MODEL=random_forest
make predict CUSTOMER_ID=7590-VHVEG   # or make predict ROW=0
```

Or show screenshots: Model Registry (us-west1), BQ `predictions`, one batch job.

**Do not live-run** `make deploy` or `make score-vertex` on the call unless
pre-tested — batch jobs have long cold starts.

---

## Closing (≈1 minute)

**What’s complete (Phases 0–4)**

- Data in BQ, train + fairness, Registry + CPR bundle, predictions table.

**What’s described but not built (Phase 5)**

- Cloud Scheduler → monthly `score-vertex`
- Prediction monitoring / drift
- Second model version + gated promotion

**Cost choices**

- Train locally; register without always-on endpoint (`REGISTER_ONLY=1`).
- Batch scoring, not 24/7 endpoint, for retention use case.

**Code organisation**

- Flat `src/` (~10 modules) — intentional for this scope; README documents
  where I’d split at scale (`cli/`, `training/`, `serving/`).

---

## Anticipated questions

| Question | Short answer |
|----------|--------------|
| Why local train, not Vertex Training? | Faster iteration, lower cost; cloud proof is Registry + batch |
| Why batch not endpoint? | Monthly retention campaigns; batch is cheaper |
| Why RF over XGBoost? | Better **test** F1/PR-AUC; deploy ≠ CV winner |
| Train/serve skew? | Same `preprocess.clean` + 18 baseline cols; full pipeline in joblib/CPR |
| Why threshold outside pipeline? | Change policy without retrain |
| Why monthly schedule? | Month-to-month contracts; features move on billing cycles |
| How monitor in prod? | Flag rate and score distribution by `run_id` in `predictions` |
| What’s `seed-scoring`? | Demo substitute for production ETL scoring population |

---

## Reviewer setup (share with the repo link)

Reviewers cloning the repo should:

```bash
uv sync
cp .env.example .env   # fill GCP values
make train             # artifacts not committed
make test              # 43 tests
```

**GCP region:** `us-west1` (Model Registry, batch jobs, Console dropdown).

**Key paths**

| Path | Content |
|------|---------|
| `README.md` | Architecture diagrams, phases, commands |
| `docs/phase-2-modeling.md` | Training & metrics |
| `docs/phase-3-deploy.md` | Registry + CPR |
| `docs/phase-4-batch.md` | Batch → BQ |
| `notebooks/01_eda.ipynb` | EDA that drove preprocessing |

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

Leave the last 5–10 minutes of the session for Q&A.
