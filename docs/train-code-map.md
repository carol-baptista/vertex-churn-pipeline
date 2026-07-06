# `train.py` code map

[`src/train.py`](../src/train.py) is ~1,500 lines. **Don't read it top to bottom.**
Follow the flow below — each section has line ranges and the functions that matter.

**Entry point:** `main()` (~1201) ← start here, then drill into `train_one()`.

```text
main()
  → load + clean + split
  → train_model_suite() × 1–2 (full features, optional probe-selected)
      → train_one() × 4 models
      → save_artifacts() + SHAP/importance for RF
  → build summary.json + optional baseline comparison
```

---

## Suggested reading order

| Order | Jump to | Focus |
|-------|---------|-------|
| 1 | `main()` ~1230–1264 | Load BQ data, clean, **stratified 70/15/15** split |
| 2 | `train_one()` ~806–866 | One model: fit → threshold on **val** → test metrics + fairness |
| 3 | `best_threshold()` ~305 | How the 0.441 cutoff is chosen |
| 4 | `fairness_audit()` ~363 | Join gender back on `customerID` after test scoring |
| 5 | `save_artifacts()` ~1129 | What lands in `models/random_forest/` |

Lower priority on first pass: probe audit, SHAP internals, grid definitions, CLI parsing.

---

## Section 1 — Config and run metadata (~94–300)

| Function | What it does |
|----------|----------------|
| `resolve_sklearn_scorer()` | Maps CLI metric (`f1`, `pr_auc`, …) to sklearn scorer |
| `build_run_config()` | Records every knob (metric, split, threshold strategy) for `summary.json` |
| `save_baseline_snapshot()` | Freezes baseline run under `experiments/baseline/` |
| `compare_to_baseline()` | Engineered vs baseline test metrics |
| `split_train_val_test()` | **Stratified** 70% train / 15% val / 15% test |
| `resolve_training_metrics()` | Aligns grid metric and winner-selection metric |
| `log_run_config()` | Prints config at run start |

**Note:** `split_train_val_test` is the guardrail — test is never used for training or threshold tuning.

---

## Section 2 — Metrics and fairness (~302–419)

| Function | What it does |
|----------|----------------|
| `best_threshold()` | Pick probability cutoff on **validation** (max F1 by default) |
| `evaluate()` | PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix at a threshold |
| `fairness_audit()` | Merge demographics on `customerID`, slice test predictions by group |
| `fairness_by_group()` | Per-group recall, precision, flag rate, churn rate |

**Note:** threshold is tuned on validation; test metrics in `metrics.json` are the headline numbers.

---

## Section 3 — Model builders (~421–780)

| Function | What it does |
|----------|----------------|
| `pos_weight_value()` | Imbalance knob: `sqrt` (default), `full`, or `none` |
| `build_logreg()` | Baseline pipeline — balanced weights, not tuned |
| `build_random_forest()` | RF pipeline + optional `GridSearchCV` param grid |
| `build_xgboost()` / `build_lightgbm()` | Boosted trees + small grids |
| `build_model_specs()` | List of `(name, pipeline, param_grid)` for all four models |
| `SelectEncodedFeatures` | Optional column filter for probe-selected training |
| `run_probe_audit()` | Feature-engine probe — **analysis only**, not default path |
| `probe_kept_features()` | Features that beat probe noise threshold |
| `_metrics_block()` / `_champion_block()` | Shape `summary.json` report vs development metrics |
| `_winner()` | CV winner among the four models |

**Note:** four models share the same features and split; LogReg is the interpretable bar, trees are tuned candidates.

---

## Section 4 — Core training loop (~782–884)

### `train_one()` — the heart of the file

One model, end to end:

```text
1. Optional GridSearchCV on train (tree models)
2. cross_val_score on train → cv_score (winner selection)
3. predict_proba on validation → best_threshold()
4. evaluate() on validation + test
5. fairness_audit() on test set
6. return (fitted_pipeline, metrics_dict)
```

**Outputs per model:** `threshold`, `cv_score`, `validation`, `test`,
`fairness_by_gender` → written to `models/<name>/metrics.json`.

---

## Section 5 — Explainability and probe (~887–1124)

| Function | What it does |
|----------|----------------|
| `tree_feature_importance()` | Gini importances from RF (encoded feature names) |
| `encode_training_matrix()` | Matrix for probe / correlation analysis |
| `run_probe_audit()` | Probe feature selection audit |
| `save_probe_audit()` | JSON + plot under `models/probe_audit/` |
| `save_rf_feature_importance()` | PNG + JSON for RF |
| `save_shap_analysis()` | SHAP summary for RF (optional, `--no-shap` to skip) |

**Note:** SHAP artifacts live under `models/random_forest/shap/` — optional path, not required for the core pipeline.

---

## Section 6 — Persistence (~1126–1195)

| Function | What it does |
|----------|----------------|
| `save_artifacts()` | `model.joblib` + `metrics.json` per model directory |
| `train_model_suite()` | Loop: `train_one` → save for logreg, RF, XGB, LGBM |

---

## Section 7 — Orchestration: `main()` (~1201–1427)

High-level steps inside `main()`:

| Step | Lines (approx) | What happens |
|------|----------------|--------------|
| Resolve metrics / feature set | 1221–1228 | CLI → sklearn scorers |
| Load data | 1230–1238 | BQ via `data.load_customers()`, `preprocess.clean()` |
| Build dataset | 1237–1238 | Features `X`, target `y`, `customer_id`, demographics |
| Split | 1247–1264 | `split_train_val_test()` |
| Imbalance weight | 1268–1271 | `pos_weight_value()` |
| Log config | 1273–1287 | `build_run_config()` |
| Optional probe | 1296–1306 | `--probe-feature` / `--probe-train` only |
| Train all models | 1308–1328 | `train_model_suite("all features", …)` |
| Optional probe train | 1330–1357 | Second suite under `models/probe_selected/` |
| Summary | 1359–1413 | `summary.json` with champion, metrics_reporting, baseline compare |
| Baseline snapshot | 1404–1405 | `make train-baseline` |

**Final artifact:** [`models/summary.json`](../models/summary.json) — comparison
across models + `metrics_reporting` guide.

---

## Section 8 — CLI (~1430–1541)

| Function | What it does |
|----------|----------------|
| `_load_sample()` | `--sample N` for smoke runs |
| `_parse_args()` | All CLI flags (`--metric`, `--feature-set`, `--no-tune`, …) |

Invoked via `make train` or `python -m src.train`.

---

## Related files (preprocessing lives elsewhere)

| File | Role in training |
|------|------------------|
| [`src/preprocess.py`](../src/preprocess.py) | `clean()`, `dataset_from_cleaned()`, `build_preprocessor()` |
| [`src/data.py`](../src/data.py) | `load_customers()` from BigQuery |
| [`docs/phase-2-modeling.md`](phase-2-modeling.md) | Defaults, experiments, reviewer checklist |

---

## Common questions about this file

| Question | Where to point |
|----------|----------------|
| How is the split done? | `split_train_val_test()` |
| Where is threshold tuned? | `best_threshold()` on **validation** in `train_one()` |
| Where is fairness? | End of `train_one()` → `fairness_audit()` |
| Why is the file so long? | Four models + grids + probe + SHAP + summary assembly in one orchestrator; at scale I'd split `training/` vs `cli/` |
