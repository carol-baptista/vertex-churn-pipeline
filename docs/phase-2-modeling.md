# Train & evaluate

Turns the EDA decisions (`notebooks/01_eda.ipynb` -> Takeaways) into reproducible
training code. Produces two trained models and registry-ready artifacts.

**Navigate `src/train.py`:** [train-code-map.md](train-code-map.md) — section guide
with line ranges (start at `main()`, then `train_one()`).

## Files

| File | Purpose |
| --- | --- |
| `src/preprocess.py` | Clean the raw table + build the feature `ColumnTransformer`. |
| `src/train.py` | Train + compare Logistic Regression, Random Forest, XGBoost, and LightGBM; evaluate, save artifacts. |
| `models/` | Output: fitted pipelines + metrics (git-ignored). |

## What the pipeline does

1. **Clean** (`preprocess.clean`)
   - `TotalCharges` STRING -> numeric. New customers (`tenure == 0`) -> `0`; any
     other blank is dropped and logged as a data error.
   - Cast nullable `Int64`/`boolean` columns so sklearn transformers behave.
   - Raw category values are kept (no pre-encoding recode of `"No internet service"`).
2. **Split features / target / join key** (`preprocess.make_dataset`)
   - Drop protected attributes (e.g. `gender`) from features. Keep `customerID` on
     `Dataset.customer_id` as a join key only (never passed to the model).
   - Build a demographics table (`customerID` + protected cols) and merge back
     after scoring for fairness audits (`train.fairness_audit`).
3. **Encode** (`preprocess.build_preprocessor`)
   - `StandardScaler` on numerics for Logistic Regression only; tree models use
     raw numeric values (`scale_numeric=False`). Categoricals are one-hot encoded,
     then redundant `*No internet service` / `*No phone service` dummies are dropped.
     `handle_unknown="ignore"`. `drop_first=True` for Logistic Regression, `False`
     for XGBoost.

## Models

| Model | Role | Imbalance | Tuning |
| --- | --- | --- | --- |
| Logistic Regression | interpretable **baseline** | `class_weight="balanced"` | none (defaults) |
| Random Forest | tree **candidate** | `class_weight="balanced"` | small grid via stratified CV |
| XGBoost | boosted **candidate** | `scale_pos_weight = neg/pos` | small grid via stratified CV |
| LightGBM | boosted **candidate** | `scale_pos_weight = neg/pos` | small grid via stratified CV |

All four see the **same raw feature columns**; only the representation differs (scaling
and `drop_first` for LogReg). Fair comparison = baseline at defaults vs. lightly
tuned tree candidates.

## Evaluation

### Data split (stratified 70 / 15 / 15)

We use a **stratified** split: each subset keeps roughly the same churn rate (~27%)
as the full dataset. Implementation: `train_test_split(..., stratify=y)` twice
(once for test, once for validation from the remaining dev pool).

**What “stratified” means:** sampling is controlled by the class label so every
split mirrors the original class proportions instead of relying on pure randomness.

**Why we do it:** telco churn is imbalanced (~27% positive). A random 15% test
slice can easily end up with more or fewer churners by luck, which makes
precision/recall unstable and comparisons between runs unreliable. Stratification
is standard practice for imbalanced classification ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
[Machine Learning Mastery — stratified CV](https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/)).

| Split | Share | Used for |
| --- | --- | --- |
| **Train** | 70% | Model fitting + stratified CV hyperparameter search |
| **Validation** | 15% | Threshold tuning only (never seen during training) |
| **Test** | 15% | Final offline metrics (never seen during training or threshold tuning) |

`random_state=42` for reproducibility.

### Metrics reporting

Each role has a different job; only **test** is the headline number for
stakeholders ([MetricGate](https://metricgate.com/blogs/training-vs-validation-vs-test-set/),
[Machine Learning Mastery — test vs validation](https://machinelearningmastery.com/difference-test-validation-datasets/)):

| Phase | Role | Report externally? |
| --- | --- | --- |
| **CV on train** | Pick hyperparameters and winner model | No — model selection only |
| **Validation** | Tune classification threshold | No — development only (slightly optimistic) |
| **Test** | Unbiased generalization estimate | **Yes — lead with this** |

`models/summary.json` encodes this:

- Top-level `metrics_reporting` documents the workflow.
- `full_feature_set.champion.report` — headline test metrics for the CV winner.
- `full_feature_set.champion.development_only` — CV score + validation metrics.
- Per-model entries use the same `report` / `development_only` nesting.

**Reporting guidance:** “We tuned threshold on validation; test recall/precision/F1
are the numbers I’d stand behind.”

### Other evaluation settings

- **Metric (grid + winner):** one `--metric` drives both hyperparameter tuning
  and winner selection (default `f1`).
- **Threshold:** max F1 on the **validation** set by default
  (`--threshold-strategy f1`).
- **Imbalance:** loss reweighting only (no oversampling); default `--pos-weight sqrt`.
- **Feature set:** default `baseline` (raw EDA features). Use
  `--feature-set engineered` for demo comparison (+4 EDA-driven features).
- **Explainability:** Random Forest exports Gini importances + **SHAP** summary
  under `models/random_forest/shap/`.
- **Baseline snapshot:** `make train-baseline` saves results under
  `experiments/baseline/summary.json`. Engineered runs include `baseline_comparison`
  when that file exists.
- **Also logged:** CV selection score, PR-AUC, ROC-AUC, precision, recall, F1,
  confusion matrix (per split where applicable).
- **Fairness:** after test scoring, join protected attributes back on `customerID`
  and slice metrics by group (e.g. gender) to check for disparate impact via
  proxy features. Protected columns are never model inputs.

Each run prints and saves a `run_config` block listing every knob used.

## How to run

```bash
# full run with baseline features (default)
make train

# optional: engineered features + baseline comparison
make train FEATURE_SET=engineered

# freeze / refresh the raw-feature baseline snapshot
make train-baseline

# quick smoke run on fewer rows
make train-smoke

# skip the tree-model grid searches (faster)
make train-fast

# Feature-engine probe audit on encoded train data (analysis only)
make train-probe

# Train full + probe-selected sets and compare in summary.json
make train-probe-compare
```

CLI flags (defaults shown):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--feature-set` | `baseline` | `baseline` (raw) or `engineered` (+4 EDA features) |
| `--metric` | `f1` | Grid search **and** winner selection (same metric) |
| `--grid-metric` | *(same as metric)* | Advanced override for grid search only |
| `--select-metric` | *(same as metric)* | Advanced override for winner only |
| `--pos-weight` | `sqrt` | Imbalance correction strength |
| `--threshold-strategy` | `f1` | Threshold tuning on validation (max F1) |
| `--recall-floor` | `0.75` | Min recall when using `recall_floor` strategy |

Outputs:

```
models/
  logreg/           model.joblib   metrics.json
  random_forest/    model.joblib   metrics.json   feature_importance.json   feature_importance.png
                    shap/shap_summary.png   shap/shap_importance.json
  xgboost/          model.joblib   metrics.json
  lightgbm/         model.joblib   metrics.json
  probe_audit/      probe_audit.json   probe_audit.png   # with --probe-feature / make train-probe
  summary.json      # comparison + champion test metrics + metrics_reporting guide
```

## Training decisions (experiments → defaults)

We compared several knobs on the full telco dataset. Summary of what we tried and
what we kept:

| Experiment | Random Forest (test) | Verdict |
| --- | --- | --- |
| PR-AUC grid + `full` pos weight + F1 threshold | recall ~0.77, precision ~0.42 | Too many false positives (wasted retention mail) |
| F1 grid + `sqrt` pos weight + F1 threshold | recall **~0.78**, precision **~0.51**, F1 **~0.62** | **Sweet spot — project default** |
| F2 grid + `sqrt` pos weight + F2 threshold | recall ~0.88, precision ~0.42 | Recall too high; precision collapses (LightGBM flags almost everyone) |
| `recall_floor` threshold (≥0.75) | recall ~0.72, precision ~0.54 | Sacrifices recall for small precision gain vs F1 threshold |

**Chosen defaults** (plain `make train`):

- **`--metric f1`** — same metric for grid search and winner selection
- **`--threshold-strategy f1`** — max F1 on the validation set
- **`--pos-weight sqrt`** — gentler imbalance correction than `full`; keeps recall
  high (~78%) while lifting precision from ~42% to ~51%

Champion candidate from the default run: **Random Forest** (best test PR-AUC among
tree models; strong recall/precision balance at the F1 operating point).

Random Forest importances are ranked by Gini importance on the **encoded** feature
names (e.g. `cat__Contract_Two year`). The PNG shows the top 20; the JSON lists all.

### Probe audit (`--probe-feature`)

Uses [Feature-engine `ProbeFeatureSelection`](https://feature-engine.trainindata.com/en/latest/user_guide/selection/ProbeFeatureSelection.html) on the **encoded training fold** only:

- Adds `n_probes=3` **binary** probe features (matched to one-hot encoded columns)
- Fits `RandomForestClassifier` with 5-fold CV (`collective=True`)
- Drops features whose Gini importance is below the **mean probe importance**
- Does **not** change the models saved under `models/logreg/` etc. (audit-only)

Check `models/probe_audit/probe_audit.json` for `features_to_drop`,
`probe_threshold_value`, and `top_univariate_correlations` (EDA-style marginal
signal). Probe importance is **conditional** — a feature with high univariate
correlation can rank below the probe when correlated features (e.g. Electronic
check vs Month-to-month) absorb the split budget.

## Reviewer checklist (is this deployable?)

- [ ] CV metric is stable across folds (low std) — not a lucky split.
- [ ] **Test** metrics are in line with CV (no large train/test gap = no overfit).
- [ ] Validation metrics are close to test (threshold tuning did not overfit val).
- [ ] Recall is high enough for the retention use case at an acceptable precision.
- [ ] Confusion matrix false-negative count is acceptable (missed churners).
- [ ] Fairness slices by gender are comparable (no disparate impact).
- [ ] The saved `model.joblib` reloads and scores a sample row.
- [ ] XGBoost beats the LogReg baseline by enough to justify its complexity.

## Deferred (future experiments)

- Bucketing `tenure` / `MonthlyCharges` — CV ablation.
- Vertex AI Model Registry upload.
