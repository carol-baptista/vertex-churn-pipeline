# Train & evaluate

Turns the EDA decisions (`notebooks/01_eda.ipynb` -> Takeaways) into reproducible
training code. Produces two trained models and registry-ready artifacts.

## Files

| File | Purpose |
| --- | --- |
| `src/preprocess.py` | Clean the raw table + build the feature `ColumnTransformer`. |
| `src/train.py` | Train + compare Logistic Regression and XGBoost, evaluate, save artifacts. |
| `models/` | Output: fitted pipelines + metrics (git-ignored). |

## What the pipeline does

1. **Clean** (`preprocess.clean`)
   - `TotalCharges` STRING -> numeric. New customers (`tenure == 0`) -> `0`; any
     other blank is dropped and logged as a data error.
   - Collapse `"No internet service"` / `"No phone service"` -> `"No"` (lossless).
   - Cast nullable `Int64`/`boolean` columns so sklearn transformers behave.
2. **Split features / target / protected attribute** (`preprocess.make_dataset`)
   - Drop `customerID` and `gender` from features. `gender` is returned separately
     and used only for the fairness audit.
3. **Encode** (`preprocess.build_preprocessor`)
   - `StandardScaler` on numerics, `OneHotEncoder(handle_unknown="ignore")` on
     categoricals. `drop_first=True` for Logistic Regression, `False` for XGBoost.

## Models

| Model | Role | Imbalance | Tuning |
| --- | --- | --- | --- |
| Logistic Regression | interpretable **baseline** | `class_weight="balanced"` | none (defaults) |
| XGBoost | **candidate** | `scale_pos_weight = neg/pos` | small grid via stratified CV |

Both see the **same feature set**; only the representation differs (drop-first for
LogReg). Fair comparison = baseline at defaults vs. a lightly tuned candidate.

## Evaluation

- **Split:** 80/20, stratified on `Churn`, `random_state=42`.
- **Selection metric:** PR-AUC (average precision) - the right headline for an
  imbalanced "catch the churners" problem.
- **Reported:** PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix.
- **Threshold:** tuned to maximise F1 on **out-of-fold training** predictions, then
  applied to the test set (so test metrics aren't tuned on the test set).
- **Fairness:** test metrics sliced by `gender` to check for disparate impact via
  proxy features, even though `gender` is not a model input.

## How to run

```bash
# full run (reads from BigQuery via ADC)
python -m src.train

# quick smoke run on fewer rows
python -m src.train --sample 2000

# skip the XGBoost grid search (faster)
python -m src.train --no-tune
```

Outputs:

```
models/
  logreg/    model.joblib   metrics.json
  xgboost/   model.joblib   metrics.json
  summary.json     # comparison + winner by CV PR-AUC
```

## Reviewer checklist (is this deployable?)

- [ ] CV PR-AUC is stable across folds (low std) - not a lucky split.
- [ ] Test metrics are in line with CV (no large train/test gap = no overfit).
- [ ] Recall is high enough for the retention use case at an acceptable precision.
- [ ] Confusion matrix false-negative count is acceptable (missed churners).
- [ ] Fairness slices by gender are comparable (no disparate impact).
- [ ] The saved `model.joblib` reloads and scores a sample row.
- [ ] XGBoost beats the LogReg baseline by enough to justify its complexity.

## Deferred (future experiments)

- Engineered features (`avg_charge`, `charge_increased`) - measure lift vs. this baseline.
- Bucketing `tenure` / `MonthlyCharges` - CV ablation.
- SHAP explanations + Vertex AI Model Registry upload.
