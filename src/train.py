"""Train and compare churn models.

Trains two models on the *same* feature set:

- **Logistic Regression** - the interpretable baseline. Defaults + balanced class
  weights, deliberately *not* tuned (its job is to set the bar).
- **XGBoost** - the candidate. Class imbalance handled with ``scale_pos_weight``
  and lightly tuned with a small grid under stratified CV.

Model selection uses **PR-AUC** (average precision), which is the right headline
metric for an imbalanced problem where we care about catching churners. We also
report ROC-AUC, precision, recall, and F1, and we tune the decision threshold from
out-of-fold predictions instead of assuming 0.5.

Artifacts (one fitted ``Pipeline`` = preprocessing + model, plus a metrics JSON)
are written under ``models/`` so they are ready to push to the Vertex AI Model
Registry later.

Run it:

    python -m src.train                 # full run, reads from BigQuery
    python -m src.train --sample 2000   # quick smoke run on fewer rows
    python -m src.train --no-tune       # skip the XGBoost grid search
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, preprocess

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
SELECTION_METRIC = "average_precision"  # = PR-AUC
MODELS_DIR = config.REPO_ROOT / "models"


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Pick the probability threshold that maximises F1 on the given data.

    Threshold tuning is done on *out-of-fold training* predictions, never on the
    test set, so the reported test metrics stay honest.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have length len(thresholds) + 1; drop the trailing point.
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_idx])


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Compute the full metric set at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def fairness_audit(
    customer_ids: pd.Series,
    y_true: pd.Series,
    y_proba: np.ndarray,
    demographics: pd.DataFrame,
    threshold: float,
    protected_col: str = "gender",
) -> dict:
    """Join protected attributes back on ``customerID`` and slice metrics by group.

    Protected columns are excluded from model features during training. After
    scoring the test set, we merge demographics onto predictions via the join key
    to check for disparate impact via proxy features.
    """
    eval_df = pd.DataFrame(
        {
            preprocess.ID_COL: customer_ids.values,
            "y_true": y_true.values,
            "y_proba": y_proba,
        }
    ).merge(
        demographics[[preprocess.ID_COL, protected_col]],
        on=preprocess.ID_COL,
        how="left",
    )
    if eval_df[protected_col].isna().any():
        missing = int(eval_df[protected_col].isna().sum())
        raise ValueError(
            f"Fairness join failed for {missing} row(s) on {preprocess.ID_COL}."
        )
    return fairness_by_group(
        eval_df["y_true"],
        eval_df["y_proba"].to_numpy(),
        eval_df[protected_col],
        threshold,
    )


def fairness_by_group(
    y_true: pd.Series, y_proba: np.ndarray, group: pd.Series, threshold: float
) -> dict:
    """Slice recall/precision/selection-rate by a group column."""
    y_pred = (y_proba >= threshold).astype(int)
    out = {}
    for value in sorted(group.unique()):
        mask = (group == value).to_numpy()
        yt = y_true.to_numpy()[mask]
        yp = y_pred[mask]
        out[str(value)] = {
            "n": int(mask.sum()),
            "actual_churn_rate": float(yt.mean()),
            "predicted_positive_rate": float(yp.mean()),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
        }
    return out


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def build_logreg() -> Pipeline:
    """Baseline: Logistic Regression with balanced class weights (untuned)."""
    return Pipeline(
        steps=[
            ("prep", preprocess.build_preprocessor(drop_first=True, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost(scale_pos_weight: float) -> Pipeline:
    """Candidate: XGBoost with imbalance handled via scale_pos_weight."""
    return Pipeline(
        steps=[
            ("prep", preprocess.build_preprocessor(drop_first=False, scale_numeric=False)),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


XGB_PARAM_GRID = {
    "model__max_depth": [3, 4, 5],
    "model__n_estimators": [200, 400],
    "model__learning_rate": [0.05, 0.1],
}


# --------------------------------------------------------------------------- #
# Training routine for a single model
# --------------------------------------------------------------------------- #
def train_one(
    name: str,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    customer_id_test: pd.Series,
    demographics: pd.DataFrame,
    cv: StratifiedKFold,
    param_grid: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Fit (optionally tune) one model and evaluate it on the held-out test set."""
    logger.info("=== %s ===", name)

    # Optional hyper-parameter search (only the candidate uses this).
    if param_grid:
        logger.info("Grid search over %d combinations...", _grid_size(param_grid))
        search = GridSearchCV(
            pipe, param_grid, scoring=SELECTION_METRIC, cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        cv_score = float(search.best_score_)
        logger.info("Best params: %s", search.best_params_)
    else:
        cv_scores = cross_val_score(
            pipe, X_train, y_train, scoring=SELECTION_METRIC, cv=cv, n_jobs=-1
        )
        cv_score = float(cv_scores.mean())
        logger.info("CV PR-AUC: %.4f +/- %.4f", cv_scores.mean(), cv_scores.std())
        pipe.fit(X_train, y_train)

    # Choose the decision threshold from out-of-fold training predictions.
    oof_proba = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = best_threshold(y_train.to_numpy(), oof_proba)
    logger.info("Tuned threshold (max F1 on OOF train): %.3f", threshold)

    # Evaluate on the untouched test set.
    test_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), test_proba, threshold)
    metrics["cv_pr_auc"] = cv_score
    metrics["fairness_by_gender"] = fairness_audit(
        customer_id_test, y_test, test_proba, demographics, threshold
    )

    logger.info(
        "TEST  pr_auc=%.4f roc_auc=%.4f precision=%.3f recall=%.3f f1=%.3f",
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return pipe, metrics


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(name: str, pipe: Pipeline, metrics: dict) -> Path:
    """Write the fitted pipeline + metrics under models/<name>/."""
    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved %s artifacts to %s", name, out_dir)
    return out_dir


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(sample: int | None = None, tune: bool = True) -> dict:
    """Run training + model comparison and persist artifacts."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logger.info("Loading + cleaning data...")
    raw = None if sample is None else _load_sample(sample)
    if raw is None:
        from . import data

        raw = data.load_customers()
    cleaned = preprocess.clean(raw)
    ds = preprocess.dataset_from_cleaned(cleaned)
    demographics = preprocess.demographics_table(cleaned)
    logger.info("rows=%d  features=%d  churn_rate=%.3f", len(ds.X), ds.X.shape[1], ds.y.mean())

    # Stratified split; carry customer_id alongside for post-scoring fairness joins.
    X_train, X_test, y_train, y_test, _, customer_id_test = train_test_split(
        ds.X,
        ds.y,
        ds.customer_id,
        test_size=TEST_SIZE,
        stratify=ds.y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # scale_pos_weight = negatives / positives on the training set.
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    results: dict[str, dict] = {}

    logreg_pipe, logreg_metrics = train_one(
        "logreg",
        build_logreg(),
        X_train,
        y_train,
        X_test,
        y_test,
        customer_id_test,
        demographics,
        cv,
    )
    save_artifacts("logreg", logreg_pipe, logreg_metrics)
    results["logreg"] = logreg_metrics

    xgb_pipe, xgb_metrics = train_one(
        "xgboost",
        build_xgboost(scale_pos_weight),
        X_train,
        y_train,
        X_test,
        y_test,
        customer_id_test,
        demographics,
        cv,
        param_grid=XGB_PARAM_GRID if tune else None,
    )
    save_artifacts("xgboost", xgb_pipe, xgb_metrics)
    results["xgboost"] = xgb_metrics

    # Select the winner by CV PR-AUC (robust, computed without the test set).
    best = max(results, key=lambda k: results[k]["cv_pr_auc"])
    summary = {
        "selection_metric": "cv_pr_auc",
        "winner": best,
        "models": {
            k: {
                "cv_pr_auc": v["cv_pr_auc"],
                "test_pr_auc": v["pr_auc"],
                "test_roc_auc": v["roc_auc"],
                "test_recall": v["recall"],
                "test_precision": v["precision"],
                "test_f1": v["f1"],
                "threshold": v["threshold"],
            }
            for k, v in results.items()
        },
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Winner by CV PR-AUC: %s", best)
    print("\n" + json.dumps(summary, indent=2))
    return summary


def _load_sample(n: int) -> pd.DataFrame:
    from . import data

    return data.load_customers(limit=n)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare churn models.")
    parser.add_argument(
        "--sample", type=int, default=None, help="limit rows for a quick smoke run"
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="skip the XGBoost grid search"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(sample=args.sample, tune=not args.no_tune)
