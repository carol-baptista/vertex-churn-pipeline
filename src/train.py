"""Train and compare churn models.

Trains four models on the *same* feature set:

- **Logistic Regression** - the interpretable baseline. Defaults + balanced class
  weights, deliberately *not* tuned (its job is to set the bar).
- **Random Forest** - tree ensemble candidate; exports Gini feature importances.
- **XGBoost** - gradient-boosted candidate, lightly tuned with a small grid.
- **LightGBM** - gradient-boosted candidate, lightly tuned with a small grid.

Tree models use ``scale_pos_weight`` / ``class_weight`` for imbalance. One metric
(default **F2**) drives both grid search and winner selection. Threshold tuning
uses a separate strategy (default: recall floor at 0.75).

Run it:

    python -m src.train                 # full run with project defaults
    python -m src.train --sample 2000   # quick smoke run on fewer rows
    python -m src.train --no-tune       # skip tree-model grid searches
    python -m src.train --probe-feature # Feature-engine probe audit (analysis only)
    python -m src.train --probe-train   # compare full vs probe-selected features
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from feature_engine.selection import ProbeFeatureSelection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
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
MODELS_DIR = config.REPO_ROOT / "models"

# Project defaults (also used by Makefile `make train`).
DEFAULT_METRIC = "f2"
DEFAULT_POS_WEIGHT_MODE = "sqrt"
DEFAULT_THRESHOLD_STRATEGY = "recall_floor"
DEFAULT_RECALL_FLOOR = 0.75

METRIC_CHOICES = ("pr_auc", "f1", "f2", "mcc")
THRESHOLD_STRATEGIES = ("f1", "f2", "recall_floor")
POS_WEIGHT_MODES = ("full", "sqrt", "none")

# Legacy alias kept for callers that import SELECTION_METRIC.
SELECTION_METRIC = "average_precision"


def resolve_sklearn_scorer(metric: str) -> str:
    """Map CLI metric names to sklearn ``scoring`` strings."""
    mapping = {
        "pr_auc": "average_precision",
        "average_precision": "average_precision",
        "f1": "f1",
        "f2": "f2",
        "mcc": "matthews_corrcoef",
    }
    key = metric.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {tuple(mapping)}"
        )
    return mapping[key]


def build_run_config(
    *,
    sample: int | None,
    tune: bool,
    probe_audit: bool,
    probe_train: bool,
    metric: str,
    grid_metric: str,
    select_metric: str,
    pos_weight_mode: str,
    pos_weight: float,
    threshold_strategy: str,
    recall_floor: float,
) -> dict:
    """Snapshot of the knobs used for one training run."""
    return {
        "sample": sample,
        "tune": tune,
        "probe_audit": probe_audit,
        "probe_train": probe_train,
        "metric": metric,
        "grid_metric": grid_metric,
        "grid_scoring": resolve_sklearn_scorer(grid_metric),
        "select_metric": select_metric,
        "select_scoring": resolve_sklearn_scorer(select_metric),
        "metrics_aligned": grid_metric == select_metric,
        "pos_weight_mode": pos_weight_mode,
        "pos_weight_value": round(pos_weight, 4),
        "threshold_strategy": threshold_strategy,
        "recall_floor": recall_floor,
    }


def resolve_training_metrics(
    metric: str,
    grid_metric: str | None,
    select_metric: str | None,
) -> tuple[str, str]:
    """Resolve grid and selection metrics; default both to ``metric``."""
    grid = grid_metric or metric
    select = select_metric or metric
    if grid != select:
        logger.warning(
            "Grid metric (%s) differs from selection metric (%s). "
            "Hyperparameter tuning and winner selection use different criteria.",
            grid,
            select,
        )
    return grid, select


def log_run_config(run_config: dict) -> None:
    """Print and log the active run configuration."""
    payload = json.dumps(run_config, indent=2)
    logger.info("Run configuration:\n%s", payload)
    print("\n=== Run configuration ===")
    print(payload)
    print()


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
    beta: float = 2.0,
) -> float:
    """Pick a probability threshold on out-of-fold training predictions.

    Strategies:
    - ``f1``: maximise F1 (balanced precision/recall).
    - ``f2``: maximise F-beta with beta=2 (recall-weighted, still penalises spam).
    - ``recall_floor``: require recall >= ``recall_floor``, then maximise precision.
    """
    if strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(
            f"threshold strategy must be one of {THRESHOLD_STRATEGIES}, got {strategy!r}"
        )

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5

    p = precision[:-1]
    r = recall[:-1]

    if strategy == "recall_floor":
        eligible = np.where(r >= recall_floor)[0]
        if len(eligible) == 0:
            best_idx = int(np.nanargmax(r))
        else:
            best_idx = int(eligible[np.nanargmax(p[eligible])])
    elif strategy == "f2":
        fbeta = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-12)
        best_idx = int(np.nanargmax(fbeta))
    else:
        f1 = 2 * p * r / (p + r + 1e-12)
        best_idx = int(np.nanargmax(f1))

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
PROBE_N_PROBES = 3
# Binary probes match one-hot encoded columns; Gaussian probes inflate the threshold
# on 0/1 dummies (see Feature-engine docs on probe distributions).
PROBE_DISTRIBUTION = "binary"
PROBE_THRESHOLD = "mean"


def pos_weight_value(mode: str, neg: int, pos: int) -> float:
    """Positive-class weight for a given imbalance-correction ``mode``.

    All models share one scalar so the knob is comparable across them:

    - ``full`` -> neg/pos, the full rebalance to 50/50 (most recall-biased).
    - ``sqrt`` -> sqrt(neg/pos), a gentler correction that recovers precision.
    - ``none`` -> 1.0, natural class prior (probabilities stay calibrated).

    For ``class_weight`` models this becomes ``{0: 1.0, 1: pos_weight}``; for
    gradient boosters it becomes ``scale_pos_weight``. ``{0: 1, 1: neg/pos}`` is
    proportionally identical to sklearn's ``class_weight="balanced"``.
    """
    if mode not in POS_WEIGHT_MODES:
        raise ValueError(f"pos_weight mode must be one of {POS_WEIGHT_MODES}, got {mode!r}")
    ratio = neg / max(pos, 1)
    if mode == "full":
        return float(ratio)
    if mode == "sqrt":
        return float(np.sqrt(ratio))
    return 1.0


def _class_weight(pos_weight: float) -> dict[int, float]:
    return {0: 1.0, 1: float(pos_weight)}


def _random_forest_estimator(
    class_weight: dict[int, float] | str = "balanced",
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_logreg(pos_weight: float, *, slim: bool = False) -> Pipeline:
    """Baseline: Logistic Regression with weighted class loss (untuned)."""
    return Pipeline(
        steps=[
            (
                "prep",
                preprocess.build_preprocessor(
                    drop_first=False if slim else True,
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=_class_weight(pos_weight),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost(pos_weight: float) -> Pipeline:
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
                    scale_pos_weight=pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_random_forest(pos_weight: float) -> Pipeline:
    """Candidate: Random Forest with weighted class loss."""
    return Pipeline(
        steps=[
            ("prep", preprocess.build_preprocessor(drop_first=False, scale_numeric=False)),
            ("model", _random_forest_estimator(_class_weight(pos_weight))),
        ]
    )


def build_lightgbm(pos_weight: float) -> Pipeline:
    """Candidate: LightGBM with imbalance handled via scale_pos_weight."""
    return Pipeline(
        steps=[
            ("prep", preprocess.build_preprocessor(drop_first=False, scale_numeric=False)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


class SelectEncodedFeatures(BaseEstimator, TransformerMixin):
    """Keep a subset of columns from an encoded preprocessor matrix."""

    def __init__(self, feature_names: list[str], prep_feature_names: list[str]):
        self.feature_names = feature_names
        self.prep_feature_names = prep_feature_names

    def fit(self, X, y=None):
        name_to_idx = {name: idx for idx, name in enumerate(self.prep_feature_names)}
        missing = [name for name in self.feature_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Unknown encoded features for selection: {missing[:5]}")
        self.indices_ = np.array([name_to_idx[name] for name in self.feature_names])
        self.feature_names_out_ = np.asarray(self.feature_names, dtype=object)
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def _with_feature_selector(
    pipe: Pipeline, feature_names: list[str], prep_feature_names: list[str]
) -> Pipeline:
    """Insert encoded-feature selection after the prep step."""
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    selector = SelectEncodedFeatures(
        feature_names=feature_names,
        prep_feature_names=prep_feature_names,
    )
    return Pipeline([("prep", prep), ("select", selector), ("model", model)])


def build_model_specs(
    pos_weight: float,
    *,
    tune: bool,
    selected_features: list[str] | None = None,
    prep_feature_names: list[str] | None = None,
) -> list[tuple[str, Pipeline, dict | None]]:
    """Return (name, pipeline, param_grid) tuples for one training pass."""
    slim = selected_features is not None
    if slim and not prep_feature_names:
        raise ValueError("prep_feature_names required when selected_features is set")
    wrap = (
        (lambda pipe: _with_feature_selector(pipe, selected_features, prep_feature_names))
        if slim
        else (lambda pipe: pipe)
    )
    return [
        ("logreg", wrap(build_logreg(pos_weight, slim=slim)), None),
        (
            "random_forest",
            wrap(build_random_forest(pos_weight)),
            RF_PARAM_GRID if tune else None,
        ),
        (
            "xgboost",
            wrap(build_xgboost(pos_weight)),
            XGB_PARAM_GRID if tune else None,
        ),
        (
            "lightgbm",
            wrap(build_lightgbm(pos_weight)),
            LGBM_PARAM_GRID if tune else None,
        ),
    ]


def probe_kept_features(audit: dict) -> list[str]:
    """Encoded feature names that survived the probe threshold."""
    dropped = set(audit["features_to_drop"])
    return [
        row["feature"]
        for row in audit["feature_importances"]
        if row["feature"] not in dropped
    ]


def _metrics_block(results: dict[str, dict]) -> dict:
    block: dict[str, dict] = {}
    for k, v in results.items():
        entry = {
            "cv_score": v["cv_score"],
            "test_pr_auc": v["pr_auc"],
            "test_roc_auc": v["roc_auc"],
            "test_recall": v["recall"],
            "test_precision": v["precision"],
            "test_f1": v["f1"],
            "threshold": v["threshold"],
        }
        if "cv_grid_score" in v:
            entry["cv_grid_score"] = v["cv_grid_score"]
        block[k] = entry
    return block


def _winner(results: dict[str, dict]) -> str:
    return max(results, key=lambda k: results[k]["cv_score"])


def _compare_feature_sets(
    full_results: dict[str, dict], slim_results: dict[str, dict]
) -> dict:
    comparison: dict[str, dict] = {}
    for name in full_results:
        full = full_results[name]
        slim = slim_results[name]
        comparison[name] = {
            "delta_cv_score": round(slim["cv_score"] - full["cv_score"], 4),
            "delta_test_pr_auc": round(slim["pr_auc"] - full["pr_auc"], 4),
            "delta_test_recall": round(slim["recall"] - full["recall"], 4),
            "delta_test_precision": round(slim["precision"] - full["precision"], 4),
            "full_test_pr_auc": full["pr_auc"],
            "slim_test_pr_auc": slim["pr_auc"],
        }
    return comparison


XGB_PARAM_GRID = {
    "model__max_depth": [3, 4, 5],
    "model__n_estimators": [200, 400],
    "model__learning_rate": [0.05, 0.1],
}

RF_PARAM_GRID = {
    "model__max_depth": [8, 12, 16, None],
    "model__n_estimators": [300, 500],
    "model__min_samples_leaf": [1, 5],
    "model__max_features": ["sqrt", "log2"],
}

LGBM_PARAM_GRID = {
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
    *,
    grid_scoring: str,
    select_scoring: str,
    select_metric: str,
    threshold_strategy: str,
    recall_floor: float,
) -> tuple[Pipeline, dict]:
    """Fit (optionally tune) one model and evaluate it on the held-out test set."""
    logger.info("=== %s ===", name)

    grid_cv_score: float | None = None
    if param_grid:
        logger.info(
            "Grid search over %d combinations (scoring=%s)...",
            _grid_size(param_grid),
            grid_scoring,
        )
        search = GridSearchCV(
            pipe, param_grid, scoring=grid_scoring, cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        grid_cv_score = float(search.best_score_)
        logger.info("Best params: %s", search.best_params_)
        logger.info("Grid CV score (%s): %.4f", grid_scoring, grid_cv_score)
    else:
        pipe.fit(X_train, y_train)

    select_scores = cross_val_score(
        pipe, X_train, y_train, scoring=select_scoring, cv=cv, n_jobs=-1
    )
    select_cv_score = float(select_scores.mean())
    logger.info(
        "CV %s: %.4f +/- %.4f",
        select_metric,
        select_scores.mean(),
        select_scores.std(),
    )

    oof_proba = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = best_threshold(
        y_train.to_numpy(),
        oof_proba,
        strategy=threshold_strategy,
        recall_floor=recall_floor,
    )
    logger.info(
        "Tuned threshold (%s, recall_floor=%.2f): %.3f",
        threshold_strategy,
        recall_floor,
        threshold,
    )

    test_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), test_proba, threshold)
    metrics["cv_score"] = select_cv_score
    metrics["select_metric"] = select_metric
    if grid_cv_score is not None:
        metrics["cv_grid_score"] = grid_cv_score
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


def tree_feature_importance(pipe: Pipeline, top_n: int | None = None) -> list[dict]:
    """Rank feature importances from a fitted tree-model pipeline."""
    if "select" in pipe.named_steps:
        names = pipe.named_steps["select"].get_feature_names_out()
    else:
        names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    importances = model.feature_importances_
    ranked = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return [{"feature": str(feature), "importance": float(score)} for feature, score in ranked]


def encode_training_matrix(
    X_train: pd.DataFrame, *, drop_first: bool = False, scale_numeric: bool = False
) -> tuple[pd.DataFrame, object]:
    """Fit the tree preprocessor on train data and return an encoded DataFrame."""
    prep = preprocess.build_preprocessor(drop_first=drop_first, scale_numeric=scale_numeric)
    matrix = prep.fit_transform(X_train)
    columns = prep.get_feature_names_out()
    return pd.DataFrame(matrix, columns=columns, index=X_train.index), prep


def _encoded_univariate_correlations(
    X_encoded: pd.DataFrame, y_train: pd.Series, top_n: int = 15
) -> list[dict]:
    """Point-biserial correlation of each encoded column with churn (EDA-style check)."""
    y = y_train.to_numpy()
    corrs: list[tuple[str, float]] = []
    for col in X_encoded.columns:
        if "probe" in col:
            continue
        x = X_encoded[col].to_numpy()
        if np.std(x) == 0:
            continue
        corrs.append((col, float(np.corrcoef(x, y)[0, 1])))
    corrs.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {"feature": name, "churn_correlation": score}
        for name, score in corrs[:top_n]
    ]


def run_probe_audit(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[ProbeFeatureSelection, dict]:
    """Run Feature-engine probe selection on encoded training features.

    Compares Gini importances from a Random Forest (fit with CV) against synthetic
    probe features. Features weaker than the probe threshold would be dropped by
    ``fit_transform``; this audit reports them without changing the training pipeline.

    Uses **binary** probes because the matrix is mostly one-hot encoded 0/1 columns.
    Univariate churn correlations are included for comparison with EDA — they measure
    marginal signal, while probe/RF importance is conditional on all other features.
    """
    X_encoded, _ = encode_training_matrix(X_train)

    selector = ProbeFeatureSelection(
        estimator=_random_forest_estimator(),
        collective=True,
        n_probes=PROBE_N_PROBES,
        distribution=PROBE_DISTRIBUTION,
        threshold=PROBE_THRESHOLD,
        cv=N_SPLITS,
        random_state=RANDOM_STATE,
    )
    selector.fit(X_encoded, y_train)

    importances = selector.feature_importances_.sort_values(ascending=False)
    probe_cols = [name for name in importances.index if "probe" in name]
    probe_threshold = float(importances[probe_cols].mean())
    ranked = [
        {"feature": str(name), "importance": float(score)}
        for name, score in importances.items()
        if "probe" not in name
    ]

    audit = {
        "method": "feature_engine.ProbeFeatureSelection",
        "estimator": "RandomForestClassifier",
        "collective": True,
        "n_probes": PROBE_N_PROBES,
        "distribution": PROBE_DISTRIBUTION,
        "threshold": PROBE_THRESHOLD,
        "probe_columns": probe_cols,
        "probe_threshold_value": probe_threshold,
        "n_features_in": int(len(importances) - len(probe_cols)),
        "n_selected": int(len(importances) - len(probe_cols) - len(selector.features_to_drop_)),
        "n_dropped": int(len(selector.features_to_drop_)),
        "features_to_drop": list(selector.features_to_drop_),
        "encoded_feature_names": list(X_encoded.columns),
        "feature_importances": ranked,
        "top_univariate_correlations": _encoded_univariate_correlations(X_encoded, y_train),
        "notes": (
            "Probe importance is conditional (RF with all features). A feature can "
            "have high EDA correlation but low probe rank when signal is absorbed by "
            "correlated columns (e.g. Electronic check overlaps Month-to-month)."
        ),
    }
    return selector, audit


def save_probe_audit(
    selector: ProbeFeatureSelection, audit: dict, out_dir: Path, top_n: int = 25
) -> None:
    """Persist probe-audit JSON and an importance plot with the probe threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_audit.json").write_text(json.dumps(audit, indent=2))

    importances = selector.feature_importances_.sort_values(ascending=True)
    probe_cols = audit["probe_columns"]
    threshold = audit["probe_threshold_value"]
    dropped = set(audit["features_to_drop"])

    top = importances.tail(top_n)
    colors = []
    for name in top.index:
        if name in probe_cols:
            colors.append("#9b59b6")
        elif name in dropped:
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    ax.barh(top.index.astype(str), top.values, color=colors)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="probe threshold")
    ax.set_xlabel("Importance (Gini, CV-averaged)")
    ax.set_title("Probe audit — features below dashed line are weaker than noise")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "probe_audit.png", dpi=120)
    plt.close(fig)
    logger.info(
        "Probe audit: %d features dropped (threshold=%.6f) -> %s",
        audit["n_dropped"],
        threshold,
        out_dir,
    )


def save_rf_feature_importance(pipe: Pipeline, out_dir: Path, top_n: int = 20) -> None:
    """Persist Random Forest Gini importances as JSON and a bar chart."""
    ranked = tree_feature_importance(pipe)
    (out_dir / "feature_importance.json").write_text(json.dumps(ranked, indent=2))

    top = ranked[:top_n]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    features = [row["feature"] for row in reversed(top)]
    values = [row["importance"] for row in reversed(top)]
    ax.barh(features, values)
    ax.set_xlabel("Importance (Gini)")
    ax.set_title(f"Random Forest — top {top_n} feature importances")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close(fig)
    logger.info("Saved Random Forest feature importances to %s", out_dir)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(name: str, pipe: Pipeline, metrics: dict, *, subdir: str = "") -> Path:
    """Write the fitted pipeline + metrics under models/<subdir><name>/."""
    out_dir = MODELS_DIR / subdir / name if subdir else MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved %s artifacts to %s", name, out_dir)
    return out_dir


def train_model_suite(
    label: str,
    model_specs: list[tuple[str, Pipeline, dict | None]],
    *,
    subdir: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    customer_id_test: pd.Series,
    demographics: pd.DataFrame,
    cv: StratifiedKFold,
    grid_scoring: str,
    select_scoring: str,
    select_metric: str,
    threshold_strategy: str,
    recall_floor: float,
) -> dict[str, dict]:
    """Train and persist all models for one feature set."""
    logger.info("=== Feature set: %s (%s) ===", label, subdir or "full")
    results: dict[str, dict] = {}
    for name, pipe, param_grid in model_specs:
        fitted_pipe, metrics = train_one(
            name,
            pipe,
            X_train,
            y_train,
            X_test,
            y_test,
            customer_id_test,
            demographics,
            cv,
            param_grid=param_grid,
            grid_scoring=grid_scoring,
            select_scoring=select_scoring,
            select_metric=select_metric,
            threshold_strategy=threshold_strategy,
            recall_floor=recall_floor,
        )
        if name == "random_forest":
            metrics["feature_importance_top20"] = tree_feature_importance(
                fitted_pipe, top_n=20
            )
        out_dir = save_artifacts(name, fitted_pipe, metrics, subdir=subdir)
        if name == "random_forest":
            save_rf_feature_importance(fitted_pipe, out_dir)
        results[name] = metrics
    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(
    sample: int | None = None,
    tune: bool = True,
    probe_audit: bool = False,
    probe_train: bool = False,
    metric: str = DEFAULT_METRIC,
    grid_metric: str | None = None,
    select_metric: str | None = None,
    pos_weight_mode: str = DEFAULT_POS_WEIGHT_MODE,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> dict:
    """Run training + model comparison and persist artifacts."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    grid_metric, select_metric = resolve_training_metrics(metric, grid_metric, select_metric)
    grid_scoring = resolve_sklearn_scorer(grid_metric)
    select_scoring = resolve_sklearn_scorer(select_metric)

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

    # Positive-class weight for imbalance correction (loss reweighting, no resampling).
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = pos_weight_value(pos_weight_mode, neg, pos)

    run_config = build_run_config(
        sample=sample,
        tune=tune,
        probe_audit=probe_audit,
        probe_train=probe_train,
        metric=metric,
        grid_metric=grid_metric,
        select_metric=select_metric,
        pos_weight_mode=pos_weight_mode,
        pos_weight=pos_weight,
        threshold_strategy=threshold_strategy,
        recall_floor=recall_floor,
    )
    log_run_config(run_config)

    logger.info(
        "Imbalance handling: pos_weight_mode=%s -> pos_weight=%.4f (neg/pos=%.4f)",
        pos_weight_mode,
        pos_weight,
        neg / max(pos, 1),
    )

    probe_summary: dict | None = None
    kept_features: list[str] | None = None
    if probe_audit or probe_train:
        selector, probe_summary = run_probe_audit(X_train, y_train)
        save_probe_audit(selector, probe_summary, MODELS_DIR / "probe_audit")
        kept_features = probe_kept_features(probe_summary)
        logger.info(
            "Probe selection kept %d / %d encoded features",
            len(kept_features),
            probe_summary["n_features_in"],
        )

    full_results = train_model_suite(
        "all features",
        build_model_specs(pos_weight, tune=tune),
        subdir="",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        customer_id_test=customer_id_test,
        demographics=demographics,
        cv=cv,
        grid_scoring=grid_scoring,
        select_scoring=select_scoring,
        select_metric=select_metric,
        threshold_strategy=threshold_strategy,
        recall_floor=recall_floor,
    )

    slim_results: dict[str, dict] | None = None
    if probe_train and kept_features:
        slim_results = train_model_suite(
            "probe-selected",
            build_model_specs(
                pos_weight,
                tune=tune,
                selected_features=kept_features,
                prep_feature_names=probe_summary["encoded_feature_names"],
            ),
            subdir="probe_selected",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            customer_id_test=customer_id_test,
            demographics=demographics,
            cv=cv,
            grid_scoring=grid_scoring,
            select_scoring=select_scoring,
            select_metric=select_metric,
            threshold_strategy=threshold_strategy,
            recall_floor=recall_floor,
        )

    summary: dict = {
        "run_config": run_config,
        "metric": metric,
        "full_feature_set": {
            "n_raw_features": int(ds.X.shape[1]),
            "winner": _winner(full_results),
            "models": _metrics_block(full_results),
        },
        "probe_audit": probe_audit or probe_train,
        "probe_train": probe_train,
    }
    if probe_summary is not None:
        summary["probe_audit_summary"] = {
            "n_dropped": probe_summary["n_dropped"],
            "n_selected": probe_summary["n_selected"],
            "probe_threshold_value": probe_summary["probe_threshold_value"],
            "selected_features": kept_features,
            "features_to_drop": probe_summary["features_to_drop"],
        }
    if slim_results is not None and kept_features is not None:
        summary["probe_selected_feature_set"] = {
            "n_encoded_features": len(kept_features),
            "winner": _winner(slim_results),
            "models": _metrics_block(slim_results),
        }
        summary["feature_set_comparison"] = _compare_feature_sets(
            full_results, slim_results
        )
        best_full = max(full_results.values(), key=lambda m: m["pr_auc"])["pr_auc"]
        best_slim = max(slim_results.values(), key=lambda m: m["pr_auc"])["pr_auc"]
        summary["recommendation"] = (
            "probe_selected"
            if best_slim >= best_full
            else "full"
        )

    # Back-compat keys used by earlier runs.
    summary["winner"] = summary["full_feature_set"]["winner"]
    summary["models"] = summary["full_feature_set"]["models"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        "Full-feature winner by CV %s: %s",
        select_metric,
        summary["winner"],
    )
    if slim_results is not None:
        logger.info(
            "Probe-selected winner: %s (recommendation=%s)",
            summary["probe_selected_feature_set"]["winner"],
            summary.get("recommendation"),
        )
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
        "--no-tune", action="store_true", help="skip tree-model grid searches"
    )
    parser.add_argument(
        "--probe-feature",
        action="store_true",
        help="run Feature-engine ProbeFeatureSelection audit on encoded train data",
    )
    parser.add_argument(
        "--probe-train",
        action="store_true",
        help="train on probe-selected features and compare against the full feature set",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=METRIC_CHOICES,
        help=(
            "metric for grid search AND winner selection "
            f"(default: {DEFAULT_METRIC})"
        ),
    )
    parser.add_argument(
        "--grid-metric",
        default=None,
        choices=METRIC_CHOICES,
        help="override grid-search metric only (default: same as --metric)",
    )
    parser.add_argument(
        "--select-metric",
        default=None,
        choices=METRIC_CHOICES,
        help="override winner-selection metric only (default: same as --metric)",
    )
    parser.add_argument(
        "--pos-weight",
        choices=POS_WEIGHT_MODES,
        default=DEFAULT_POS_WEIGHT_MODE,
        help=(
            "positive-class weight for imbalance correction: "
            f"full=neg/pos, sqrt=sqrt(neg/pos) (default: {DEFAULT_POS_WEIGHT_MODE}), "
            "none=1.0 (natural prior)"
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=THRESHOLD_STRATEGIES,
        default=DEFAULT_THRESHOLD_STRATEGY,
        help=(
            "how to tune the decision threshold on OOF train predictions "
            f"(default: {DEFAULT_THRESHOLD_STRATEGY})"
        ),
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=DEFAULT_RECALL_FLOOR,
        help=(
            "minimum recall when --threshold-strategy=recall_floor "
            f"(default: {DEFAULT_RECALL_FLOOR})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        sample=args.sample,
        tune=not args.no_tune,
        probe_audit=args.probe_feature,
        probe_train=args.probe_train,
        metric=args.metric,
        grid_metric=args.grid_metric,
        select_metric=args.select_metric,
        pos_weight_mode=args.pos_weight,
        threshold_strategy=args.threshold_strategy,
        recall_floor=args.recall_floor,
    )
