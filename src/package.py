"""Assemble the Vertex serving bundle for the Random Forest champion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .champion import (
    CHAMPION_MODEL,
    SERVING_DIR,
    SERVING_VERSION,
    baseline_feature_columns,
    build_manifest,
    build_serving_metadata,
    load_pipeline,
    load_threshold,
    pipeline_path,
    score_rows,
    validate_feature_frame,
)

PREDICTOR_SRC = config.REPO_ROOT / "serving" / "churn-rf" / "v1" / "predictor.py"
REQUIREMENTS_SRC = config.REPO_ROOT / "serving" / "churn-rf" / "v1" / "requirements.txt"


def bundle_paths(out_dir: Path | None = None) -> dict[str, Path]:
    root = out_dir or SERVING_DIR
    return {
        "root": root,
        "model": root / "model.joblib",
        "threshold": root / "threshold.json",
        "feature_schema": root / "feature_schema.json",
        "manifest": root / "manifest.json",
        "predictor": root / "predictor.py",
        "requirements": root / "requirements.txt",
    }


def package_champion(*, out_dir: Path | None = None, model: str = CHAMPION_MODEL) -> Path:
    """Copy champion artifacts into ``serving/churn-rf/v1/``."""
    if not pipeline_path(model).exists():
        raise FileNotFoundError(
            f"{pipeline_path(model)} not found. Run `make train` first."
        )

    paths = bundle_paths(out_dir)
    root = paths["root"]
    root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pipeline_path(model), paths["model"])

    metadata = build_serving_metadata(model)
    manifest = build_manifest(model=model, release_id=SERVING_VERSION)
    paths["threshold"].write_text(json.dumps(metadata, indent=2))
    paths["manifest"].write_text(json.dumps(manifest, indent=2))
    paths["feature_schema"].write_text(
        json.dumps({"feature_columns": metadata["feature_columns"]}, indent=2)
    )
    if PREDICTOR_SRC.resolve() != paths["predictor"].resolve():
        shutil.copy2(PREDICTOR_SRC, paths["predictor"])
    if REQUIREMENTS_SRC.resolve() != paths["requirements"].resolve():
        shutil.copy2(REQUIREMENTS_SRC, paths["requirements"])

    return root


def load_predictor_class(bundle_dir: Path):
    """Import ``ChurnPredictor`` from a packaged bundle directory."""
    spec = importlib.util.spec_from_file_location(
        "churn_predictor", bundle_dir / "predictor.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load predictor from {bundle_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ChurnPredictor


def load_predictor_from_dir(bundle_dir: Path):
    """Load ``ChurnPredictor`` from a local bundle (no GCS download)."""
    import joblib

    predictor_cls = load_predictor_class(bundle_dir)
    predictor = predictor_cls()
    predictor._pipe = joblib.load(bundle_dir / "model.joblib")
    with (bundle_dir / "threshold.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    predictor._threshold = float(meta["threshold"])
    with (bundle_dir / "feature_schema.json").open(encoding="utf-8") as f:
        schema = json.load(f)
    predictor._feature_columns = list(schema["feature_columns"])
    return predictor


def predict_with_bundle(
    bundle_dir: Path,
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the packaged CPR predictor locally (no Vertex endpoint)."""
    predictor = load_predictor_from_dir(bundle_dir)
    frame = predictor.preprocess({"instances": instances})
    proba = predictor.predict(frame)
    return predictor.postprocess(proba)["predictions"]


def _synthetic_instance() -> dict[str, Any]:
    """One valid baseline feature row for local bundle smoke tests."""
    from .preprocess import make_dataset

    raw = pd.DataFrame(
        [
            {
                "customerID": "smoke-1",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": True,
                "Dependents": False,
                "tenure": 12,
                "PhoneService": True,
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": True,
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0,
                "Churn": False,
            }
        ]
    )
    return make_dataset(raw, engineered=False).X.iloc[0].to_dict()


def smoke_test_bundle(bundle_dir: Path | None = None) -> None:
    """Verify bundle scoring matches champion.load_pipeline on a synthetic row."""
    root = bundle_dir or SERVING_DIR
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run `make package` first.")

    instance = _synthetic_instance()
    frame = pd.DataFrame([instance])
    pipe = load_pipeline()
    threshold = load_threshold()
    validate_feature_frame(frame, columns=baseline_feature_columns())

    expected = score_rows(pipe, frame, threshold)[0]
    actual = predict_with_bundle(root, [instance])[0]

    if abs(expected["churn_probability"] - actual["churn_probability"]) > 1e-6:
        raise AssertionError(
            f"Probability mismatch: expected {expected['churn_probability']}, "
            f"got {actual['churn_probability']}"
        )
    if expected["churn_flag"] != actual["churn_flag"]:
        raise AssertionError(
            f"Flag mismatch: expected {expected['churn_flag']}, got {actual['churn_flag']}"
        )

    print(f"Bundle smoke test OK ({root})")
    print(
        f"  churn_probability={actual['churn_probability']:.4f}  "
        f"churn_flag={actual['churn_flag']}  threshold={actual['threshold']:.4f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Package or smoke-test the serving bundle.")
    parser.add_argument(
        "command",
        choices=("package", "smoke-test"),
        help="package=assemble serving/churn-rf/v1; smoke-test=verify local scoring",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"output directory (default: {SERVING_DIR})",
    )
    args = parser.parse_args(argv)

    if args.command == "package":
        root = package_champion(out_dir=args.out_dir)
        print(f"Packaged champion bundle -> {root}")
        for name in (
            "model.joblib",
            "manifest.json",
            "threshold.json",
            "feature_schema.json",
            "predictor.py",
        ):
            print(f"  {root / name}")
    else:
        smoke_test_bundle(args.out_dir)


if __name__ == "__main__":
    main()
