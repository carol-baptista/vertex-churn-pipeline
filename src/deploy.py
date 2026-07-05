"""Deploy the champion serving bundle to Vertex AI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from google.cloud import aiplatform, storage
from google.cloud.aiplatform.prediction import LocalModel

from . import config
from .champion import CHAMPION_MODEL, SERVING_DIR, build_serving_metadata
from .package import load_predictor_class, package_champion

GCS_PREFIX = "models/churn-rf/v1"
ARTIFACT_REGISTRY_REPO = os.getenv("VERTEX_ARTIFACT_REPO", "vertex-churn")
IMAGE_NAME = os.getenv("VERTEX_SERVING_IMAGE", "churn-rf-serving")
MODEL_DISPLAY_NAME = os.getenv("VERTEX_MODEL_DISPLAY_NAME", "churn-predictor")
ENDPOINT_DISPLAY_NAME = os.getenv("VERTEX_ENDPOINT_DISPLAY_NAME", "churn-endpoint")
MACHINE_TYPE = os.getenv("VERTEX_MACHINE_TYPE", "n1-standard-2")


def gcs_uri() -> str:
    bucket = os.getenv("GCS_BUCKET", "")
    if not bucket:
        raise ValueError("GCS_BUCKET is not set in .env")
    return f"gs://{bucket}/{GCS_PREFIX}"


def artifact_registry_image_uri() -> str:
    project = os.getenv("GCP_PROJECT_ID", config.PROJECT_ID)
    region = os.getenv("GCP_REGION", config.REGION)
    return f"{region}-docker.pkg.dev/{project}/{ARTIFACT_REGISTRY_REPO}/{IMAGE_NAME}:v1"


def ensure_artifact_registry_repo() -> None:
    project = os.getenv("GCP_PROJECT_ID", config.PROJECT_ID)
    region = os.getenv("GCP_REGION", config.REGION)
    repo = ARTIFACT_REGISTRY_REPO
    describe = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            repo,
            f"--location={region}",
            f"--project={project}",
        ],
        capture_output=True,
        text=True,
    )
    if describe.returncode == 0:
        return
    subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "create",
            repo,
            f"--location={region}",
            f"--repository-format=docker",
            f"--project={project}",
        ],
        check=True,
    )


def upload_bundle(bundle_dir: Path) -> str:
    uri = gcs_uri()
    bucket_name = uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(uri.replace("gs://", "").split("/")[1:])

    client = storage.Client(project=os.getenv("GCP_PROJECT_ID", config.PROJECT_ID))
    bucket = client.bucket(bucket_name)
    for path in sorted(bundle_dir.iterdir()):
        if path.is_file():
            blob = bucket.blob(f"{prefix}/{path.name}")
            blob.upload_from_filename(str(path))
    return uri


def build_and_push_image(bundle_dir: Path) -> str:
    ensure_artifact_registry_repo()
    predictor_cls = load_predictor_class(bundle_dir)
    image_uri = artifact_registry_image_uri()
    local_model = LocalModel.build_cpr_model(
        str(bundle_dir),
        image_uri,
        predictor=predictor_cls,
        requirements_path=str(bundle_dir / "requirements.txt"),
    )
    local_model.push_image()
    return image_uri


def register_model(*, artifact_uri: str, serving_image: str) -> aiplatform.Model:
    manifest_path = SERVING_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        test = manifest.get("headline_metrics") or {}
        threshold = manifest.get("threshold")
        model_name = manifest.get("model", CHAMPION_MODEL)
        release_id = manifest.get("release_id", "v1")
    else:
        meta = build_serving_metadata()
        test = meta.get("test_metrics") or {}
        threshold = meta.get("threshold")
        model_name = CHAMPION_MODEL
        release_id = "v1"

    return aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        labels={
            "feature_set": "baseline",
            "model": model_name.replace("_", "-"),
            "release": release_id,
            "threshold": str(round(float(threshold), 4)).replace(".", "-"),
        },
        description=(
            f"Churn RF release {release_id} ({model_name}). "
            f"test_f1={test.get('f1')}, test_recall={test.get('recall')}, "
            f"threshold={float(threshold):.4f}"
        ),
    )


def get_or_create_endpoint() -> aiplatform.Endpoint:
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if endpoints:
        return endpoints[0]
    return aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)


def deploy_model(model: aiplatform.Model) -> aiplatform.Endpoint:
    endpoint = get_or_create_endpoint()
    endpoint.deploy(
        model,
        deployed_model_display_name=f"{CHAMPION_MODEL}-v1",
        machine_type=MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
    )
    return endpoint


def undeploy_endpoint() -> None:
    endpoint = get_or_create_endpoint()
    endpoint.undeploy_all()
    print(f"Undeployed all models from {ENDPOINT_DISPLAY_NAME}")


def smoke_predict_endpoint(endpoint: aiplatform.Endpoint, instance: dict) -> dict:
    response = endpoint.predict(instances=[instance])
    preds = response.predictions
    if isinstance(preds, list) and preds:
        return preds[0]
    return {"predictions": preds}


def deploy_champion(*, dry_run: bool = False, register_only: bool = False) -> None:
    bundle_dir = package_champion()
    uri = gcs_uri()
    image_uri = artifact_registry_image_uri()

    print("Deploy plan:")
    print(f"  bundle:  {bundle_dir}")
    print(f"  gcs:     {uri}")
    print(f"  image:   {image_uri}")
    print(f"  model:   {MODEL_DISPLAY_NAME}")
    print(f"  endpoint:{ENDPOINT_DISPLAY_NAME}")

    if dry_run:
        print("Dry run — no GCP changes made.")
        return

    project = os.getenv("GCP_PROJECT_ID", config.PROJECT_ID)
    region = os.getenv("GCP_REGION", config.REGION)
    aiplatform.init(project=project, location=region)

    upload_bundle(bundle_dir)
    print(f"Uploaded bundle -> {uri}")

    serving_image = build_and_push_image(bundle_dir)
    print(f"Pushed CPR image -> {serving_image}")

    model = register_model(artifact_uri=uri, serving_image=serving_image)
    print(f"Registered model -> {model.resource_name}")

    if register_only:
        print("Register-only — skipping endpoint deploy.")
        return

    endpoint = deploy_model(model)
    print(f"Deployed to endpoint -> {endpoint.resource_name}")
    print("Run `make undeploy` when finished to stop endpoint billing.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deploy champion bundle to Vertex AI.")
    parser.add_argument("--dry-run", action="store_true", help="print plan only")
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="upload + register model, skip endpoint deploy",
    )
    parser.add_argument(
        "--undeploy",
        action="store_true",
        help="undeploy all models from the endpoint",
    )
    args = parser.parse_args(argv)

    if args.undeploy:
        project = os.getenv("GCP_PROJECT_ID", config.PROJECT_ID)
        region = os.getenv("GCP_REGION", config.REGION)
        aiplatform.init(project=project, location=region)
        undeploy_endpoint()
        return

    deploy_champion(dry_run=args.dry_run, register_only=args.register_only)


if __name__ == "__main__":
    main()
