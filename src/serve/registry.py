"""
Resolve which model the API should serve.

Two modes, chosen by environment:

  registry  MODEL_NAME set -> fetch the version carrying MODEL_ALIAS from the
            MLflow model registry and download its artifacts locally
  local     otherwise -> use MODEL_PATH / ROI_PATH from disk

The registry mode is what makes promotion and rollback cheap: the serving image
contains no model at all, so shipping a new one is an alias change plus a reload,
and rolling back is the same operation pointing at the previous version. The local
mode exists so the container still runs standalone, without the full stack.

Uses mlflow-skinny - the client, without the server and UI dependencies.
"""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResolvedModel:
    model_path: Path
    roi_path: Path
    source: str          # "registry" or "local"
    version: str | None
    run_id: str | None

    def describe(self):
        return {"source": self.source, "version": self.version,
                "run_id": self.run_id, "model": self.model_path.name}


def _from_registry(name, alias, model_file):
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    # Download the whole artifact directory: the model, its card, and the ROI
    # polygons were registered together precisely so serving gets a consistent set
    # and can never pair a model with the wrong polygons.
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=version.source, dst_path=tempfile.mkdtemp())
    local = Path(local)

    model_path = local / model_file
    if not model_path.exists():
        found = sorted(p.name for p in local.glob("*"))
        raise RuntimeError(
            f"{model_file} not in registered artifacts for {name}@{alias}; found {found}")

    return ResolvedModel(model_path, local / "roi_regions.json", "registry",
                         version.version, version.run_id)


def resolve() -> ResolvedModel:
    name = os.getenv("MODEL_NAME")
    model_file = os.getenv("MODEL_FILE", "model_int8.tflite")

    if name and os.getenv("MLFLOW_TRACKING_URI"):
        alias = os.getenv("MODEL_ALIAS", "production")
        try:
            return _from_registry(name, alias, model_file)
        except Exception as e:
            # A registry lookup can legitimately fail before the first training run
            # has promoted anything. Fall back only if a local model exists, and be
            # loud about which one is actually being served.
            local = Path(os.getenv("MODEL_PATH", model_file))
            if not local.exists():
                raise RuntimeError(
                    f"no model {name}@{alias} in the registry and no local fallback "
                    f"at {local}. Run the training pipeline first.") from e
            print(f"WARNING: registry lookup failed ({e}); serving local {local}")

    return ResolvedModel(
        Path(os.getenv("MODEL_PATH", model_file)),
        Path(os.getenv("ROI_PATH", "roi_regions.json")),
        "local", None, None)
