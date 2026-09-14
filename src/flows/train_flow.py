"""
Orchestrated retraining workflow (Prefect).

    prepare_features → run_search → export_model → run_perturbation
        → quantize → log_and_register
        → evaluate_gate → promote → reload_api

Each stage is its own Prefect task calling the matching function in `pipeline`, so a
failure names the stage that broke. Stages retry independently - a flaky MLflow
connection re-runs `log_and_register`, not the twenty-minute search.

Runs on a schedule (TRAIN_CRON, default Mondays 03:00) and from the UI on :4200.
"""
import os
import sys
import time
import urllib.request

import httpx
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

import pipeline
from pipeline import Config

MODEL_NAME = os.getenv("MODEL_NAME", "border-traffic-density")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# Validation, not test: gating on test_macro_f1 would spend the held-out set on
# model selection and stop it being an unbiased estimate.
GATE_METRIC = os.getenv("GATE_METRIC", "val_macro_f1")
GATE_MARGIN = float(os.getenv("GATE_MARGIN", "0.0"))
# Clean accuracy alone would promote a model that collapses under sensor noise, so
# the gate also reads what run_perturbation measured.
ROBUSTNESS_METRIC = os.getenv("ROBUSTNESS_METRIC", "robustness_worst_macro_f1")
ROBUSTNESS_CLEAN_METRIC = os.getenv("ROBUSTNESS_CLEAN_METRIC",
                                    "robustness_clean_macro_f1")
# A fraction of the candidate's own clean score, not an absolute floor: the ratio
# keeps meaning the same thing as models improve or the task gets harder.
ROBUSTNESS_MIN_RETENTION = float(os.getenv("ROBUSTNESS_MIN_RETENTION", "0.50"))
# Bounds how much robustness a single promotion may give up; retention alone would
# permit slow drift.
ROBUSTNESS_REGRESSION_TOLERANCE = float(
    os.getenv("ROBUSTNESS_REGRESSION_TOLERANCE", "0.02"))


def _cfg(backbone: str, trials: int, letterbox: bool) -> Config:
    return Config(backbone=backbone, trials=trials, letterbox=letterbox)


# ---------------------------------------------------------------- pipeline stages
@task(name="prepare-features", retries=1, retry_delay_seconds=30, log_prints=True)
def prepare_features(backbone: str, trials: int, letterbox: bool):
    """Cache frozen backbone features. No-op when already cached."""
    pipeline.step_prepare(_cfg(backbone, trials, letterbox))


@task(name="run-search", retries=1, retry_delay_seconds=60, log_prints=True)
def run_search(backbone: str, trials: int, letterbox: bool):
    """Bayesian + random hyperparameter search under one budget, then the final model."""
    pipeline.step_search(_cfg(backbone, trials, letterbox))


@task(name="export-model", retries=1, retry_delay_seconds=30, log_prints=True)
def export_model(backbone: str, trials: int, letterbox: bool):
    """Merge backbone and head into one servable file, write the model card."""
    pipeline.step_export(_cfg(backbone, trials, letterbox))


@task(name="run-perturbation", retries=1, retry_delay_seconds=30, log_prints=True)
def run_perturbation(backbone: str, trials: int, letterbox: bool):
    """Robustness under noise, JPEG, blur and low light - re-measured every retrain."""
    pipeline.step_perturb(_cfg(backbone, trials, letterbox))


@task(name="quantize", retries=1, retry_delay_seconds=30, log_prints=True)
def quantize(backbone: str, trials: int, letterbox: bool):
    """TFLite float16 / dynamic-range / int8 plus the size-latency-accuracy table."""
    pipeline.step_quantize(_cfg(backbone, trials, letterbox))


@task(name="log-and-register", retries=2, retry_delay_seconds=30, log_prints=True)
def log_and_register(backbone: str, trials: int, letterbox: bool) -> str:
    """Log the run to MLflow and register a model version. Returns the run id."""
    run_id = pipeline.step_register(_cfg(backbone, trials, letterbox))
    get_run_logger().info(f"registered from run {run_id}")
    return run_id


# ---------------------------------------------------------------- promotion
@task(name="evaluate-gate", log_prints=True)
def evaluate_gate(run_id: str) -> bool:
    """True when the candidate beats whatever is serving right now.

    Without this, an automated pipeline promotes whatever it last trained - so one
    bad run silently replaces a good model. Rejecting is the useful behaviour.

    Three conditions, all of which must hold:

      accuracy    candidate beats production on GATE_METRIC by GATE_MARGIN
      retention   candidate keeps ROBUSTNESS_MIN_RETENTION of its own clean score
                  under the worst perturbation measured
      regression  candidate is no less robust than production, within tolerance

    Retention is self-referential, so it applies even to the first-ever model. The
    regression check needs something to compare against and is skipped when there is
    no incumbent, or when the incumbent predates the perturbation stage.
    """
    log = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    metrics = client.get_run(run_id).data.metrics
    new = metrics.get(GATE_METRIC)
    if new is None:
        raise RuntimeError(f"run {run_id} logged no {GATE_METRIC}")

    # run_perturbation is unconditional upstream, so missing metrics here mean an
    # inconsistent run. Fail loudly rather than skip a safety check.
    worst = metrics.get(ROBUSTNESS_METRIC)
    clean = metrics.get(ROBUSTNESS_CLEAN_METRIC)
    if worst is None or clean is None:
        raise RuntimeError(
            f"run {run_id} logged no {ROBUSTNESS_METRIC} / {ROBUSTNESS_CLEAN_METRIC} "
            f"- did the perturbation stage run?")

    retention = worst / clean if clean > 0 else 0.0
    robust_enough = retention >= ROBUSTNESS_MIN_RETENTION
    log.info(f"retention: worst {worst:.4f} / clean {clean:.4f} = {retention:.3f} "
             f"(min {ROBUSTNESS_MIN_RETENTION}) -> "
             f"{'PASS' if robust_enough else 'FAIL'}")

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        incumbent_metrics = client.get_run(current.run_id).data.metrics
    except Exception:
        log.info("no model in production yet - the first version only has to be "
                 "robust enough")
        return robust_enough

    incumbent = incumbent_metrics.get(GATE_METRIC, 0.0)
    better = new > incumbent + GATE_MARGIN
    log.info(f"{GATE_METRIC}: candidate {new:.4f} vs production {incumbent:.4f} "
             f"(margin {GATE_MARGIN}) -> {'PASS' if better else 'FAIL'}")

    incumbent_worst = incumbent_metrics.get(ROBUSTNESS_METRIC)
    if incumbent_worst is None:
        # Nothing to regress from.
        log.info(f"production logged no {ROBUSTNESS_METRIC} - regression check skipped")
        no_regression = True
    else:
        no_regression = worst >= incumbent_worst - ROBUSTNESS_REGRESSION_TOLERANCE
        log.info(f"{ROBUSTNESS_METRIC}: candidate {worst:.4f} vs production "
                 f"{incumbent_worst:.4f} (tolerance {ROBUSTNESS_REGRESSION_TOLERANCE}) "
                 f"-> {'PASS' if no_regression else 'FAIL'}")

    decision = better and robust_enough and no_regression
    log.info(f"gate -> {'PROMOTE' if decision else 'REJECT'}")
    return decision


@task(name="promote", log_prints=True)
def promote(run_id: str) -> str:
    """Move the production alias onto the version this run produced."""
    log = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    versions = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
                if v.run_id == run_id]
    if not versions:
        raise RuntimeError(f"no registered version for run {run_id}")
    version = max(versions, key=lambda v: int(v.version)).version
    # Promotion and rollback are the same one-line operation.
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    log.info(f"{MODEL_NAME} v{version} is now @{MODEL_ALIAS}")
    return version


@task(name="reload-api", retries=2, retry_delay_seconds=10, log_prints=True)
def reload_api():
    """Ask the API process to pick up the newly promoted version.

    Not fatal: the promotion is already durable and the API re-resolves on its next
    start, so an unreachable API must not fail an otherwise successful retrain.
    """
    log = get_run_logger()
    # The API leaves /reload open while RELOAD_TOKEN is unset.
    headers = {"X-Reload-Token": RELOAD_TOKEN} if RELOAD_TOKEN else {}
    try:
        r = httpx.post(f"{API_URL}/reload", headers=headers, timeout=120)
        r.raise_for_status()
        log.info(f"api reloaded: {r.json()}")
    except Exception as e:
        log.warning(f"could not reload the API ({e}). The new version is promoted "
                    f"and will be served on the API's next start.")


@flow(name="retrain-border-traffic", log_prints=True)
def retrain(backbone: str = "mobilenetv2", trials: int = 25, letterbox: bool = True):
    args = (backbone, trials, letterbox)
    prepare_features(*args)
    run_search(*args)
    export_model(*args)
    run_perturbation(*args)
    quantize(*args)
    run_id = log_and_register(*args)

    if evaluate_gate(run_id):
        promote(run_id)
        reload_api()
    else:
        get_run_logger().info("candidate rejected - production left untouched")


def _wait_for_prefect(url: str, timeout: int = 120):
    """supervisord starts this alongside the server, so wait for the API."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(2)
    sys.exit(f"Prefect API did not come up at {url} within {timeout}s")


if __name__ == "__main__":
    api = os.getenv("PREFECT_API_URL", "http://127.0.0.1:4200/api")
    print(f"waiting for the Prefect API at {api} ...", flush=True)
    _wait_for_prefect(api.rstrip("/") + "/health")
    cron = os.getenv("TRAIN_CRON", "0 3 * * 1")
    print(f"serving flow 'retrain-border-traffic' on cron {cron!r} (concurrency 1)",
          flush=True)
    # limit=1 is correctness, not tuning: every stage writes to the same
    # artifacts/<backbone>/ directory, so concurrent runs would interleave their
    # tuner state, results.json and model files. A second trigger queues.
    retrain.serve(name="scheduled-retrain", cron=cron, limit=1)
