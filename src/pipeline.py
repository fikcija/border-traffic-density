"""
The training pipeline, as importable stages.

    # as a library (what the Prefect flow does)
    from pipeline import Config, step_search
    step_search(Config(backbone="mobilenetv2", letterbox=True))

    # as a CLI (unchanged, still works standalone)
    python src/pipeline.py --step search --backbone mobilenetv2 --letterbox

Stages, each reusing the existing script rather than reimplementing it:

    prepare    cache frozen backbone features    (no-op if already cached)
    search     AutoML search + final model       (train.py)
    export     merge backbone + head, model card (export_model.py)
    perturb    robustness under degraded input   (perturb.py)
    quantize   TFLite variants + size/latency/accuracy table
    register   log to MLflow and register a model version -> returns the run id

The flow calls these functions directly, one per Prefect task, so a failure names
the stage instead of reporting that "training died" somewhere.
"""
import argparse, json, shutil, subprocess, sys, tempfile
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

STEPS = ["prepare", "search", "export", "perturb", "quantize", "register"]


@dataclass
class Config:
    backbone: str = "mobilenetv2"
    classes: int = 3
    trials: int = 25
    roi: str = "config/roi_regions.json"
    letterbox: bool = True
    no_roi: bool = False
    data_dir: str = "data"
    experiment: str = "border-traffic-density"
    model_name: str = "border-traffic-density"

    @property
    def art(self) -> Path:
        suffix = "" if self.no_roi else ("_roi_lb" if self.letterbox else "_roi")
        return ROOT / "artifacts" / f"{self.backbone}{suffix}"

    @property
    def run_dir(self) -> Path:
        return self.art / f"run_{self.classes}cls"

    @property
    def quant_dir(self) -> Path:
        return self.art / "quant"


def run(cmd):
    """Run a sub-step, streaming its output. Raise loudly on failure."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[1])).name} failed (exit {r.returncode})")


# ---------------------------------------------------------------- stages
def step_prepare(cfg: Config):
    """Expensive and deterministic, so it is cached and skipped when present."""
    if (cfg.art / "features.npy").exists():
        print(f"features already cached at {cfg.art}/features.npy - nothing to do")
        return
    cmd = [sys.executable, "src/data/prepare.py", "--data-dir", cfg.data_dir,
           "--backbone", cfg.backbone]
    if not cfg.no_roi:
        cmd += ["--roi", cfg.roi]
        if cfg.letterbox:
            cmd += ["--roi-letterbox"]
    run(cmd)


def step_search(cfg: Config):
    run([sys.executable, "src/training/train.py", "--artifacts", cfg.art,
         "--classes", cfg.classes, "--trials", cfg.trials])


def step_export(cfg: Config):
    run([sys.executable, "src/deployment/export_model.py", "--artifacts", cfg.art,
         "--run", f"run_{cfg.classes}cls", "--classes", cfg.classes])


def step_perturb(cfg: Config):
    """Re-score the model under noise, JPEG, blur and low light.

    A pipeline stage rather than a one-off script: robustness belongs to a specific
    model, so it has to be re-measured whenever one is retrained.
    """
    run([sys.executable, "src/evaluation/perturb.py",
         "--model", cfg.run_dir / "model_full.keras",
         "--classes", cfg.classes, "--roi", cfg.roi])


def step_quantize(cfg: Config):
    run([sys.executable, "src/deployment/quantize.py",
         "--model", cfg.run_dir / "model_full.keras",
         "--classes", cfg.classes, "--roi", cfg.roi, "--out", cfg.quant_dir])


def step_register(cfg: Config) -> str:
    """Log the run to MLflow, register a model version, return the MLflow run id."""
    import mlflow, pandas as pd

    results = json.loads((cfg.run_dir / "results.json").read_text())
    prep = json.loads((cfg.art / "prepare_config.json").read_text())
    quant = pd.read_csv(cfg.quant_dir / "quantization.csv").set_index("variant")
    pert_path = cfg.run_dir / "perturbation_summary.json"
    pert = json.loads(pert_path.read_text()) if pert_path.exists() else {}

    bayes = results["search"]["bayesian"]["best_val_macro_f1"]
    rand = results["search"]["random"]["best_val_macro_f1"]

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run() as active:
        mlflow.log_params({
            "backbone": cfg.backbone, "classes": cfg.classes, "trials": cfg.trials,
            "roi": not cfg.no_roi, "letterbox": cfg.letterbox,
            "input_size": prep["size"], "val_start": prep["val_start"],
            "test_start": prep["test_start"],
            "search_winner": results["winner"]["search"],
            **{f"hp_{k}": v for k, v in results["winner"]["hyperparameters"].items()},
        })
        mlflow.log_metrics({
            # The promotion gate reads val_macro_f1. Promotion is a SELECTION
            # decision, and selecting on test would turn the held-out set into a
            # validation set - the estimate stops being unbiased after the first
            # few promotions. Test metrics are logged for reporting only.
            "val_macro_f1": max(bayes, rand),
            "val_macro_f1_bayesian": bayes,
            "val_macro_f1_random": rand,
            "test_macro_f1": results["test"]["macro_f1"],
            "test_accuracy": results["test"]["accuracy"],
            "baseline_zero_rule_macro_f1": results["baselines"]["zero_rule"]["macro_f1"],
            # Robustness travels with the model: a candidate that scores the same
            # on clean data but collapses under noise should be visible here.
            **{f"robustness_{k}": v for k, v in pert.items()
               if isinstance(v, (int, float))},
            **{f"quant_{v.replace(' ', '_').replace('-', '_').lower()}_{m}":
               quant.loc[v, m]
               for v in quant.index
               for m in ("size_mb", "latency_p50_ms", "macro_f1", "agreement_with_fp32")
               if pd.notna(quant.loc[v, m])},
        })
        for f in ("confusion_matrix.png", "search_comparison.png", "per_camera.csv",
                  "perturbation.png", "perturbation.csv"):
            if (cfg.run_dir / f).exists():
                mlflow.log_artifact(cfg.run_dir / f, "evaluation")
        for f in ("quantization.csv", "quantization.png"):
            mlflow.log_artifact(cfg.quant_dir / f, "evaluation")

        # The registered artifact is exactly the serving payload: quantized model,
        # its card, and the ROI polygons it requires. Registered together so serving
        # can never pair a model with the wrong polygons.
        staging = Path(tempfile.mkdtemp()) / "model"
        staging.mkdir(parents=True)
        shutil.copy(cfg.quant_dir / "model_int8.tflite", staging / "model_int8.tflite")
        shutil.copy(cfg.run_dir / "model_card.json", staging / "model_card.json")
        if not cfg.no_roi:
            shutil.copy(ROOT / cfg.roi, staging / "roi_regions.json")
        mlflow.log_artifacts(staging, "model")

        mlflow.register_model(f"runs:/{active.info.run_id}/model", cfg.model_name)
        print(f"val macro F1 {max(bayes, rand):.4f} | "
              f"test macro F1 {results['test']['macro_f1']:.4f} "
              f"(baseline {results['baselines']['zero_rule']['macro_f1']:.4f})")
        print(f"MLFLOW_RUN_ID={active.info.run_id}")
        return active.info.run_id


STAGE_FN = {"prepare": step_prepare, "search": step_search, "export": step_export,
            "perturb": step_perturb, "quantize": step_quantize,
            "register": step_register}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=STEPS + ["all"], default="all")
    ap.add_argument("--backbone", default="mobilenetv2")
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--roi", default="config/roi_regions.json")
    ap.add_argument("--letterbox", action="store_true")
    ap.add_argument("--no-roi", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--experiment", default="border-traffic-density")
    ap.add_argument("--model-name", default="border-traffic-density")
    a = ap.parse_args()

    cfg = Config(backbone=a.backbone, classes=a.classes, trials=a.trials, roi=a.roi,
                 letterbox=a.letterbox, no_roi=a.no_roi, data_dir=a.data_dir,
                 experiment=a.experiment, model_name=a.model_name)
    for name in (STEPS if a.step == "all" else [a.step]):
        print(f"\n=== step: {name} ===", flush=True)
        STAGE_FN[name](cfg)


if __name__ == "__main__":
    main()
