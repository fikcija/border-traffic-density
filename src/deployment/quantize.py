"""
Post-training quantization + the size / latency / accuracy table.

    python src/quantize.py --model artifacts/mobilenetv2_roi_lb/run_3cls/model_full.keras

Converts the trained model to three TFLite variants and scores every one on the same
held-out test set, then writes artifacts/quantization.{csv,png}.

    float32        the Keras model, unchanged - the reference row
    float16        weights halved, activations still float
    dynamic-range  weights int8, activations float, no calibration data needed
    int8           weights AND activations int8, calibrated on training images

Note on the conversion path: `TFLiteConverter.from_keras_model` is unreliable under
Keras 3, so the model is exported to a SavedModel first and converted from that.

All four variants are scored in a single pass over the test images, so each image is
loaded and preprocessed exactly once and every variant sees byte-identical input.
Latency is measured separately, one image at a time, which is what a REST endpoint
actually does.
"""
import argparse, importlib, json, shutil, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import load_image, BACKBONES
from predict import resolve_card
from roi import load_rois, camera_dir_from_path
from train import CLASSES, BLUE, INK, MUTED, GRID

BATCH = 32


# ------------------------------------------------------------------ conversion
def to_tflite(saved_model_dir, mode, rep_gen=None):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if mode == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif mode == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_gen
        # float in/out with int8 internals: keeps the evaluation loop identical
        # across variants. Fully int8 I/O would need manual input quantization.
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    else:
        raise ValueError(mode)
    return conv.convert()


class TFLiteModel:
    """Single-image TFLite runner. Batch stays 1 - that is the serving shape."""

    def __init__(self, blob):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_content=blob)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict_batch(self, x):
        out = np.zeros((len(x), self.out["shape"][-1]), dtype=np.float32)
        for i, row in enumerate(x):
            self.interp.set_tensor(self.inp["index"], row[None].astype(np.float32))
            self.interp.invoke()
            out[i] = self.interp.get_tensor(self.out["index"])[0]
        return out


# ------------------------------------------------------------------ data
def image_batches(paths, size, rois, letterbox, preprocess_input, batch=BATCH):
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        arr = np.stack([
            load_image(str(p), size,
                       rois.get(camera_dir_from_path(p)) if rois else None,
                       letterbox)
            for p in chunk])
        yield i, preprocess_input(arr)


# ------------------------------------------------------------------ plot
def plot(df, baseline_f1, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.iloc[::-1]
    ys = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.62 * len(d) + 2.4), sharey=True)

    panels = [("size_mb", "size (MB)", "{:.1f}"),
              ("latency_p50_ms", "latency p50 (ms)", "{:.1f}"),
              ("macro_f1", "test macro F1", "{:.3f}")]

    for ax, (col, label, fmt) in zip(axes, panels):
        ax.barh(ys, d[col], height=0.55, color=BLUE)
        for y, v in zip(ys, d[col]):
            ax.text(v + d[col].max() * 0.03, y, fmt.format(v),
                    va="center", fontsize=9, color=MUTED)
        ax.set_xlim(0, d[col].max() * 1.28)
        ax.set_xlabel(label, color=MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)

    axes[2].axvline(baseline_f1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[2].annotate(f"zero-rule {baseline_f1:.2f}", (baseline_f1, 1.01),
                     xycoords=("data", "axes fraction"), ha="center", va="bottom",
                     fontsize=8, color=MUTED)

    axes[0].set_yticks(ys, d["variant"])
    fig.suptitle("Post-training quantization", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    ap.add_argument("--roi", type=Path, default=Path("config/roi_regions.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--latency-n", type=int, default=100,
                    help="images timed one at a time per variant")
    ap.add_argument("--calib-n", type=int, default=200,
                    help="training images used to calibrate int8 activations")
    args = ap.parse_args()

    import keras, tensorflow as tf
    from sklearn.metrics import f1_score, accuracy_score

    backbone_name, size, labels, roi_cfg = resolve_card(args.model, args.classes)
    preprocess_input = importlib.import_module(
        f"keras.applications.{BACKBONES[backbone_name][0]}").preprocess_input

    rois, letterbox = None, bool(roi_cfg.get("letterbox"))
    if roi_cfg.get("active"):
        if not args.roi.exists():
            raise SystemExit(f"ROI model needs polygons; not found at {args.roi}")
        rois = load_rois(args.roi)
        print(f"ROI masking ACTIVE ({len(rois)} polygons)"
              + (" letterboxed" if letterbox else ""))

    run_dir = args.model.parent
    manifest = pd.read_csv(run_dir.parent / "manifest.csv")
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    train = manifest[manifest.split == "train"]
    y_true = test[f"class_{args.classes}"].map(
        {c: i for i, c in enumerate(labels)}).values
    print(f"test images: {len(test)}   calibration images: {args.calib_n}")

    keras_model = keras.models.load_model(args.model)

    # --- export once, convert three ways --------------------------------------
    tmp = Path(tempfile.mkdtemp())
    sm_dir = tmp / "saved_model"
    keras_model.export(str(sm_dir))          # Keras 3: .export, not .save

    calib_paths = train.path.sample(
        min(args.calib_n, len(train)), random_state=42).tolist()

    def rep_gen():
        for _, batch in image_batches(calib_paths, size, rois, letterbox,
                                      preprocess_input, batch=1):
            yield [batch.astype(np.float32)]

    variants = {"float32": None}
    for mode in ("float16", "dynamic", "int8"):
        print(f"converting {mode} ...", flush=True)
        t0 = time.time()
        variants[mode] = to_tflite(sm_dir, mode, rep_gen if mode == "int8" else None)
        print(f"  done in {time.time()-t0:.0f}s ({len(variants[mode])/1e6:.1f} MB)")

    runners = {"float32": keras_model}
    for mode in ("float16", "dynamic", "int8"):
        runners[mode] = TFLiteModel(variants[mode])

    # --- score every variant in ONE pass over the images ----------------------
    preds = {k: np.zeros((len(test), len(labels)), dtype=np.float32) for k in runners}
    for i, batch in image_batches(test.path.tolist(), size, rois, letterbox,
                                  preprocess_input):
        preds["float32"][i:i + len(batch)] = keras_model.predict(batch, verbose=0)
        for mode in ("float16", "dynamic", "int8"):
            preds[mode][i:i + len(batch)] = runners[mode].predict_batch(batch)
        print(f"\r  scoring {min(i+BATCH, len(test))}/{len(test)}", end="", flush=True)
    print()

    # --- latency, one image at a time -----------------------------------------
    _, warm = next(image_batches(test.path.tolist()[:args.latency_n], size, rois,
                                 letterbox, preprocess_input, batch=args.latency_n))

    # Graph-compile the Keras call. Timing it in eager mode would charge float32 a
    # per-call Python overhead that a real server does not pay, making the
    # quantization speed-up look better than it is.
    fp32_call = tf.function(lambda x: keras_model(x, training=False))
    calls = {"float32": lambda row: fp32_call(row[None])}
    for mode in ("float16", "dynamic", "int8"):
        calls[mode] = (lambda r, m=mode: runners[m].predict_batch(r[None]))

    latency = {}
    for name, call in calls.items():
        for row in warm[:5]:                                   # warm-up / trace
            call(row)
        ts = []
        for row in warm:
            t0 = time.perf_counter()
            call(row)
            ts.append((time.perf_counter() - t0) * 1000)
        latency[name] = float(np.median(ts))

    # --- table ----------------------------------------------------------------
    sizes = {"float32": args.model.stat().st_size}
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in ("float16", "dynamic", "int8"):
        p = args.out / f"model_{mode}.tflite"
        p.write_bytes(variants[mode])
        sizes[mode] = p.stat().st_size

    pretty = {"float32": "Keras float32", "float16": "TFLite float16",
              "dynamic": "TFLite dynamic-range", "int8": "TFLite int8"}
    rows = []
    for mode in runners:
        yp = preds[mode].argmax(1)
        rows.append({
            "variant": pretty[mode],
            "size_mb": sizes[mode] / 1e6,
            "latency_p50_ms": latency[mode],
            "accuracy": accuracy_score(y_true, yp),
            "macro_f1": f1_score(y_true, yp, average="macro", zero_division=0),
            "agreement_with_fp32": float((yp == preds["float32"].argmax(1)).mean()),
        })
    df = pd.DataFrame(rows)

    base = json.loads((run_dir / "results.json").read_text())
    baseline_f1 = base["baselines"]["zero_rule"]["macro_f1"]

    df.to_csv(args.out / "quantization.csv", index=False)
    plot(df, baseline_f1, args.out / "quantization.png")
    shutil.rmtree(tmp, ignore_errors=True)

    pd.set_option("display.width", 200)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nzero-rule baseline macro F1: {baseline_f1:.4f}")
    print(f"wrote {args.out}/quantization.csv, quantization.png, model_*.tflite")


if __name__ == "__main__":
    main()
