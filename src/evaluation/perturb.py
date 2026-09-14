"""
Perturbation testing - a pipeline stage, not a one-off script.

    python src/perturb.py --model artifacts/<run>/model_full.keras

Measures how the trained model holds up when the input degrades in ways border
cameras actually degrade: sensor noise, JPEG re-compression, defocus or rain on the
lens, and low light. Each family is applied at three increasing severities and the
model is re-scored on the whole test set.

Why this belongs in the pipeline: robustness is a property of *this* model, so it
has to be re-measured every time a model is retrained. Run by hand once, it is a
historical note; run as a stage, it is a standing check that a newly promoted model
is not more fragile than the one it replaced.

Note on circularity: perturbing brightness would be meaningless if the model had
been trained with brightness augmentation - you would be testing on the
distribution you trained on. The served model is trained on cached frozen features,
so it has seen NO augmentation of any kind, and every family below is a genuinely
out-of-distribution test.

Writes perturbation.csv and perturbation.png next to the model.
"""
import argparse, importlib, io, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from prepare import load_image, BACKBONES
from predict import resolve_card
from roi import load_rois, camera_dir_from_path
from train import CLASSES, INK, MUTED, GRID

BATCH = 32

# Four families, three severities each. Severity 0 is the clean image, so every
# curve starts from the same point and the drop is directly readable.
SEVERITIES = [1, 2, 3]
PALETTE = {"gaussian_noise": "#2563eb", "jpeg": "#ea580c",
           "blur": "#0d9488", "darkness": "#9333ea"}


def perturb(arr: np.ndarray, family: str, severity: int) -> np.ndarray:
    """Apply one perturbation to a 0-255 float array of shape (H, W, 3)."""
    if severity == 0:
        return arr

    if family == "gaussian_noise":
        sigma = {1: 5.0, 2: 15.0, 3: 30.0}[severity]
        rng = np.random.default_rng(0)
        return np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255)

    if family == "jpeg":
        quality = {1: 60, 2: 30, 3: 15}[severity]
        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)

    if family == "blur":
        radius = {1: 1.0, 2: 2.0, 3: 3.0}[severity]
        img = Image.fromarray(arr.astype(np.uint8))
        return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)),
                          dtype=np.float32)

    if family == "darkness":
        factor = {1: 0.7, 2: 0.5, 3: 0.3}[severity]
        return np.clip(arr * factor, 0, 255)

    raise ValueError(family)


def plot(df, clean_f1, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ends = []
    for family, g in df[df.severity > 0].groupby("family"):
        g = g.sort_values("severity")
        xs = [0] + g.severity.tolist()
        ys = [clean_f1] + g.macro_f1.tolist()
        ax.plot(xs, ys, color=PALETTE[family], linewidth=2, marker="o",
                markersize=5, label=family.replace("_", " "))
        ends.append((ys[-1], xs[-1], family))

    # Curves often converge at the highest severity, so end labels would overlap.
    # Nudge them apart vertically, keeping their order, before drawing.
    span = max(df.macro_f1.max(), clean_f1) - df.macro_f1.min() or 1.0
    gap = span * 0.055
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i] = (ends[i - 1][0] + gap, ends[i][1], ends[i][2])
    for y, x, family in ends:
        ax.annotate(f" {family.replace('_', ' ')}", (x, y), fontsize=9,
                    color=MUTED, va="center")

    ax.axhline(clean_f1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"clean {clean_f1:.3f}", (0, clean_f1), xytext=(2, 6),
                textcoords="offset points", fontsize=9, color=MUTED)

    ax.set_xticks([0, 1, 2, 3], ["clean", "mild", "moderate", "severe"])
    ax.set_xlim(-0.1, 3.9)
    ax.set_ylabel("test macro F1")
    ax.set_title("Robustness to input degradation", color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    ap.add_argument("--roi", type=Path, default=Path("config/roi_regions.json"))
    ap.add_argument("--out", type=Path, default=None, help="default: next to the model")
    args = ap.parse_args()
    out = args.out or args.model.parent

    import keras
    from sklearn.metrics import f1_score, accuracy_score

    backbone_name, size, labels, roi_cfg = resolve_card(args.model, args.classes)
    preprocess_input = importlib.import_module(
        f"keras.applications.{BACKBONES[backbone_name][0]}").preprocess_input

    rois, letterbox = None, bool(roi_cfg.get("letterbox"))
    if roi_cfg.get("active"):
        if not args.roi.exists():
            raise SystemExit(f"ROI model needs polygons; not found at {args.roi}")
        rois = load_rois(args.roi)

    manifest = pd.read_csv(args.model.parent.parent / "manifest.csv")
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    y_true = test[f"class_{args.classes}"].map(
        {c: i for i, c in enumerate(labels)}).values

    model = keras.models.load_model(args.model)
    cases = [("clean", 0)] + [(f, s) for f in PALETTE for s in SEVERITIES]
    preds = {c: np.zeros(len(test), dtype=int) for c in cases}

    print(f"{len(test)} test images x {len(cases)} conditions")
    paths = test.path.tolist()
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        # Load and ROI-crop ONCE per image, then perturb in array space. Reloading
        # per condition would multiply the slowest part of the loop by thirteen.
        base = [load_image(str(p), size,
                           rois.get(camera_dir_from_path(p)) if rois else None,
                           letterbox)
                for p in chunk]
        for family, sev in cases:
            batch = np.stack([perturb(a, family, sev) for a in base])
            p = model.predict(preprocess_input(batch), verbose=0).argmax(1)
            preds[(family, sev)][i:i + len(chunk)] = p
        print(f"\r  {min(i + BATCH, len(paths))}/{len(paths)}", end="", flush=True)
    print()

    rows = []
    for (family, sev), yp in preds.items():
        rows.append({
            "family": family, "severity": sev,
            "accuracy": accuracy_score(y_true, yp),
            "macro_f1": f1_score(y_true, yp, average="macro", zero_division=0),
        })
    df = pd.DataFrame(rows)
    clean_f1 = float(df[df.family == "clean"].macro_f1.iloc[0])
    df["drop_vs_clean"] = clean_f1 - df.macro_f1

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "perturbation.csv", index=False)
    plot(df, clean_f1, out / "perturbation.png")

    worst = df[df.severity > 0].sort_values("macro_f1").iloc[0]
    summary = {
        "clean_macro_f1": clean_f1,
        "worst_macro_f1": float(worst.macro_f1),
        "worst_condition": f"{worst.family}@{int(worst.severity)}",
        "mean_drop_severe": float(df[df.severity == 3].drop_vs_clean.mean()),
    }
    (out / "perturbation_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nclean {clean_f1:.4f} | worst {worst.macro_f1:.4f} "
          f"({worst.family} severity {int(worst.severity)})")
    print(f"wrote {out}/perturbation.csv, perturbation.png, perturbation_summary.json")


if __name__ == "__main__":
    main()
