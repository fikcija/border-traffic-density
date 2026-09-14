"""
Stage 1: build manifest, mask burned-in overlays, cache frozen backbone features.

Run once. Everything downstream reads artifacts/ and never touches the images again.

    python src/prepare.py --data-dir data --out artifacts
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from roi import load_rois, apply_roi

# labeled:   CAMERA_DIR__CAMERA_DIR_YYYY_M_D_HH-MM-SS.jpg   (month/day NOT zero-padded)
# unlabeled: CAMERA_DIR__YYYY-MM-DD_HH-MM-SS.jpg
RE_LABELED = re.compile(
    r"^([A-Z\-]+)_([UI])__.*?_(\d{4})_(\d{1,2})_(\d{1,2})_(\d{2})-(\d{2})-(\d{2})\.jpg$")
RE_UNLABELED = re.compile(
    r"^([A-Z\-]+)_([UI])__(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.jpg$")

# Backbone registry. Each entry: (keras.applications module path, class name).
# The matching preprocess_input is pulled from the same module - it differs per
# family (MobileNet scales to [-1,1], ResNet does caffe-style mean subtraction,
# EfficientNet expects raw 0-255), and using the wrong one quietly ruins accuracy.
BACKBONES = {
    "mobilenetv2":     ("mobilenet_v2", "MobileNetV2", 224),
    "mobilenetv3":     ("mobilenet_v3", "MobileNetV3Large", 224),
    "efficientnetb0":  ("efficientnet", "EfficientNetB0", 224),
    "efficientnetv2b0": ("efficientnet_v2", "EfficientNetV2B0", 224),
    "resnet50":        ("resnet", "ResNet50", 224),
    "resnet50v2":      ("resnet_v2", "ResNet50V2", 224),
    "densenet121":     ("densenet", "DenseNet121", 224),
    "convnexttiny":    ("convnext", "ConvNeXtTiny", 224),
    "convnextsmall":   ("convnext", "ConvNeXtSmall", 224),
    "xception":        ("xception", "Xception", 299),
    "inceptionv3":     ("inception_v3", "InceptionV3", 299),
}

# Backbones that want a gentler fine-tuning learning rate than the 1e-4 default.
# ConvNeXt uses LayerNorm + LayerScale rather than BatchNorm and degrades quickly
# if the steps are too large.
FINETUNE_LR = {"convnexttiny": 1e-5, "convnextsmall": 1e-5}


def load_backbone(name, size):
    """Returns (model, preprocess_input) for a registry name."""
    import importlib
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone '{name}'. choose from: {', '.join(BACKBONES)}")
    module_name, cls_name, _ = BACKBONES[name]
    mod = importlib.import_module(f"keras.applications.{module_name}")
    model = getattr(mod, cls_name)(include_top=False, weights="imagenet",
                                   input_shape=(size, size, 3))
    return model, mod.preprocess_input


CLASS_4 = ["no_traffic", "low_traffic", "moderate_traffic", "high_traffic"]
# low + moderate collapse: the boundary between them is the subjective one
TO_3 = {"no_traffic": "no_traffic", "low_traffic": "light",
        "moderate_traffic": "light", "high_traffic": "high"}
CLASS_3 = ["no_traffic", "light", "high"]

# Overlay boxes as fractions of (H, W). Timestamp sits top-right, camera name
# bottom-left, in the same relative spot on every camera. The information is in
# the filename, so blanking these costs nothing and removes a shortcut the CNN
# would otherwise learn (date/hour correlates with class).
OVERLAY_BOXES = [
    (0.00, 0.11, 0.54, 1.00),   # top-right timestamp
    (0.91, 1.00, 0.00, 0.30),   # bottom-left camera label
]


def parse_name(name: str):
    m = RE_LABELED.match(name) or RE_UNLABELED.match(name)
    if not m:
        return None
    cam, direction, y, mo, d, H, M, S = m.groups()
    return cam, direction, pd.Timestamp(int(y), int(mo), int(d), int(H), int(M), int(S))


def build_manifest(data_dir: Path) -> pd.DataFrame:
    rows, skipped = [], []
    for cls in CLASS_4:
        folder = data_dir / "labeled" / cls
        if not folder.is_dir():
            sys.exit(f"missing folder: {folder}")
        for p in sorted(folder.glob("*.jpg")):
            parsed = parse_name(p.name)
            if parsed is None:
                skipped.append(p.name)
                continue
            cam, direction, ts = parsed
            rows.append({
                "path": str(p), "camera": cam, "direction": direction,
                "camera_dir": f"{cam}_{direction}", "timestamp": ts,
                "class_4": cls, "class_3": TO_3[cls],
            })
    if skipped:
        print(f"WARNING: {len(skipped)} filenames did not parse, e.g. {skipped[:3]}")
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    print(f"manifest: {len(df)} images, {df.camera_dir.nunique()} camera-directions, "
          f"{df.timestamp.min().date()} -> {df.timestamp.max().date()}")
    return df


def assign_splits(df: pd.DataFrame, val_start: str, test_start: str) -> pd.DataFrame:
    """Chronological, per the course guidance. Never random: 18.5% of consecutive
    frames from one camera are <5 min apart, so a random split leaks near-duplicates."""
    df = df.copy()
    df["split"] = "train"
    df.loc[df.timestamp >= pd.Timestamp(val_start), "split"] = "val"
    df.loc[df.timestamp >= pd.Timestamp(test_start), "split"] = "test"
    print("\nsplit sizes:")
    print(pd.crosstab(df.split, df.class_3))
    for s in ("train", "val", "test"):
        missing = set(CLASS_3) - set(df.loc[df.split == s, "class_3"])
        if missing:
            print(f"WARNING: split '{s}' is missing class(es) {missing}")
    return df


def load_image(path: str, size: int, roi=None, letterbox=False) -> np.ndarray:
    """Overlay-mask, optionally ROI-mask/crop, then resize.

    Overlay masking is kept even when ROI is active. The bundle's README says the
    polygon already excludes the overlay corners, but that was written for its own
    full-width top/bottom crop. Measured against OVERLAY_BOXES here, every polygon
    overlaps them - up to 14% of SPILJANI_I's ROI area - so skipping the overlay
    mask would leave the burned-in clock inside the crop, and date correlates with
    traffic density.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)   # np.asarray on a PIL image is read-only
    for y0, y1, x0, x1 in OVERLAY_BOXES:
        arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
    img = Image.fromarray(arr)
    if roi is not None:
        img = apply_roi(img, roi, letterbox=letterbox)
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: artifacts/<backbone>")
    ap.add_argument("--backbone", default="mobilenetv2", choices=list(BACKBONES))
    ap.add_argument("--size", type=int, default=None,
                    help="default: the backbone's native size")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--roi", type=Path, default=None,
                    help="path to roi_regions.json; enables per-camera ROI "
                         "masking and writes to artifacts/<backbone>_roi/")
    ap.add_argument("--roi-letterbox", action="store_true",
                    help="pad the ROI crop to a square before resize, so narrow "
                         "ROIs are not stretched; writes to <backbone>_roi_lb")
    ap.add_argument("--val-start", default="2024-06-01")
    ap.add_argument("--test-start", default="2025-01-01")
    args = ap.parse_args()
    if args.size is None:
        args.size = BACKBONES[args.backbone][2]
    if args.out is None:
        suffix = ("_roi_lb" if args.roi_letterbox else "_roi") if args.roi else ""
        args.out = Path("artifacts") / (args.backbone + suffix)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"backbone: {args.backbone} @ {args.size}px -> {args.out}")

    df = assign_splits(build_manifest(args.data_dir), args.val_start, args.test_start)

    rois = None
    if args.roi:
        rois = load_rois(args.roi)
        missing = sorted(set(df.camera_dir) - set(rois))
        if missing:
            sys.exit(f"no ROI polygon for camera(s): {missing}")
        print(f"\nROI masking ACTIVE - {len(rois)} polygons from {args.roi}")

    import keras
    base, preprocess_input = load_backbone(args.backbone, args.size)
    base.trainable = False
    # 7x7 -> 3x3 keeps coarse spatial layout (left/right lanes matter for direction)
    # while staying small enough to cache. Head can still average it down to 1x1.
    model = keras.Sequential([base, keras.layers.AveragePooling2D(pool_size=3, strides=2)])
    print(f"\nfeature shape: {model.output_shape[1:]}")

    feats = np.zeros((len(df), *model.output_shape[1:]), dtype=np.float16)
    for i in range(0, len(df), args.batch):
        chunk = df.path.iloc[i:i + args.batch]
        cams = df.camera_dir.iloc[i:i + args.batch]
        batch = np.stack([load_image(p, args.size, rois.get(c) if rois else None,
                                     args.roi_letterbox)
                          for p, c in zip(chunk, cams)])
        feats[i:i + args.batch] = model.predict(
            preprocess_input(batch), verbose=0).astype(np.float16)
        done = min(i + args.batch, len(df))
        print(f"\r  features {done}/{len(df)}", end="", flush=True)
    print()

    np.save(args.out / "features.npy", feats)
    df.to_csv(args.out / "manifest.csv", index=False)
    (args.out / "prepare_config.json").write_text(json.dumps({
        "size": args.size, "backbone": args.backbone,
        "feature_shape": list(model.output_shape[1:]),
        "val_start": args.val_start, "test_start": args.test_start,
        "overlay_boxes": OVERLAY_BOXES,
        "roi_path": str(args.roi) if args.roi else None,
        "roi_active": bool(args.roi),
        "roi_letterbox": bool(args.roi_letterbox),
    }, indent=2))
    print(f"\nwrote {args.out}/features.npy {feats.shape} ({feats.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
