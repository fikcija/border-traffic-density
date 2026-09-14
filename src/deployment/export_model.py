"""
Package a trained run into a self-contained, servable model file.

train.py saves only the head - its input is a cached 3x3xC feature tensor, not an
image, and the backbone weights live in Keras's global download cache, not in the
repo. So run_<n>cls/model.keras cannot be handed to anyone on its own.

    python src/export_model.py --artifacts artifacts/mobilenetv2_roi --run run_3cls

Writes <run>/model_full.keras (a preprocessed image -> class probabilities) and
<run>/model_card.json, which the register stage then logs to MLflow.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import load_backbone, load_image, OVERLAY_BOXES, BACKBONES
from train import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts/mobilenetv2"))
    ap.add_argument("--run", default="run_3cls")
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    ap.add_argument("--check", type=int, default=8,
                    help="how many test images to verify against cached features")
    args = ap.parse_args()

    import keras
    run_dir = args.artifacts / args.run
    cfg = json.loads((args.artifacts / "prepare_config.json").read_text())
    size, backbone_name = cfg["size"], cfg["backbone"]
    labels = CLASSES[args.classes]
    roi_path = cfg.get("roi_path")
    roi_active = bool(cfg.get("roi_active"))
    letterbox = bool(cfg.get("roi_letterbox"))

    # compile=False drops Adam's optimizer state. It is 2 extra float32 values per
    # trainable parameter - here ~30 MB of a 54 MB file - and it is only needed to
    # resume training, never to predict. The recipient does not want to download it.
    head = keras.models.load_model(run_dir / "model.keras", compile=False)
    already_full = tuple(head.input_shape[1:]) == (size, size, 3)
    out_path = run_dir / "model_full.keras"

    if already_full:
        print(f"{run_dir}/model.keras already takes images end-to-end; using it as-is")
        out_path = run_dir / "model.keras"
    else:
        base, preprocess_input = load_backbone(backbone_name, size)
        base.trainable = False
        pool = keras.layers.AveragePooling2D(pool_size=3, strides=2)
        inp = keras.layers.Input(shape=(size, size, 3), name="preprocessed_image")
        model = keras.Model(inp, head(pool(base(inp))),
                            name=f"{backbone_name}_{args.run}")
        model.save(out_path)
        print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, "
              f"{model.count_params():,} params)")

        if args.check:
            reloaded = keras.models.load_model(out_path)
            df = pd.read_csv(args.artifacts / "manifest.csv")
            X = np.load(args.artifacts / "features.npy")
            rois = None
            if roi_active:
                from roi import load_rois
                rois = load_rois(roi_path)
            sel = np.where((df.split == "test").values)[0][:args.check]
            imgs = np.stack([
                load_image(p, size, rois.get(c) if rois else None, letterbox)
                for p, c in zip(df.path.iloc[sel], df.camera_dir.iloc[sel])])
            a = head.predict(X[sel].astype(np.float32), verbose=0)
            b = reloaded.predict(preprocess_input(imgs), verbose=0)
            agree = int((a.argmax(1) == b.argmax(1)).sum())
            print(f"verification on {len(sel)} test images: max prob difference "
                  f"{float(np.abs(a-b).max()):.2e}, labels agree {agree}/{len(sel)}")
            if agree != len(sel):
                raise SystemExit("MISMATCH - exported model disagrees with the head")

    res = json.loads((run_dir / "results.json").read_text())
    steps = ["open RGB, black out the overlay boxes "
             f"{OVERLAY_BOXES} at native resolution"]
    if roi_active:
        steps.append("look up the camera's ROI polygon by the filename prefix, "
                     "grey-mask outside it (fill 124,116,104), crop to its bounding box"
                     + (", pad to square" if letterbox else ""))
    steps += [f"resize to {size}x{size} bilinear",
              f"apply keras.applications.{BACKBONES[backbone_name][0]}.preprocess_input"]

    card = {
        "model_file": out_path.name,
        "task": "border traffic density classification",
        "labels": labels,
        "output": "softmax probabilities, index order matches `labels`",
        "input": {"shape": [size, size, 3], "dtype": "float32",
                  "preprocessing": steps,
                  "warning": "the model does NOT preprocess internally - "
                             "use predict.py"},
        "roi": {"active": roi_active, "letterbox": letterbox,
                "regions_file": "roi_regions.json" if roi_active else None,
                "n_cameras": len(json.loads(Path(roi_path).read_text()))
                             if roi_active else 0,
                "camera_from": "filename prefix before '__'" if roi_active else None},
        "backbone": backbone_name,
        "backbone_weights": "imagenet (keras.applications default checkpoint)",
        "head_hyperparameters": res.get("winner", {}).get("hyperparameters"),
        "splits": {"val_start": cfg["val_start"], "test_start": cfg["test_start"]},
        "test_metrics": res["test"],
        "baselines": res["baselines"],
        "keras_version": keras.__version__,
    }
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {run_dir}/model_card.json  (roi_active={roi_active})")


if __name__ == "__main__":
    main()
