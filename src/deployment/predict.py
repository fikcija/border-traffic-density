"""
Run a saved model on image files. Handles all preprocessing, so this is the entry
point to hand to someone else along with the model.

    python src/predict.py --model artifacts/mobilenetv2/run_3cls/model_full.keras \
                          data/unlabeled/*.jpg
    python src/predict.py --model artifacts/mobilenetv2/finetune_3cls/model.keras \
                          --dir data/unlabeled --csv predictions.csv

Works with both export types: model_full.keras (frozen run, merged by
export_model.py) and finetune_*/model.keras (already end-to-end).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

from prepare import load_image, BACKBONES
from roi import load_rois, camera_dir_from_path
from train import CLASSES


def resolve_card(model_path: Path, classes: int):
    """Backbone/size/labels/ROI settings, from the model card or the run configs."""
    card = model_path.parent / "model_card.json"
    if card.exists():
        c = json.loads(card.read_text())
        return c["backbone"], c["input"]["shape"][0], c["labels"], (c.get("roi") or {})
    # fine-tuned runs have no card; fall back to the backbone's prepare_config
    cfg_path = model_path.parent.parent / "prepare_config.json"
    if not cfg_path.exists():
        sys.exit(f"cannot determine backbone: no {card} and no {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    roi = {"active": bool(cfg.get("roi_active")),
           "letterbox": bool(cfg.get("roi_letterbox")),
           "regions_file": cfg.get("roi_path")}
    return cfg["backbone"], cfg["size"], CLASSES[classes], roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path, help="predict on every .jpg in this folder")
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--csv", type=Path, help="also write results here")
    ap.add_argument("--roi", type=Path, default=None,
                    help="override the ROI polygon file (default: the one named "
                         "in the model card, alongside the model)")
    args = ap.parse_args()

    paths = list(args.images)
    if args.dir:
        paths += sorted(args.dir.glob("*.jpg"))
    if not paths:
        sys.exit("no images given - pass paths or --dir")

    import importlib, keras
    backbone_name, size, labels, roi_cfg = resolve_card(args.model, args.classes)

    # An ROI-trained model is WRONG on whole frames, so refuse rather than guess.
    rois, letterbox = None, bool(roi_cfg.get("letterbox"))
    if roi_cfg.get("active"):
        rp = args.roi or (args.model.parent / (roi_cfg.get("regions_file")
                                               or "roi_regions.json"))
        if not Path(rp).exists():
            sys.exit(f"this model was trained with ROI masking but the polygon file "
                     f"is missing: {rp}\npass --roi <roi_regions.json>")
        rois = load_rois(rp)
        unknown = sorted({camera_dir_from_path(p) for p in paths} - set(rois))
        if unknown:
            sys.exit(f"no ROI polygon for camera(s) {unknown}. The camera is read "
                     f"from the filename prefix before '__'; this model can only "
                     f"score the {len(rois)} cameras in {rp}.")
        print(f"ROI masking ACTIVE - {len(rois)} polygons from {rp}"
              + (" (letterboxed)" if letterbox else ""))
    # only the preprocess function is needed - the backbone weights are already
    # inside the saved model, so do not rebuild it (that would force a download)
    preprocess_input = importlib.import_module(
        f"keras.applications.{BACKBONES[backbone_name][0]}").preprocess_input
    model = keras.models.load_model(args.model)
    print(f"{args.model.name}: {backbone_name} @ {size}px, {len(paths)} images")

    rows = []
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        batch = np.stack([
            load_image(str(p), size,
                       rois.get(camera_dir_from_path(p)) if rois else None,
                       letterbox)
            for p in chunk])
        probs = model.predict(preprocess_input(batch), verbose=0)
        for p, pr in zip(chunk, probs):
            rows.append({"path": str(p), "prediction": labels[int(pr.argmax())],
                         "confidence": float(pr.max()),
                         **{f"p_{l}": float(v) for l, v in zip(labels, pr)}})
        print(f"\r  {min(i+args.batch, len(paths))}/{len(paths)}", end="", flush=True)
    print()

    for r in rows[:20]:
        print(f"  {Path(r['path']).name:55s} {r['prediction']:12s} {r['confidence']:.3f}")
    if len(rows) > 20:
        print(f"  ... {len(rows)-20} more")

    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
