"""
Guard against training/serving skew.

The container runs tflite-runtime and has no Keras, so `preprocess_input` is
reimplemented in app.py as plain numpy. This asserts that reimplementation matches
the Keras function training actually used, on real images.

Run it in the dev environment (where Keras IS installed), before building the image:

    python src/serve/verify_preprocessing.py --dir data/unlabeled --n 20

Exit code 0 means the container preprocesses exactly as training did.
"""
import argparse, sys
from pathlib import Path

import numpy as np

from prepare import load_image
from roi import load_rois, camera_dir_from_path
from app import preprocess_input as container_preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/unlabeled"))
    ap.add_argument("--roi", type=Path, default=Path("config/roi_regions.json"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    from keras.applications.mobilenet_v2 import preprocess_input as keras_preprocess

    rois = load_rois(args.roi)
    paths = sorted(args.dir.glob("*.jpg"))[:args.n]
    if not paths:
        sys.exit(f"no images in {args.dir}")

    worst = 0.0
    for p in paths:
        roi = rois.get(camera_dir_from_path(p))
        if roi is None:
            continue
        arr = load_image(str(p), args.size, roi, letterbox=True)[None]
        a = keras_preprocess(arr.copy())
        b = container_preprocess(arr.copy().astype(np.float32))
        worst = max(worst, float(np.abs(np.asarray(a) - b).max()))

    print(f"checked {len(paths)} images, max abs difference: {worst:.3e}")
    if worst > 1e-6:
        sys.exit("FAIL: container preprocessing differs from training preprocessing")
    print("PASS: container preprocessing matches training")


if __name__ == "__main__":
    main()
