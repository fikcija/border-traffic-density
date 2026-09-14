"""
Per-camera region-of-interest masking, ported from the partner's PyTorch bundle
so it runs inside this repo's PIL/Keras pipeline (no torch dependency).

The polygon selects the road; everything outside it is filled with the ImageNet
mean colour - grey rather than black, so the mask edge is not a hard high-contrast
artifact and the masked region sits near zero after preprocess_input. The image is
then cropped to the polygon's bounding box, so the monitored lanes fill the frame
instead of occupying a corner of it.

Runs BEFORE resize/preprocess, and must run identically at inference time - a model
trained on ROI crops is wrong on whole frames.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

# ImageNet mean in 0-255 pixel space.
ROI_FILL = (124, 116, 104)


def load_rois(roi_path):
    """{camera_dir: roi_dict} from roi_regions.json."""
    return {r["cam"]: r for r in json.loads(Path(roi_path).read_text())}


def polygon_for(roi, w, h):
    """Pick the polygon matching this image's resolution.

    KELEBIJA_U was physically relocated, so it carries `variants` keyed by native
    resolution; every other camera has a single polygon. Falls back to the
    top-level polygon_norm when no variant matches.
    """
    for v in roi.get("variants") or ():
        if v["native_w"] == w and v["native_h"] == h:
            return v["polygon_norm"]
    return roi["polygon_norm"]


def pad_to_square(img, fill=ROI_FILL):
    """Centre the image on a square canvas so a later resize cannot distort it.

    ROI bounding boxes vary from 0.50 (VRSKA-CUKA_U, a narrow strip) to 1.53
    (KOTROMAN_U) in aspect ratio. Resizing those straight to 224x224 stretches the
    narrow ones 2x along one axis, which changes apparent vehicle shape and spacing
    per camera. Padding first keeps geometry consistent across cameras.
    """
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def apply_roi(img, roi, letterbox=False):
    """Grey-mask outside the polygon, then crop to its bounding box + margin.

    With letterbox=True the crop is padded to a square before it is returned, so
    the downstream resize preserves aspect ratio.
    """
    w, h = img.size
    pts = [(x * w, y * h) for x, y in polygon_for(roi, w, h)]

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    img = Image.composite(img, Image.new("RGB", (w, h), ROI_FILL), mask)

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    m = roi.get("margin", 0.0) or 0.0
    mx, my = (x1 - x0) * m, (y1 - y0) * m
    img = img.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)),
                    min(w, int(x1 + mx)), min(h, int(y1 + my))))
    return pad_to_square(img) if letterbox else img


def camera_dir_from_path(path):
    """'.../GRADINA_U__GRADINA_U_2024_1_2_03-04-05.jpg' -> 'GRADINA_U'.

    ROI lookup is keyed on the camera, and the only place the camera is recorded
    is the filename prefix before the double underscore. Images renamed away from
    that convention cannot be ROI-processed.
    """
    return Path(path).name.split("__", 1)[0]
