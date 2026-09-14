"""
REST serving for the border traffic density classifier.

Deployment pattern: dynamic deployment on a server, containerised, exposed over HTTP.
Serving mode: on-demand (synchronous, one request -> one prediction).

    uvicorn app:app --host 0.0.0.0 --port 8000

Environment (registry mode):
    MLFLOW_TRACKING_URI  e.g. http://mlflow:5000
    MODEL_NAME           registered model, e.g. border-traffic-density
    MODEL_ALIAS          which version to serve      (default production)
    MODEL_FILE           artifact to load            (default model_int8.tflite)

Environment (local mode, when MODEL_NAME is unset):
    MODEL_PATH  .tflite model                 (default model_int8.tflite)
    ROI_PATH    per-camera polygons           (default roi_regions.json)

Always:
    LABELS             comma-separated, index order  (default no_traffic,light,high)
    INPUT_SIZE         square input edge             (default 224)
    MAX_UPLOAD_BYTES   per-image cap                 (default 10485760)
    MAX_BATCH_FILES    images per batch request      (default 32)
    RELOAD_TOKEN       if set, POST /reload requires it in X-Reload-Token

See auth.py for JWT settings and db.py for storage.

Endpoints:
    GET  /health              open - the container healthcheck polls it
    POST /auth/register       open unless ALLOW_REGISTRATION=false
    POST /auth/login          open, rate limited hard
    POST /predict             bearer token
    POST /predict/batch       bearer token
    GET  /predictions         bearer token
    GET  /predictions/stats   bearer token
    POST /reload              open, or gated on RELOAD_TOKEN when that is set

The model is not in the image: it is pulled from the MLflow registry at startup,
falling back to a local file when no registry is configured.
"""
import hmac
import io
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Request,
                     Response, UploadFile, status)
from PIL import Image
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# The same preprocessing module training used - importing it rather than copying
# its logic is what keeps serving from drifting away from training.
from prepare import load_image
from roi import load_rois, camera_dir_from_path
import auth
import db
import registry

LABELS = os.getenv("LABELS", "no_traffic,light,high").split(",")
SIZE = int(os.getenv("INPUT_SIZE", "224"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "32"))
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN")

# Keyed on client IP. Behind a reverse proxy every caller collapses to one address,
# so the proxy has to do the limiting there.
limiter = Limiter(key_func=get_remote_address)


def _interpreter(path):
    """Load a TFLite interpreter, trying each runtime spelling in turn.

    tf.lite exposes Interpreter as an attribute, not an importable submodule.
    """
    Interpreter = None
    for mod, attr in (("tflite_runtime.interpreter", "Interpreter"),
                      ("ai_edge_litert.interpreter", "Interpreter"),
                      ("tensorflow", "lite")):
        try:
            import importlib
            m = importlib.import_module(mod)
            Interpreter = m.lite.Interpreter if attr == "lite" else getattr(m, attr)
            break
        except (ImportError, AttributeError):
            continue
    if Interpreter is None:
        raise RuntimeError("no TFLite runtime found - install tflite-runtime")
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def preprocess_input(x):
    """keras.applications.mobilenet_v2.preprocess_input, without importing Keras.

    That function is exactly `x / 127.5 - 1.0` on float 0-255 input.
    verify_preprocessing.py asserts this matches it on real images.
    """
    return x / 127.5 - 1.0


STATE = {}


def _load_model():
    """Resolve, download and load. Used at startup and by POST /reload."""
    resolved = registry.resolve()
    if not resolved.model_path.exists():
        raise RuntimeError(f"model not found: {resolved.model_path}")
    # An ROI-trained model is wrong on whole frames, so the polygons are required.
    if not resolved.roi_path.exists():
        raise RuntimeError(f"ROI polygons not found: {resolved.roi_path}")

    interp = _interpreter(resolved.model_path)
    STATE.update(
        interp=interp,
        **{"in": interp.get_input_details()[0], "out": interp.get_output_details()[0]},
        rois=load_rois(resolved.roi_path),
        resolved=resolved,
    )
    print(f"serving {resolved.describe()}, {len(STATE['rois'])} ROI polygons, "
          f"labels {LABELS}", flush=True)
    return resolved


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load once at startup, not per request.

    A missing model is a normal state on a fresh registry, not a crash: the first
    training run has not promoted anything yet. Exiting here would be retried three
    times by supervisord and then left FATAL, so the API would still be dead when
    the flow's reload_api fires - a promoted model that nothing serves. Starting
    degraded instead keeps /reload reachable, which is what recovers it.
    """
    db.init()
    try:
        _load_model()
    except Exception as e:
        print(f"WARNING: starting without a model ({e}). /health reports 503 and "
              f"prediction endpoints refuse until one is promoted; the retrain flow's "
              f"reload_api call will load it.", flush=True)
    yield
    STATE.clear()


app = FastAPI(title="Border traffic density", version="2.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ----------------------------------------------------------------- inference
def _quantize_in(x, detail):
    """Match the interpreter's expected input dtype.

    A no-op for float-input models such as the default model_int8.tflite, which
    quantises only its weights.
    """
    dtype = detail["dtype"]
    if dtype == np.float32:
        return x.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if not scale:
        raise HTTPException(500, f"model input is {dtype.__name__} but carries no "
                                 f"quantization parameters")
    info = np.iinfo(dtype)
    return np.clip(np.round(x / scale + zero_point), info.min, info.max).astype(dtype)


def _dequantize_out(y, detail):
    """Inverse of the above on the way back out."""
    if detail["dtype"] == np.float32:
        return y.astype(float)
    scale, zero_point = detail["quantization"]
    if not scale:
        return y.astype(float)
    return (y.astype(float) - zero_point) * scale


def _infer(raw: bytes, cam: str):
    """Decode, preprocess and run one image. Returns (probabilities, latency_ms)."""
    if not STATE.get("resolved"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no model loaded. The registry has no version carrying the "
            f"'{os.getenv('MODEL_ALIAS', 'production')}' alias yet - run the retrain "
            "flow, or POST /reload once one is promoted.")
    roi = STATE["rois"].get(cam)
    if roi is None:
        raise HTTPException(
            400, f"unknown camera {cam!r}. This model only covers "
                 f"{len(STATE['rois'])} configured cameras; pass one of them as the "
                 f"'camera' field. A new crossing needs its ROI polygon added first.")

    t0 = time.perf_counter()
    # Written to disk so the upload goes through load_image() unchanged.
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            tmp.write(raw)
            tmp.flush()
            arr = load_image(tmp.name, SIZE, roi, letterbox=True)
    except HTTPException:
        raise
    except Exception as e:
        # Image.verify() passes on files that still fail a full decode.
        raise HTTPException(400, f"could not decode image: {e}")

    interp, in_det, out_det = STATE["interp"], STATE["in"], STATE["out"]
    x = _quantize_in(preprocess_input(arr[None].astype(np.float32)), in_det)
    interp.set_tensor(in_det["index"], x)
    interp.invoke()
    probs = _dequantize_out(interp.get_tensor(out_det["index"])[0], out_det)

    return probs, (time.perf_counter() - t0) * 1000


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload, refusing anything over the cap.

    Reads one byte past the limit to detect an oversized body without holding it.
    """
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"image exceeds MAX_UPLOAD_BYTES ({MAX_UPLOAD_BYTES} bytes)")
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(400, "not a readable image")
    return raw


def _result(cam, probs, latency_ms):
    top = int(np.argmax(probs))
    return {
        "camera": cam,
        "prediction": LABELS[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {l: round(float(p), 4) for l, p in zip(LABELS, probs)},
        # End to end: decode + overlay mask + ROI crop + resize + inference.
        "latency_ms": round(latency_ms, 1),
    }


def _record(result, username, filename=None, batch_id=None):
    """Log a served prediction. A logging failure must not break the response."""
    resolved = STATE.get("resolved")
    try:
        db.log_prediction(
            username=username, camera=result["camera"],
            prediction=result["prediction"], confidence=result["confidence"],
            probabilities=result["probabilities"], latency_ms=result["latency_ms"],
            model_version=resolved.version if resolved else None,
            model_source=resolved.source if resolved else None,
            filename=filename, batch_id=batch_id)
    except Exception as e:
        print(f"WARNING: could not log prediction: {e}", flush=True)


# --------------------------------------------------------------------- schemas
class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


# ------------------------------------------------------------------- endpoints
@app.get("/health")
def health(response: Response):
    """Open - docker's healthcheck and healthcheck.sh both poll it.

    503 until a model is loaded, so an API that cannot serve reads as unhealthy.
    """
    resolved = STATE.get("resolved")
    if resolved is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading", "model": None, "cameras": [], "labels": LABELS}
    return {"status": "ok", "model": resolved.describe(),
            "cameras": sorted(STATE.get("rois", {})), "labels": LABELS}


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def register(request: Request, creds: Credentials):
    if not auth.ALLOW_REGISTRATION:
        raise HTTPException(403, "registration is closed")
    try:
        db.create_user(creds.username, auth.hash_password(creds.password))
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"username {creds.username!r} is taken")
    return {"username": creds.username, **auth.create_token(creds.username)}


@app.post("/auth/login")
@limiter.limit("10/minute")
def login(request: Request, creds: Credentials):
    user = db.get_user(creds.username)
    # One message for both "no such user" and "wrong password", so the response
    # does not reveal which usernames exist.
    if user is None or not auth.verify_password(creds.password, user["password_hash"]):
        raise HTTPException(401, "invalid username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return {"username": creds.username, **auth.create_token(creds.username)}


@app.get("/auth/me")
def me(username: str = Depends(auth.current_user)):
    return {"username": username}


@app.post("/predict")
@limiter.limit("60/minute")
async def predict(request: Request, file: UploadFile = File(...),
                  camera: str = Form(None),
                  username: str = Depends(auth.current_user)):
    """One image in, class probabilities out.

    `camera` selects the ROI polygon; omitted, it is read from the filename prefix
    before '__'. There is no default - the wrong polygon yields a confident wrong
    answer rather than an error.
    """
    raw = await _read_capped(file)
    cam = camera or camera_dir_from_path(file.filename or "")
    probs, latency_ms = _infer(raw, cam)
    result = _result(cam, probs, latency_ms)
    _record(result, username, filename=file.filename)
    return result


@app.post("/predict/batch")
@limiter.limit("10/minute")
async def predict_batch(request: Request, files: list[UploadFile] = File(...),
                        camera: str = Form(None),
                        username: str = Depends(auth.current_user)):
    """Several images in one request.

    Each item carries either a prediction or an error, so one corrupt file does not
    fail the batch; the status stays 200 as long as the request itself was valid.
    `camera` applies to every image when given, otherwise each filename supplies
    its own.
    """
    if not files:
        raise HTTPException(400, "no files uploaded")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{len(files)} images exceeds MAX_BATCH_FILES ({MAX_BATCH_FILES})")

    batch_id = uuid.uuid4().hex[:12]
    results, ok = [], 0
    for i, file in enumerate(files):
        try:
            raw = await _read_capped(file)
            cam = camera or camera_dir_from_path(file.filename or "")
            probs, latency_ms = _infer(raw, cam)
            result = _result(cam, probs, latency_ms)
            _record(result, username, filename=file.filename, batch_id=batch_id)
            results.append({"index": i, "filename": file.filename, **result})
            ok += 1
        except HTTPException as e:
            results.append({"index": i, "filename": file.filename,
                            "error": e.detail, "status": e.status_code})
    return {"batch_id": batch_id, "total": len(files), "succeeded": ok,
            "failed": len(files) - ok, "results": results}


@app.get("/predictions")
@limiter.limit("30/minute")
def predictions(request: Request, limit: int = 50, offset: int = 0,
                camera: str = None, prediction: str = None,
                username: str = Depends(auth.current_user)):
    """Recent served predictions, newest first. Shared across users."""
    limit = max(1, min(limit, 500))
    return {"limit": limit, "offset": offset,
            "predictions": db.recent_predictions(limit, offset, camera, prediction)}


@app.get("/predictions/stats")
@limiter.limit("30/minute")
def predictions_stats(request: Request, username: str = Depends(auth.current_user)):
    """Class distribution, confidence and latency over everything served so far.

    Training mix for comparison: roughly 41% no_traffic / 41% light / 18% high.
    """
    return db.prediction_stats()


@app.post("/reload")
def reload_model(x_reload_token: str = Header(None)):
    """Re-resolve the production alias without rebuilding or restarting.

    Called by the orchestrator after it moves the alias; rollback is the same call.
    Open while RELOAD_TOKEN is unset.
    """
    if RELOAD_TOKEN:
        if not x_reload_token or not hmac.compare_digest(x_reload_token, RELOAD_TOKEN):
            raise HTTPException(401, "invalid or missing X-Reload-Token")
    try:
        resolved = _load_model()
    except Exception as e:
        raise HTTPException(503, f"reload failed, previous model still serving: {e}")
    return {"reloaded": True, "model": resolved.describe()}
