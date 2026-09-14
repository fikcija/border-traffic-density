# Serving

`POST /predict` takes a border-camera image and returns the density class with all
three probabilities. Runs in a container on `tflite-runtime` — **no TensorFlow**, so
the image is ~200 MB instead of ~1 GB.

| MLOps requirement | How it is satisfied |
|---|---|
| Deployment pattern | Dynamic deployment on a server, containerised, exposed as a REST API |
| Compression technique | Post-training quantization — the container serves `model_int8.tflite` (6.4 MB, 74% smaller and 3.7× faster than the Keras float32 model) |
| Serving mode | On-demand — synchronous, one request → one prediction |

## Build and run

```bash
bash src/serve/build.sh --run          # assembles context, builds, runs on :8000
```

`build.sh` copies `prepare.py` and `roi.py` **verbatim** from `src/`, plus
`artifacts/model_int8.tflite` and `config/roi_regions.json`, into the build context.

## Endpoints

| endpoint | auth | limit | what it does |
|---|---|---|---|
| `GET /health` | open | — | liveness; 503 until a model is loaded |
| `POST /auth/register` | open¹ | 5/min | create an account, returns a token |
| `POST /auth/login` | open | 10/min | exchange credentials for a token |
| `GET /auth/me` | bearer | — | who the token belongs to |
| `POST /predict` | bearer | 60/min | one image → one prediction |
| `POST /predict/batch` | bearer | 10/min | up to `MAX_BATCH_FILES` images |
| `GET /predictions` | bearer | 30/min | recent predictions, newest first |
| `GET /predictions/stats` | bearer | 30/min | class mix, confidence, latency |
| `POST /reload` | open² | — | re-resolve the `production` alias |

¹ closed by setting `ALLOW_REGISTRATION=false`.
² gated on an `X-Reload-Token` header when `RELOAD_TOKEN` is set. Left open by default
so the orchestrator's `reload_api` task works untouched; set the variable on both sides
to close it.

`/health` is deliberately open — both Docker's healthcheck and `healthcheck.sh` poll it,
and an endpoint that reports whether the service is up is not a secret.

## Use

```bash
curl -s localhost:8000/health
```

Register once, then keep the token:

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"marko","password":"a-good-long-password"}' | jq -r .access_token)
```

```bash
curl -s -X POST localhost:8000/predict \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@data/unlabeled/BATROVCI_I__2023-12-23_14-02-17.jpg" \
     -F "camera=BATROVCI_I"
```

```json
{
  "camera": "BATROVCI_I",
  "prediction": "no_traffic",
  "confidence": 0.9961,
  "probabilities": {"no_traffic": 0.9961, "light": 0.0, "high": 0.0},
  "latency_ms": 78.7
}
```

`camera` may be omitted if the filename follows the dataset convention
(`CAMERA_DIR__...`), from which it is read.

Several images at once — `camera` is optional here too, and when omitted each filename
supplies its own, which is what makes a mixed-camera upload work:

```bash
curl -s -X POST localhost:8000/predict/batch \
     -H "Authorization: Bearer $TOKEN" \
     -F "files=@data/unlabeled/HORGOS_I__a.jpg" \
     -F "files=@data/unlabeled/RACA_U__b.jpg"
```

```json
{"batch_id": "6f9eece72462", "total": 2, "succeeded": 2, "failed": 0,
 "results": [{"index": 0, "camera": "HORGOS_I", "prediction": "high", "confidence": 0.9922},
             {"index": 1, "camera": "RACA_U", "prediction": "no_traffic", "confidence": 0.332}]}
```

## API schema

```bash
python src/serve/export_api_schema.py          # needs the stack running
```

| file | import as |
|---|---|
| `api/openapi.json` | OpenAPI 3.1 — Postman *APIs → Import*, or any OpenAPI tool |
| `api/postman_collection.json` | Postman collection v2.1 — *Collections → Import* |

Both are **generated from the live app**, not written by hand — the same reason the
container imports `load_image` rather than reimplementing it. A hand-maintained second
description of the API drifts silently. Re-run the exporter after changing endpoints.

FastAPI's `/openapi.json` is the source of truth; the exporter adds only what the
generator cannot know: the server URL, the rate limits (they live in decorators, not in
signatures) and response examples (the endpoints return plain dicts, so there is no
response model to read).

The collection is the friendlier of the two to click around in: **Register** and
**Login** carry a test script that saves `access_token` into the collection's `token`
variable, and collection-level bearer auth means every other request then just works.
Set `baseUrl`, `username` and `password` in the collection variables first. For
`/predict`, pick a file in the form-data `file` row — Postman cannot store the file
itself in a collection.

**Importing the collection consumes the file.** Postman's current format stores a
collection as a *directory* of `.request.yaml` files, so after importing,
`api/postman_collection.json` is replaced by `api/Border traffic density/`. That is
Postman working normally, not a lost file — the requests, variables, bearer auth and
the token-capture script are all in there. Re-run the exporter to get the single-file
version back; it will be converted again on the next import.

Import **one or the other**, not both. Generating requests from `openapi.json` gives no
collection variables, no pre-wired bearer auth and no token capture, so the token has to
be pasted by hand after every expiry. Keep `openapi.json` for non-Postman consumers —
codegen, docs renderers, contract tests.

## Auth, limits and the prediction log

**JWT, with `hashlib.scrypt` for passwords.** scrypt is a memory-hard KDF in the
standard library, so password hashing adds no dependency to an image that must build on
aarch64. Tokens are signed with PyJWT rather than hand-rolled HMAC: signing a JWT
correctly is easy, but verifying one — rejecting `alg: none`, avoiding algorithm
confusion, comparing in constant time — is where the subtle holes are, and that is not a
good place to save a dependency.

**`JWT_SECRET` has no default.** Unset, the API generates an ephemeral key at startup
and says so in the log; tokens then stop working across a restart, which is visible and
harmless. A hardcoded fallback would instead ship one forgeable secret to every
deployment that forgot to override it.

**Rate limits are keyed on client IP.** That is the right granularity for the case that
actually matters — slowing credential guessing against `/auth/login`, before there is a
token to key on. Behind a reverse proxy every caller collapses to one address, so a
deployment that fronts this service needs the proxy to do the limiting.

**Every served prediction is logged** to SQLite (`SERVE_DB_PATH`, its own volume since
`data/` is mounted read-only). This is what turns the robustness numbers the training
pipeline measures into something observable in production: the model was fitted on
roughly 41% `no_traffic` / 41% `light` / 18% `high`, so a production mix far from that
in `/predictions/stats` is either genuine traffic change or drift worth investigating.
A logging failure is caught and warned about rather than turning a correct prediction
into a 500.

**A batch is partially fault-tolerant.** One corrupt file reports an error in its own
slot and the rest still score; the request stays 200 as long as the request itself was
valid. Scoring a backlog of frames should not be defeated by one bad file halfway
through.

## Design decisions

**Preprocessing is imported, not reimplemented.** The service calls the same
`load_image()` from `prepare.py` that produced the training features — it writes the
upload to a temp file specifically so the request goes through that exact function.
This is the training/serving skew problem from the lecture: a parallel copy of the
preprocessing drifts, and the served model then silently disagrees with its own
evaluation.

The one exception is `preprocess_input`, reimplemented as `x / 127.5 - 1.0` so the
container does not need Keras. That reimplementation is **verified, not trusted**:

```bash
python src/serve/verify_preprocessing.py --dir data/unlabeled --n 20
# checked 20 images, max abs difference: 0.000e+00
# PASS: container preprocessing matches training
```

Run it before building. It exits non-zero if the two ever diverge.

**An unknown camera is an error, not a guess.** The model is ROI-masked per camera and
is *wrong* on whole frames, so a request naming a camera without a polygon returns 400
rather than a confident answer. A new crossing requires its ROI polygon to be added —
which is how such a system is really deployed.

**Model and polygons load once at startup**, via the lifespan handler, not per request.
Missing files fail the container immediately rather than at first traffic.

## Known properties worth stating

- **int8 output probabilities are coarse.** They come in steps of 1/256 ≈ 0.0039 —
  the `0.9961` above is exactly `255/256`. Fine for `argmax`, but a decision-threshold
  on `p_high` would want the float16 model, whose probabilities are continuous.
- **`latency_ms` is end to end** — decode, overlay mask, ROI crop, resize, inference.
  Inference alone is ~3 ms; preprocessing dominates.
- **`pandas` is in the image** only because `prepare.py` imports it at module level for
  the training paths. Making that import lazy would shrink the image further; it was
  left alone so the serving container runs byte-identical code to training.
- **Uploads are capped** at `MAX_UPLOAD_BYTES` (10 MiB) per image and `MAX_BATCH_FILES`
  (32) per batch. The size check runs before the decode, so an oversized body is
  refused with 413 without ever being held whole in memory.
- **The prediction log is shared, not per-user.** It exists to observe what the model
  is doing in production, which is not a per-account view.
- **The quantised-input path is written but unexercised.** `model_int8.tflite` as the
  pipeline produces it has float32 in and out — only the weights are quantised — so
  `_quantize_in` / `_dequantize_out` are no-ops on the default artifact. They exist so
  that pointing `MODEL_FILE` at a fully integer-quantised model gives a correct answer
  rather than an opaque `set_tensor` dtype error, but that path has not been run
  against such a model.
