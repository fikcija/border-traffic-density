"""
Export the serving API schema for Postman (and anything else that reads OpenAPI).

    python src/serve/export_api_schema.py                    # against localhost:8000
    python src/serve/export_api_schema.py --url http://host:8000 --out api

Writes two files:

    api/openapi.json            OpenAPI 3.1 - import this as an API definition
    api/postman_collection.json Postman collection v2.1 - import this to click around

Generated from the live app so it cannot drift - re-run after changing endpoints.
FastAPI's /openapi.json is the source of truth; this adds what it cannot know: the
server URL, the rate limits (they live in decorators) and example responses (the
endpoints return plain dicts, so there is no response model to read).
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Rate limits are slowapi decorators, invisible to the schema generator.
RATE_LIMITS = {
    ("post", "/auth/register"): "5/minute",
    ("post", "/auth/login"): "10/minute",
    ("post", "/predict"): "60/minute",
    ("post", "/predict/batch"): "10/minute",
    ("get", "/predictions"): "30/minute",
    ("get", "/predictions/stats"): "30/minute",
}

EXAMPLES = {
    ("get", "/health"): {
        "status": "ok",
        "model": {"source": "registry", "version": "2",
                  "run_id": "6f22630c318d4d10831d205f0c9d2965",
                  "model": "model_int8.tflite"},
        "cameras": ["BATROVCI_I", "BATROVCI_U", "HORGOS_I", "RACA_U"],
        "labels": ["no_traffic", "light", "high"],
    },
    ("post", "/auth/register"): {
        "username": "marko", "access_token": "eyJhbGciOiJIUzI1NiIs...",
        "token_type": "bearer", "expires_at": "2026-09-14T17:00:00+00:00",
        "expires_in": 3600,
    },
    ("post", "/auth/login"): {
        "username": "marko", "access_token": "eyJhbGciOiJIUzI1NiIs...",
        "token_type": "bearer", "expires_at": "2026-09-14T17:00:00+00:00",
        "expires_in": 3600,
    },
    ("get", "/auth/me"): {"username": "marko"},
    ("post", "/predict"): {
        "camera": "HORGOS_I", "prediction": "high", "confidence": 0.9922,
        "probabilities": {"no_traffic": 0.0, "light": 0.0078, "high": 0.9922},
        "latency_ms": 13.4,
    },
    ("post", "/predict/batch"): {
        "batch_id": "6f9eece72462", "total": 2, "succeeded": 1, "failed": 1,
        "results": [
            {"index": 0, "filename": "HORGOS_I__a.jpg", "camera": "HORGOS_I",
             "prediction": "high", "confidence": 0.9922,
             "probabilities": {"no_traffic": 0.0, "light": 0.0078, "high": 0.9922},
             "latency_ms": 12.1},
            {"index": 1, "filename": "broken.jpg",
             "error": "not a readable image", "status": 400},
        ],
    },
    ("get", "/predictions"): {
        "limit": 2, "offset": 0,
        "predictions": [{
            "id": 5, "ts": "2026-09-14T15:58:12+00:00", "username": "marko",
            "camera": "RACA_U", "prediction": "no_traffic", "confidence": 0.332,
            "probabilities": {"no_traffic": 0.332, "light": 0.3359, "high": 0.332},
            "latency_ms": 6.2, "model_version": "2", "model_source": "registry",
            "filename": "RACA_U__b.jpg", "batch_id": "6f9eece72462"}],
    },
    ("get", "/predictions/stats"): {
        "total": 5, "mean_confidence": 0.8609, "mean_latency_ms": 7.5,
        "first_seen": "2026-09-14T15:58:10+00:00",
        "last_seen": "2026-09-14T15:58:12+00:00",
        "by_class": [
            {"prediction": "high", "n": 4, "mean_confidence": 0.9931, "share": 0.8},
            {"prediction": "no_traffic", "n": 1, "mean_confidence": 0.332, "share": 0.2}],
        "by_camera": [{"camera": "HORGOS_I", "n": 3, "mean_confidence": 0.9935},
                      {"camera": "RACA_U", "n": 2, "mean_confidence": 0.6621}],
    },
    ("post", "/reload"): {
        "reloaded": True,
        "model": {"source": "registry", "version": "3",
                  "run_id": "1dc3d1ad05a745ebaf9927f6dcfc1582",
                  "model": "model_int8.tflite"},
    },
}

DESCRIPTION = """\
Border traffic density classifier - REST serving.

Classifies a border-camera image as `no_traffic`, `light` or `high`. The model is
resolved from the MLflow registry at startup (whichever version carries the
`production` alias), so deploying a new one is an alias change plus `POST /reload`.

**Authentication.** Register or log in, then send the returned JWT as
`Authorization: Bearer <token>`. Tokens expire after `JWT_TTL_MINUTES` (default 60).

**Cameras.** The model is ROI-masked per camera and is *wrong* on whole frames, so
`camera` must name one of the configured crossings - `GET /health` lists them. It may
be omitted when the filename follows the `CAMERA_DIR__...` dataset convention.

**Rate limits** are per client IP; the limit is noted on each endpoint.
"""


def fetch(url: str) -> dict:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/openapi.json", timeout=20) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"could not reach the API at {url} ({e}).\n"
                 f"Start the stack first: docker compose -f docker/compose.yaml up -d")


def enrich(spec: dict, url: str) -> dict:
    spec["servers"] = [{"url": url.rstrip("/"), "description": "local stack"}]
    spec["info"]["description"] = DESCRIPTION

    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            key = (method.lower(), path)

            if key in RATE_LIMITS:
                note = f"\n\n**Rate limit:** {RATE_LIMITS[key]} per client IP."
                op["description"] = op.get("description", "") + note

            example = EXAMPLES.get(key)
            if example:
                for code in ("200", "201"):
                    resp = op.get("responses", {}).get(code)
                    if resp is not None:
                        resp.setdefault("content", {}).setdefault(
                            "application/json", {})["example"] = example
    return spec


# ------------------------------------------------------------------- postman
def _url(path: str, query=None):
    parts = [p for p in path.strip("/").split("/") if p]
    u = {"raw": "{{baseUrl}}" + path, "host": ["{{baseUrl}}"], "path": parts}
    if query:
        u["query"] = query
        # Postman shows `raw` as the request URL, so disabled params must not
        # appear there.
        active = [q for q in query if not q.get("disabled")]
        if active:
            u["raw"] += "?" + "&".join(f"{q['key']}={q['value']}" for q in active)
    return u


# Captures the token so every later request inherits it.
CAPTURE_TOKEN = [
    "const b = pm.response.json();",
    "if (b.access_token) {",
    "  pm.collectionVariables.set('token', b.access_token);",
    "  console.log('token saved, expires in ' + b.expires_in + 's');",
    "}",
]


def _req(name, method, path, *, auth=True, body=None, formdata=None, query=None,
         headers=None, scripts=None, description=""):
    request = {"method": method, "header": headers or [], "url": _url(path, query),
               "description": description}
    if not auth:
        request["auth"] = {"type": "noauth"}
    if body is not None:
        request["header"] = [{"key": "Content-Type", "value": "application/json"}]
        request["body"] = {"mode": "raw", "raw": json.dumps(body, indent=2),
                           "options": {"raw": {"language": "json"}}}
    if formdata is not None:
        request["body"] = {"mode": "formdata", "formdata": formdata}
    item = {"name": name, "request": request}
    if scripts:
        item["event"] = [{"listen": "test",
                          "script": {"type": "text/javascript", "exec": scripts}}]
    return item


def build_collection(url: str) -> dict:
    creds = {"username": "{{username}}", "password": "{{password}}"}
    return {
        "info": {
            "name": "Border traffic density",
            "description": DESCRIPTION,
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        # Every request inherits this; health and the auth routes opt out.
        "auth": {"type": "bearer",
                 "bearer": [{"key": "token", "value": "{{token}}", "type": "string"}]},
        "variable": [
            {"key": "baseUrl", "value": url.rstrip("/")},
            {"key": "username", "value": "marko"},
            {"key": "password", "value": "change-me-a-long-password"},
            {"key": "token", "value": ""},
            # Only used when the server sets RELOAD_TOKEN; its header ships disabled.
            {"key": "reloadToken", "value": ""},
        ],
        "item": [
            _req("Health", "GET", "/health", auth=False,
                 description="Open. 503 until a model is loaded. Lists the cameras "
                             "this model covers."),
            {"name": "Auth", "item": [
                _req("Register", "POST", "/auth/register", auth=False, body=creds,
                     scripts=CAPTURE_TOKEN,
                     description="Creates an account and returns a token, which this "
                                 "request saves to the `token` variable. 409 if the "
                                 "name is taken. 5/minute."),
                _req("Login", "POST", "/auth/login", auth=False, body=creds,
                     scripts=CAPTURE_TOKEN,
                     description="Exchanges credentials for a token and saves it to "
                                 "the `token` variable. 10/minute."),
                _req("Me", "GET", "/auth/me",
                     description="Who the current token belongs to."),
            ]},
            {"name": "Predict", "item": [
                _req("Predict", "POST", "/predict", formdata=[
                    {"key": "file", "type": "file", "src": [],
                     "description": "A border-camera JPEG. Max MAX_UPLOAD_BYTES "
                                    "(10 MiB)."},
                    {"key": "camera", "value": "HORGOS_I", "type": "text",
                     "description": "Optional when the filename is CAMERA_DIR__...",
                     "disabled": True}],
                    description="One image, one prediction. 60/minute."),
                _req("Predict batch", "POST", "/predict/batch", formdata=[
                    {"key": "files", "type": "file", "src": []},
                    {"key": "files", "type": "file", "src": []},
                    {"key": "camera", "value": "HORGOS_I", "type": "text",
                     "description": "Applies to every image when set; otherwise each "
                                    "filename supplies its own.",
                     "disabled": True}],
                    description="Up to MAX_BATCH_FILES (32) images. A bad file fails "
                                "in its own slot rather than the whole batch, so the "
                                "status stays 200. 10/minute."),
            ]},
            {"name": "Monitoring", "item": [
                _req("Recent predictions", "GET", "/predictions", query=[
                    {"key": "limit", "value": "50"},
                    {"key": "offset", "value": "0"},
                    {"key": "camera", "value": "HORGOS_I", "disabled": True},
                    {"key": "prediction", "value": "high", "disabled": True}],
                    description="Newest first. Shared across users - this is a view "
                                "of what the model is doing, not a per-account log. "
                                "30/minute."),
                _req("Prediction stats", "GET", "/predictions/stats",
                     description="Class mix, mean confidence and latency. Compare "
                                 "`by_class` shares against the training mix (~41% "
                                 "no_traffic / 41% light / 18% high) to spot drift. "
                                 "30/minute."),
            ]},
            _req("Reload model", "POST", "/reload", auth=False, headers=[
                {"key": "X-Reload-Token", "value": "{{reloadToken}}", "disabled": True}],
                description="Re-resolves the production alias. Open unless "
                            "RELOAD_TOKEN is set on the server, in which case enable "
                            "the X-Reload-Token header."),
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000",
                    help="running API to read the schema from")
    ap.add_argument("--out", default="api", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = enrich(fetch(args.url), args.url)
    (out / "openapi.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "postman_collection.json").write_text(
        json.dumps(build_collection(args.url), indent=2) + "\n")

    print(f"wrote {out/'openapi.json'} ({len(spec['paths'])} paths)")
    print(f"wrote {out/'postman_collection.json'}")


if __name__ == "__main__":
    main()
