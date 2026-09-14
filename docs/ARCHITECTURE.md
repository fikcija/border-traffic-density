# System architecture

Two services. One application image running three processes under supervisord, plus
MLflow.

```
   ┌──────────┐    ┌──────────────────────────────┐    ┌──────────────┐
   │  data/   │───▶│  app                         │───▶│   mlflow     │ :5001
   │  ~900MB  │    │  supervisord:                │◀───│  tracking +  │
   └──────────┘    │   1 prefect server    :4200  │    │  registry    │
   ┌──────────┐    │   2 flow runner              │    └──────────────┘
   │artifacts/│◀──▶│   3 uvicorn API       :8000  │      models:/…@production
   └──────────┘    │  TF + training code + API    │
                   └──────────────────────────────┘
```

MLflow stays separate because it is a third-party server, not our code. Everything
else lives in one image: the flow **imports** the pipeline rather than starting
sibling containers, so no Docker socket is mounted anywhere.

## Why this shape

**One image, three processes.** Training, orchestration and serving share a single
environment, so the flow calls `pipeline.step_search(cfg)` directly. No Docker socket,
no host-path translation, no sibling containers — the three most fragile parts of the
previous design are simply gone.

**MLflow stays its own service.** It is someone else's server; bundling it into the
application image buys nothing and complicates upgrades.

**MLflow is still the seam to serving.** The flow writes; the API reads the registry.
The model file lives in neither image.

**What this costs.** Docker supervises one container, so a dead process inside is
invisible to it — `autorestart=true` on each program is the mitigation, not a fix.
And the serving footprint argument weakens: the image carries TensorFlow regardless,
so quantization is justified by latency (3.7× faster) and model size (74% smaller)
rather than by a ~200 MB image.

## The model never sits in the serving image

The API resolves `models:/border-traffic-density@production` at startup, downloads that
version's artifacts, and serves them. Consequences:

- **Deploying a new model** is an alias change plus `POST /reload`. No rebuild, no
  redeploy.
- **Rollback is the same operation** pointing at the previous version — which is
  exactly the "ease of recovery" property the serving lecture asks for.
- **The model and its ROI polygons are registered together**, so serving can never pair
  a model with the wrong polygons.

## The gate is the point of the workflow

`src/flows/train_flow.py`:

```
prepare_features → run_search → export_model → run_perturbation → quantize
                                                                      │
                                                            log_and_register
                                                                      │
                                                            evaluate_gate
                                                             ╱          ╲
                                                   promote → reload_api   reject
                                                                     (production
                                                                      untouched)
```

**Each stage is its own Prefect task calling the matching `pipeline` function.**
That granularity is deliberate: a failure names the stage that broke and the traceback
points at the line, instead of reporting that training died somewhere inside a
twenty-minute subprocess. Stages also retry independently - a flaky MLflow connection
re-runs `log_and_register`, not the whole search.

`reload_api` is deliberately non-fatal. Promotion is already durable by that point and
the API picks the new version up on its next start, so an unreachable API logs a
warning rather than reporting a successful retrain as a failed flow.

An automated pipeline that promotes whatever it just trained is *worse* than no
automation — one bad run silently replaces a good model. `evaluate_gate` compares the
candidate against the version currently carrying the `production` alias and only
promotes when it wins on **all three** of these:

| condition | check | knob |
|---|---|---|
| accuracy | candidate beats production on `GATE_METRIC` | `GATE_MARGIN` |
| retention | `robustness_worst / robustness_clean` ≥ floor | `ROBUSTNESS_MIN_RETENTION` (0.50) |
| regression | candidate no less robust than production | `ROBUSTNESS_REGRESSION_TOLERANCE` (0.02) |

Verified behaviour: a candidate at 0.7750 against production 0.8093 is **rejected**; a
candidate at 0.8400 is **promoted** and the alias moves.

**Why accuracy alone is not enough.** `run_perturbation` measures what the model does
on degraded input, and the spread is large: the current model scores 0.8331 clean but
0.4352 under `gaussian_noise@3`. Nothing in a clean-data comparison would stop a
candidate that is marginally more accurate and far more brittle from being promoted —
so the gate reads the robustness metrics too.

Retention is expressed as a *fraction of the candidate's own clean score* rather than
an absolute floor. An absolute number is guesswork from a single observed model and
goes stale in both directions: toothless once models improve, blocking everything if
the task gets harder. The ratio keeps meaning the same thing — "the model must retain
at least half its clean-data performance under the worst condition tested."

The regression check bounds drift that retention alone would permit, where successive
models each land just above the line while robustness decays. It is skipped when there
is no incumbent, or when the incumbent predates the perturbation stage and logged no
robustness metrics — which is the case for the version in production today.

Retention is self-referential, so it applies even to the first-ever model: on an empty
registry there is no accuracy comparison to make, but a brittle first model is still
rejected.

**The gate compares `val_macro_f1`, not test.** Promotion is a *selection* decision,
and selecting on the test set spends it: pick the best of several noisy measurements
and the winner's score is biased upward, with no untouched estimate left to catch it.
Test is reported, never used to choose. `GATE_METRIC` can be set to `test_macro_f1`,
but only with that trade-off understood.

One wrinkle to state rather than hide: `train.py` retrains the winning configuration on
train+val, so the final model has no clean validation score of its own. The gate uses
the *search's* best validation score as a proxy - itself a max over trials, so mildly
optimistic, but consistent between candidate and incumbent, which is what a comparison
needs.

## Running it

```bash
docker compose -f docker/compose.yaml up -d --build
```

| service | URL |
|---|---|
| MLflow UI | http://localhost:5001 |
| Prefect UI | http://localhost:4200 |
| API | http://localhost:8000/health |

**First run.** The registry is empty until a training run completes, so the API starts
**degraded**: it comes up, `GET /health` reports 503, and the prediction endpoints
return 503 with a message naming the missing alias. The container reads as unhealthy,
which is accurate — it genuinely cannot serve. When the flow promotes a version its
`reload_api` task loads it and the API becomes healthy without a restart.

The API deliberately does *not* exit when no model resolves. supervisord retries a
failed program three times and then leaves it FATAL, so exiting would mean the API was
still dead when `reload_api` fired — a promoted model that nothing serves, needing a
manual `supervisorctl start api`.

Trigger a run from the Prefect UI, or directly:

```bash
docker compose -f docker/compose.yaml exec app \
  prefect deployment run 'retrain-border-traffic/scheduled-retrain'
```

The schedule (`TRAIN_CRON`, default Mondays 03:00) is set in `compose.yaml`.

**Running one stage by hand**, bypassing Prefect:

```bash
docker compose -f docker/compose.yaml exec app \
  python src/pipeline.py --step search --backbone mobilenetv2 --letterbox
```

## Gotchas worth knowing

**`tensorflow`, not `tensorflow-cpu`**, in the trainer image: the `-cpu` wheels are
x86_64 only, so on an Apple Silicon host the build cannot resolve them.

**Training inside the container is CPU-only** — there is no Metal passthrough on macOS.
The trainer container is for reproducible automation; heavy runs are still faster
natively. This is a property of the platform, not the design.

**MLflow is published on host port 5001, not 5000.** macOS AirPlay Receiver holds
5000, so binding it fails. Inside the compose network the service is still
`http://mlflow:5000`; only the browser URL differs. Override with `MLFLOW_PORT` if
5001 is also taken.

**The feature cache is skipped when present.** `pipeline.py` will not recompute
`features.npy` if `artifacts/<backbone>/` already has it. Delete that folder to force a
clean rebuild.

**Retraining on unchanged inputs is a no-op, by design.** `train.py` sets a fixed seed
(`--seed`, default 42), the tuner runs with `overwrite=True`, and the feature cache is
deterministic — so the same data and the same budget produce a bit-identical model.
Versions 1, 2 and 3 in the registry all score `val_macro_f1 = 0.8313267827033997`, and
the gate's strict `>` rejects each successive tie. This is reproducibility working
correctly, but it surprises: re-running to "see if the results change" after a pipeline
change will not move the numbers unless the change affects training. Adding a
measurement stage such as `run_perturbation` adds *new* metrics and leaves the existing
ones untouched. To get a genuinely different model, vary `--seed`, `--trials`, the
backbone, or the data.

## Two ways to run the API

| | how | when |
|---|---|---|
| **stack** | the `app` service via compose, model from the registry | normal |
| **standalone** | `src/serve/Dockerfile` + `src/serve/build.sh`, model baked in | demoing serving alone, or if compose misbehaves |

`registry.py` chooses automatically: registry mode when `MODEL_NAME` and
`MLFLOW_TRACKING_URI` are both set, local file otherwise.

## What was tested, and what was not

Verified in development: registry resolution and download, the API serving a real
prediction from a registry-sourced model, `POST /reload` re-resolving the alias, and the
promotion gate accepting and rejecting correctly against a live MLflow server.

**Verified running.** Both containers build, start and talk to each other. The stack has
completed end-to-end retrains and produced three registered model versions; the API
reports healthy and serves the int8 variant resolved from the registry:

```json
{"status":"ok","model":{"source":"registry","version":"2","model":"model_int8.tflite"}}
```

The robustness conditions were exercised against the live MLflow server: the current
candidate passes retention at 0.522 (floor 0.50), the regression check correctly skips
against a production version that predates the perturbation stage, and raising the
floor to 0.60 flips the same candidate to a rejection.

Not verified: behaviour on a GPU host, and any run where `prepare_features` rebuilds
the cache from scratch rather than reusing `features.npy`.
