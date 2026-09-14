# Border Traffic Density Classifier

Classifies a border-crossing camera image into traffic density: **no_traffic / light / high**.

MLOps components implemented:

- *Treniranje i evaluacija modela* — AutoML (Bayesian hyperparameter optimization),
  offline evaluation against baselines, robustness measurement under degraded input.
- *Primena i serviranje modela* — containerised stack, Prefect-orchestrated
  retraining, MLflow registry with a promotion gate, quantised model served over REST.

## The whole thing in one command

```bash
docker compose -f docker/compose.yaml up -d --build
```

| service | URL |
|---|---|
| inference API | http://localhost:8000/health |
| Prefect UI | http://localhost:4200 |
| MLflow UI | http://localhost:5001 |

Retraining is triggered from the Prefect UI, or on the command line:

```bash
docker compose -f docker/compose.yaml exec app \
  prefect deployment run 'retrain-border-traffic/scheduled-retrain'
```

It also runs on a schedule (`TRAIN_CRON`, default Mondays 03:00). Three documents cover
that half of the project and are kept current:

| doc | what it covers |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | the two services, the pipeline stages, the promotion gate |
| [SERVING.md](docs/SERVING.md) | the REST API — auth, batch prediction, the prediction log, the Postman/OpenAPI schema |
| [RESULTS.md](docs/RESULTS.md) | measured numbers |

The rest of this file is the modelling work: how the classifier is built and why.

## How it works

A frozen ImageNet backbone (MobileNetV2 by default, any of eleven available) is run
over every image **once** and the
features are cached to disk. Every AutoML trial then trains only a small
classification head on those cached vectors, so a trial takes seconds instead of
minutes. That is what makes a 25-trial search feasible on a laptop.

Two preprocessing decisions worth knowing:

- **Burned-in overlays are masked.** The timestamp (top right) and camera name
  (bottom left) are painted black. Both are already in the filename, so nothing is
  lost — but a CNN can read them, and date/hour correlates with traffic density.
  Without masking the model can score well by reading the clock.
- **Features keep a 3×3 spatial grid** rather than being globally pooled to a single
  vector, so coarse layout (which side of the frame the queue is on) survives. The
  head can still average it away — that choice is part of the search space.

Metadata (camera, direction, timestamp) is deliberately **not** a model input. Camera
identity is strongly predictive of the label in this dataset — GRADINA_U is 30
no_traffic vs 298 low, SID_U is 305 vs 45 high — so a model given the camera would
learn base rates instead of looking at cars, and would fail exactly on the anomalous
cases that matter.

## Data

```bash
unzip <path-to>/border-traffic-dataset.zip -d data   # -> data/labeled/, data/unlabeled/
```

The dataset archive is kept outside the repository (it is ~865 MB); skip this if
`data/labeled/` already exists. `data/` is bind-mounted read-only into the container.

## Running a stage by hand

The pipeline runs in the container — there is no local install. Individual stages can
still be driven directly, which is what the Prefect tasks do internally:

```bash
C="docker compose -f docker/compose.yaml exec app"
$C python src/data/prepare.py  --data-dir data --backbone mobilenetv2   # ~10-20 min, once per backbone
$C python src/training/train.py    --artifacts artifacts/mobilenetv2 --classes 3 --trials 25
$C python src/training/finetune.py --artifacts artifacts/mobilenetv2 --classes 3 --unfreeze 30
```

Training in the container is CPU-only on macOS (no Metal passthrough), so these are
slower than they would be natively — see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Trying several backbones

`prepare.py` writes to `artifacts/<backbone>/`, so each backbone gets its own cached
features and its own runs, and nothing collides. Repeat the three commands per
backbone, then collect everything into one table and chart:

```bash
bash scripts/run_all.sh
```

That runs prepare → search → fine-tune for each backbone and then `compare.py`.
Every stage is **skipped if its output already exists**, so a crash or a Ctrl-C costs
only the stage that was running — just re-run it. Delete the relevant folder under
`artifacts/` to force a redo. Output goes to `logs/run_<timestamp>.log`.

Override anything via the environment:

```bash
BACKBONES="convnexttiny" bash scripts/run_all.sh     # one backbone
TRIALS=10 EPOCHS=6 bash scripts/run_all.sh           # smaller budget if time is short
```

Or run the stages by hand:

```bash
for bb in convnexttiny efficientnetb0 resnet50 mobilenetv2; do
  python src/data/prepare.py  --data-dir data --backbone $bb
  python src/training/train.py    --artifacts artifacts/$bb --classes 3 --trials 25
  python src/training/finetune.py --artifacts artifacts/$bb --classes 3 --unfreeze 30
done
python src/evaluation/compare.py --classes 3
```

Available: `mobilenetv2`, `mobilenetv3`, `efficientnetb0`, `efficientnetv2b0`,
`resnet50`, `resnet50v2`, `densenet121`, `convnexttiny`, `convnextsmall`, `xception`,
`inceptionv3`. Each uses its own `preprocess_input` — they differ per family
(MobileNet maps to [-1,1], ResNet does caffe-style mean subtraction, EfficientNet and
ConvNeXt expect raw 0-255) and mixing them up quietly ruins accuracy. Xception and
InceptionV3 default to 299px, so they are slower; the rest run at 224.

**ConvNeXt** uses LayerNorm and LayerScale instead of BatchNorm and degrades quickly
under large fine-tuning steps, so `finetune.py` drops its learning rate to 1e-5
automatically. It is also the heaviest of the 224px options — expect roughly 3–5×
MobileNetV2's epoch time.

### Accuracy vs deployability

These two things pull in opposite directions, and the split matters for the
*Primena i serviranje* component:

| | ConvNeXtTiny | MobileNetV2 |
|---|---|---|
| parameters | ~28M | ~3.4M |
| ops | LayerNorm, GELU, 7×7 depthwise | BatchNorm, ReLU6, 3×3 depthwise |
| post-training INT8 | partial — expect op fallbacks | clean, it is the canonical TFLite target |

If ConvNeXt wins on macro F1 and MobileNetV2 wins on size and latency, report both
rather than picking one. An explicit accuracy/size/latency table *is* the
cloud-vs-edge argument from the deployment lecture, and it is a stronger result than
either model alone. Serve the accurate one, and show what the compressed one costs
you in macro F1.

`compare.py` writes `artifacts/comparison_3cls/backbones.{csv,png}` — a ranked table
and a chart of frozen vs fine-tuned macro F1 per backbone against the baseline. Only
compare backbones under the same trial budget, or the comparison means nothing.

### Why three stages

Transfer learning here is done in the standard two phases, and they have opposite
compute profiles:

1. **Feature caching** (`prepare.py`) — every image through the frozen backbone
   once, features written to disk. Paid once per backbone.
2. **Head search** (`train.py`) — backbone frozen, so features are identical every
   epoch and can be reused. A trial costs seconds, which is what makes a 25-trial
   Bayesian search plus a 25-trial random comparison affordable.
3. **Fine-tuning** (`finetune.py`) — unfreezes the top of the backbone so the
   features themselves adapt to traffic-camera imagery. Every epoch must now push all
   images through the network, roughly 100× the compute per epoch, so this is a
   *single run* using the head configuration the search already found — not a second
   search.

Stage 2 is where the AutoML component lives. Stage 3 is where most of the remaining
accuracy lives. `finetune.py` prints both numbers side by side and the delta.

Fine-tuning uses a low learning rate (1e-4 by default) and keeps BatchNorm layers
frozen — large steps or moving BN statistics destroy the pretrained features. It also
enables light augmentation (brightness, contrast, small rotation and translation),
which is possible here because the full image pipeline is running anyway. No flip,
zoom, or crop: a flip moves the monitored lanes to the wrong side of the ROI, and
zoom or crop can push vehicles out of frame and change the label.

Rough epoch times for ~8.5k images at 224px: Apple Silicon with `tensorflow-metal`
1–3 min, plain CPU 8–15 min, discrete GPU well under a minute. Lower `--unfreeze` or
`--epochs` if that's too slow.

`prepare.py` builds `manifest.csv` and `features.npy` under `artifacts/<backbone>/`,
suffixed `_roi_lb` when ROI masking and letterboxing are on (the default in the
pipeline). `train.py` then writes to `run_3cls/` beside them:

| File | What |
|---|---|
| `results.json` | baselines, both search histories, test metrics, confusion matrix |
| `search_comparison.png` | Bayesian vs Random, best-so-far under an equal budget |
| `confusion_matrix.png` | test-set confusion matrix |
| `per_camera.csv` | macro F1 per camera-direction, worst first |
| `model.keras` | the head trained on the winning configuration |

### Justifying the 3-class merge

The dataset is annotated in four classes; `low_traffic` and `moderate_traffic` are
merged because that boundary is the subjective one. To show this empirically rather
than assert it, run the four-class variant — it reuses the same cached features, so
it costs one extra search:

```bash
python src/training/train.py --classes 4 --trials 25
```

If `low` and `moderate` bleed into each other in `run_4cls/confusion_matrix.png`
while both stay separable from `no_traffic` and `high`, the merge is justified by
evidence.

## Splits

Chronological, never random — 18.5% of consecutive frames from the same camera are
under five minutes apart, so a random split would put near-duplicate images in both
train and test.

| Split | Period | ~Images |
|---|---|---:|
| train | ≤ 2024-05-31 | 6,080 |
| val | 2024-06-01 → 2024-12-31 | 2,378 |
| test | ≥ 2025-01-01 | 1,411 |

Change with `--val-start` / `--test-start` on `prepare.py`.

## Baselines

Reported by `train.py` before any training, because metrics mean nothing without
them. On the 3-class split: zero-rule ≈ 0.19 macro F1 (39% accuracy), random
following the label distribution ≈ 0.33. Note the inversion — random *beats* zero
rule on macro F1 while losing on accuracy, which is why macro F1 is the headline
metric and accuracy is not.

## Per-camera ROI masking

Labels depend on cars in the monitored direction only, so each of the 20 cameras has
its own polygon in `config/roi_regions.json`. Preprocessing greys out everything
outside it, crops to its bounding box and pads to square. Artifact directories built
this way carry the `_roi_lb` suffix.

This is why the serving API needs to know the camera: an ROI-trained model is wrong on
a whole frame, and the wrong polygon produces a confident wrong answer rather than an
error. The polygons are registered in MLflow alongside the model, so serving can never
pair a model with the wrong ones.

## Known limitations

- **Confidence is not calibrated.** Measured on the 1,411-image test split: mean
  confidence 0.980 against 0.878 accuracy. 91% of predictions land above 0.99, where
  the model is right about 90% of the time. Fine for `argmax`, but a decision
  threshold on a probability would be reading noise. Temperature scaling would fix
  it; it is not implemented.
- **Augmentation is not searched.** It runs at a fixed strength during fine-tuning
  only; the head search uses cached features, which precludes it.
- **Will not transfer to an unseen crossing** — relevant-lane geometry is
  camera-specific. A new camera needs its ROI polygon added before it can be served
  at all; the API returns 400 for any camera it has no polygon for.
