# Results — border traffic density classification

MLOps components: *Treniranje i evaluacija modela* and *Primena i serviranje modela*.

## Setup

- **9,869** hand-labeled images, **20 camera-directions** (10 crossings × entry/exit), 2023-12-22 → 2026-01-02
- Four annotated classes collapsed to three: `low` + `moderate` → **`light`** (that boundary is the subjective one)
- **Chronological** splits — 18.5% of consecutive same-camera frames are <5 min apart, so a random split would leak near-duplicates
- Burned-in timestamp and camera-name overlays masked; the same information is in the filename, and unmasked a CNN can read the clock (date correlates with density)
- Metadata never used as a model input — camera identity strongly predicts the label, so a model given it would learn base rates instead of counting cars
- Headline metric: **macro F1**

| split | period | no_traffic | light | high | high % |
|---|---|---:|---:|---:|---:|
| train | ≤ 2024-05-31 | 2,313 | 2,659 | 1,108 | 18% |
| val | 2024-06 → 2024-12 | 1,008 | 789 | 581 | 24% |
| test | ≥ 2025-01-01 | 750 | 557 | 104 | **7%** |

The `high` share falls sharply in the test period — real temporal drift, and the reason
`high` metrics rest on only 104 images.

## Baselines

| baseline | accuracy | macro F1 |
|---|---:|---:|
| Zero rule (always `light`, the largest training class) | 0.395 | **0.189** |
| Random following the label distribution | 0.404 | **0.322** |

The random baseline *beats* zero rule on macro F1 while losing on accuracy — a compact
demonstration of why macro F1 is the headline metric.

---

## Headline result: ROI masking, not the backbone

Labels depend on traffic in the monitored direction only, but a raw frame also contains
the opposite-direction queue. Per-camera ROI polygons mask everything outside the
monitored lanes (filled with ImageNet-mean grey, then cropped to the polygon's bounding
box and letterboxed to a square so narrow ROIs are not stretched).

**Validation macro F1, every backbone, before and after:**

| backbone | raw frames | + ROI letterbox | Δ |
|---|---:|---:|---:|
| efficientnetb0 | 0.7938 | **0.8480** | **+0.054** |
| resnet50 | 0.8007 | 0.8428 | +0.042 |
| mobilenetv2 | 0.7707 | 0.8291 | +0.058 |
| convnexttiny | 0.7855 | 0.8281 | +0.043 |

A consistent +0.04 to +0.06 on all four. **Input representation mattered far more than
architecture choice** — the spread between backbones is 0.020, and their test ordering
disagrees with validation, so they are within noise of one another.

## Full run table

| run | val macro F1 | test macro F1 | test accuracy |
|---|---:|---:|---:|
| efficientnetb0_roi_lb | **0.8480** | 0.8035 | 0.852 |
| resnet50_roi_lb | 0.8428 | **0.8196** | 0.858 |
| mobilenetv2_roi_lb | 0.8291 | 0.8093 | 0.852 |
| convnexttiny_roi_lb | 0.8281 | 0.8106 | **0.864** |
| mobilenetv2_roi (no letterbox) | 0.8089 | 0.8084 | 0.854 |
| resnet50 | 0.8007 | 0.7992 | 0.851 |
| efficientnetb0 | 0.7938 | 0.7581 | 0.821 |
| convnexttiny | 0.7855 | 0.7783 | 0.828 |
| mobilenetv2 | 0.7707 | 0.8039 | 0.853 |

**Selection is on validation, never test** — choosing by test score would void it as a
held-out estimate. On validation the winner is `efficientnetb0_roi_lb`.

Letterboxing is worth +0.020 on its own (`mobilenetv2_roi` 0.8089 → `mobilenetv2_roi_lb`
0.8291): ROI bounding boxes range from aspect 0.50 to 1.53, so resizing straight to
224×224 distorts narrow ones.

**The partner's ConvNeXt finding did not reproduce**, before or after ROI — it places
third or fourth in both regimes.

---

## AutoML: Bayesian optimization vs random search

Equal budgets (25 trials each), identical search space, same seed.

| run | Bayesian | Random | winner |
|---|---:|---:|---|
| mobilenetv2 | 0.7623 | 0.7707 | random |
| resnet50 | 0.7922 | 0.8007 | random |
| convnexttiny | 0.7837 | 0.7855 | random |
| efficientnetb0 | **0.7938** | 0.7885 | bayesian |
| mobilenetv2_roi | **0.8089** | 0.8056 | bayesian |
| mobilenetv2_roi_lb | **0.8291** | 0.8256 | bayesian |
| convnexttiny_roi_lb | **0.8281** | 0.8260 | bayesian |
| efficientnetb0_roi_lb | **0.8480** | 0.8477 | bayesian |
| resnet50_roi_lb | 0.8370 | **0.8428** | random |

**On raw frames random search won 3 of 4. On ROI inputs Bayesian won 4 of 5.**

The explanation is the same in both regimes: Bayesian optimization builds a surrogate
model of configuration → score, so it needs that relationship to be learnable. On raw
frames the opposite-direction confounder adds noise to every trial's validation score,
the surrogate fits noise, and the method degenerates toward random. Once ROI masking
removes the confounder the signal is cleaner and the informed search pulls ahead.

*Caveat:* random search shares seed and space across backbones, so it draws identical
configurations each time — fair for comparison, but the runs are not independent.

---

## Best model in detail

Per-class figures below are for `mobilenetv2_roi_lb` (the deployment candidate, see
serving section). Confusion matrix on the 1,411-image test set:

```
                 predicted
              none  light  high
true none      687     61     2
true light      79    428    50
true high        1     16    87
```

**Errors are almost entirely between adjacent density levels** — 2 empty crossings called
busy, 1 busy crossing called empty. The model does not make the catastrophic confusion,
which also retroactively supports merging `low` and `moderate`.

The weak class remains `high`, limited by precision rather than recall — the model
over-calls busy traffic, a consequence of class-weighted loss on a rare class.

## Slice-based evaluation

Per camera-direction, aggregate macro F1 of ~0.81 hides real variation: the weakest
cameras score around 0.45–0.55. VRSKA-CUKA is degenerate by construction — that crossing
is almost entirely `no_traffic` in the dataset (VRSKA-CUKA_U: 188 / 8 / 1 / 0) — so its
per-class metrics are unstable by design, not by fault. SID_U and SID_I are the
substantive weak spots.

## Fine-tuning — tested at two settings, neutral at best

| run | macro F1 | trainable backbone weights |
|---|---:|---:|
| frozen (head only) | **0.804** | 0% |
| unfreeze 10, aug 0 | **0.807** | 32% |
| unfreeze 30, aug 0.5 | **0.775** | 67% |

The gentle configuration ties frozen (+0.003 is inside noise). "Unfreeze 30" is not a
timid setting — MobileNetV2's parameters concentrate at the end, so 19% of layers is 67%
of weights.

The three runs trace one precision/recall curve on `high` (precision 0.511 / 0.593 /
0.629 against recall 0.923 / 0.827 / 0.798) and macro F1 is nearly flat along it. But the
aggressive run also lost `light` F1 (0.777 vs 0.811/0.815), which is not part of that
trade-off, so it degraded outright — most likely the augmentation, since brightness and
contrast jitter attacks the cues that separate a full lane from an empty one.

**Conclusion: adapting the features buys nothing here.** Frozen ImageNet features are
sufficient; the bottleneck was the input representation, which is exactly what ROI
masking addressed.

*Two honest caveats:* the aggressive run varied unfreeze depth **and** augmentation
together, so the drop cannot be cleanly attributed; and all three were compared on test,
which is selection-on-test. The defensible claim is that fine-tuning was tested and found
neutral — not that `unfreeze 10` is best.

---

## Post-training quantization

Applied to `mobilenetv2_roi_lb`. All four variants scored in a single pass over the test
set, so every one sees byte-identical input; latency measured one image at a time, which
is what a REST endpoint does; the float32 baseline is `tf.function`-compiled so it is not
charged eager-mode Python overhead.

| variant | size (MB) | latency p50 (ms) | accuracy | macro F1 | agreement with fp32 |
|---|---:|---:|---:|---:|---:|
| Keras float32 | 24.57 | 10.57 | 0.8512 | **0.8087** | — |
| TFLite float16 | 11.84 | 7.48 | 0.8526 | **0.8106** | 99.9% |
| TFLite dynamic-range | 6.20 | 4.24 | 0.8469 | 0.7977 | 97.0% |
| TFLite int8 | 6.41 | 2.83 | 0.8561 | 0.8008 | 92.8% |

- **float16 is free**: 52% smaller, 1.4× faster, no measurable quality change.
- **int8 is 74% smaller and 3.7× faster** for −0.008 macro F1.

Two findings worth reporting:

**int8 accuracy rose (0.8512 → 0.8561) while macro F1 fell (0.8087 → 0.8008).** This is
consistent with quantization noise nudging borderline predictions toward the majority
classes — the same accuracy/macro-F1 divergence that motivated the metric choice in the
first place, reappearing in a completely different part of the pipeline.

**Agreement with float32 is only 92.8% for int8.** The model holds its aggregate score
while changing roughly one prediction in fourteen. Aggregate metrics alone would have
hidden that entirely, which is the argument for reporting agreement alongside them.

---

## Deployment note: accuracy vs deployability

`efficientnetb0_roi_lb` wins on validation; `mobilenetv2_roi_lb` is the deployment
candidate. The validation spread across the four ROI runs is 0.020 with disagreeing test
ordering, so they are statistically indistinguishable — and MobileNetV2 is materially
cheaper to serve (BatchNorm/ReLU6 quantize cleanly to int8, where LayerNorm/GELU
architectures do not). Choosing the indistinguishable-but-cheaper model is the
cloud-versus-edge trade-off from the deployment lecture, made explicitly rather than by
accident.

---

## Limitations

- **`high` is estimated from 104 test images** — wide error bars on the class that
  dominates macro F1. Small between-model differences should not be over-read.
- **Augmentation is not searched**; fixed strength during fine-tuning only, since the head
  search runs on cached features.
- **Will not transfer to an unseen crossing** — ROI polygons are per-camera and lane
  geometry is camera-specific. In deployment a new camera is configured, not inferred.
- **Temporal drift is present** — `high` falls from 18% of the training period to 7% of
  test. Expect degradation over time.

## Questions to expect

- *Why macro F1 and not accuracy?* — zero rule scores 0.395 accuracy but 0.189 macro F1.
  And int8 quantization raised accuracy while lowering macro F1.
- *Which backbone is best?* — on validation, EfficientNetB0 with ROI. But the spread is
  0.020 and test ordering disagrees, so they are within noise; ROI masking is what
  mattered, worth +0.04 to +0.06 on every one of them.
- *Random search beat your Bayesian optimization?* — on raw frames, yes. On ROI inputs
  Bayesian won 4 of 5. A surrogate model needs a learnable signal; the confounder was
  adding noise to every trial.
- *Why did fine-tuning not help?* — tested at two settings, neutral at best. The
  bottleneck was input representation, not feature quality.
- *How was the light/high boundary defined?* — cars per lane in the monitored direction,
  0–2 per lane ≈ under 10 minutes' wait.
