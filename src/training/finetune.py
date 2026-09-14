"""
Stage 3 (optional but usually worth it): fine-tune the backbone end-to-end.

train.py searches the head on cached features — fast, but the backbone stays frozen
at ImageNet weights. This takes the winning head configuration and unfreezes the top
of MobileNetV2 so the features themselves adapt to traffic cameras. One run, not a
search: the search already happened in stage 2.

    python src/finetune.py --classes 3 --unfreeze 30 --epochs 12

Expect a real gain over the frozen-feature model. Cost: every epoch now pushes all
images through the network. Rough guide for ~8.5k training images at 224px --
Apple Silicon with tensorflow-metal ~1-3 min/epoch, plain CPU ~8-15 min/epoch,
discrete GPU well under a minute.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import OVERLAY_BOXES, FINETUNE_LR, load_backbone
from train import CLASSES, plot_confusion, camera_slices


def build_dataset(paths, labels, size, batch, mask, training, aug_strength,
                  preprocess_input):
    import tensorflow as tf

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        return tf.image.resize(img, (size, size)) * mask, label   # stays 0-255

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch)

    # Augment in 0-255 space, then preprocess. Doing it the other way round would
    # need a per-backbone value_range: MobileNet maps to [-1,1], ResNet does
    # caffe-style mean subtraction, EfficientNet keeps raw 0-255.
    if training and aug_strength > 0:
        import keras
        from keras import layers
        # No flip / zoom / crop: without per-camera ROI masks a flip moves the
        # monitored lanes to the wrong side, and zoom or crop can push vehicles out
        # of frame, which changes the label.
        aug = keras.Sequential([
            layers.RandomBrightness(0.4 * aug_strength, value_range=(0, 255)),
            layers.RandomContrast(0.4 * aug_strength),
            layers.RandomRotation(0.01 * aug_strength, fill_mode="constant"),
            layers.RandomTranslation(0.03 * aug_strength, 0.03 * aug_strength,
                                     fill_mode="constant"),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.map(lambda x, y: (preprocess_input(x), y),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build_head(hp_values, feat_shape, n_classes):
    """Rebuild the head that train.py's search selected, from its recorded values."""
    import keras
    from keras import layers
    inp = layers.Input(shape=feat_shape)
    x = (layers.GlobalAveragePooling2D()(inp) if hp_values["pool"] == "gap"
         else layers.Flatten()(inp))
    x = layers.BatchNormalization()(x)
    for i in range(hp_values["n_layers"]):
        x = layers.Dense(hp_values[f"units_{i}"], activation="relu")(x)
        x = layers.Dropout(hp_values[f"dropout_{i}"])(x)
    return keras.Model(inp, layers.Dense(n_classes, activation="softmax")(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    ap.add_argument("--unfreeze", type=int, default=30,
                    help="how many trailing backbone layers to unfreeze (0 = none)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-4 (1e-5 for ConvNeXt); keep it low - large "
                         "steps destroy pretrained features")
    ap.add_argument("--aug", type=float, default=0.5,
                    help="augmentation strength 0-1, 0 disables")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import keras, tensorflow as tf
    from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                                 classification_report)
    from sklearn.utils.class_weight import compute_class_weight

    keras.utils.set_random_seed(args.seed)
    labels = CLASSES[args.classes]
    src_run = args.artifacts / f"run_{args.classes}cls"
    out = args.artifacts / f"finetune_{args.classes}cls"
    out.mkdir(parents=True, exist_ok=True)

    prev = json.loads((src_run / "results.json").read_text())
    hp_values = prev["winner"]["hyperparameters"]
    cfg = json.loads((args.artifacts / "prepare_config.json").read_text())
    size, backbone_name = cfg["size"], cfg["backbone"]
    if args.lr is None:
        args.lr = FINETUNE_LR.get(backbone_name, 1e-4)
    print(f"backbone: {backbone_name} @ {size}px, fine-tune lr {args.lr:g}")
    print(f"reusing head from {src_run.name}: {hp_values}")
    print(f"frozen-feature test macro F1 was {prev['test']['macro_f1']:.4f}")

    df = pd.read_csv(args.artifacts / "manifest.csv")
    y = df[f"class_{args.classes}"].map({c: i for i, c in enumerate(labels)}).values
    Y = keras.utils.to_categorical(y, len(labels))
    idx = {s: (df.split == s).values for s in ("train", "val", "test")}

    # overlay mask, same boxes as prepare.py, applied after the resize
    mask = np.ones((size, size, 1), dtype=np.float32)
    for y0, y1, x0, x1 in OVERLAY_BOXES:
        mask[int(y0 * size):int(y1 * size), int(x0 * size):int(x1 * size)] = 0.0
    mask = tf.constant(mask)

    base, preprocess_input = load_backbone(backbone_name, size)

    def ds(split, training):
        return build_dataset(df.path[idx[split]].tolist(), Y[idx[split]],
                             size, args.batch, mask, training,
                             args.aug if training else 0.0, preprocess_input)

    base.trainable = True
    frozen = len(base.layers) - args.unfreeze
    for layer in base.layers[:frozen]:
        layer.trainable = False
    # BatchNorm layers must stay in inference mode: fine-tuning with tiny batches
    # otherwise wrecks the pretrained running statistics.
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    print(f"unfrozen: {args.unfreeze} of {len(base.layers)} backbone layers")

    pool = keras.layers.AveragePooling2D(pool_size=3, strides=2)
    head = build_head(hp_values, pool.compute_output_shape(base.output_shape)[1:],
                      len(labels))
    head.load_weights(src_run / "model.keras")   # start from the searched head

    inp = keras.layers.Input(shape=(size, size, 3))
    model = keras.Model(inp, head(pool(base(inp))))
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="categorical_crossentropy",
                  metrics=[keras.metrics.F1Score(average="macro", name="f1_score")])

    cw = compute_class_weight("balanced", classes=np.arange(len(labels)),
                              y=y[idx["train"]])
    model.fit(ds("train", True), validation_data=ds("val", False),
              epochs=args.epochs, class_weight=dict(enumerate(cw)), verbose=2,
              callbacks=[
                  keras.callbacks.EarlyStopping("val_f1_score", mode="max",
                                                patience=4, restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau("val_f1_score", mode="max",
                                                    factor=0.3, patience=2, min_lr=1e-6),
              ])

    pred = model.predict(ds("test", False), verbose=0).argmax(1)
    true = y[idx["test"]]
    res = {
        "n_classes": args.classes, "labels": labels, "backbone": backbone_name,
        "baselines": prev["baselines"],
        "frozen_features_test": prev["test"],
        "head_hyperparameters": hp_values,
        "unfreeze": args.unfreeze, "lr": args.lr, "aug_strength": args.aug,
        "test": {"accuracy": float(accuracy_score(true, pred)),
                 "macro_f1": float(f1_score(true, pred, average="macro",
                                            zero_division=0))},
    }
    delta = res["test"]["macro_f1"] - prev["test"]["macro_f1"]
    print(f"\nfrozen features : {prev['test']['macro_f1']:.4f} macro F1")
    print(f"fine-tuned      : {res['test']['macro_f1']:.4f} macro F1  ({delta:+.4f})")
    print(classification_report(true, pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(true, pred, labels=range(len(labels)))
    plot_confusion(cm, labels, out / "confusion_matrix.png")
    res["confusion_matrix"] = cm.tolist()

    test_df = df[idx["test"]].copy(); test_df["pred"] = pred; test_df["true"] = true
    camera_slices(test_df, labels).to_csv(out / "per_camera.csv")

    model.save(out / "model.keras")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
