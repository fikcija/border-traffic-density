"""
Stage 2: baselines, AutoML search (Bayesian vs Random under one budget), final model.

    python src/train.py --classes 3 --trials 25

Reads artifacts/features.npy + artifacts/manifest.csv written by prepare.py.
Every trial trains only a small head on cached features, so a trial is seconds.
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = {3: ["no_traffic", "light", "high"],
           4: ["no_traffic", "low_traffic", "moderate_traffic", "high_traffic"]}

BLUE, ORANGE = "#2563eb", "#ea580c"   # CVD-validated pair
INK, MUTED, GRID = "#1c1c1a", "#6b6b66", "#e4e4e1"


# ---------------------------------------------------------------- baselines
def baselines(y_train, y_test, n_classes, rng):
    #Zero rule and label-distribution random.
    from sklearn.metrics import f1_score, accuracy_score
    counts = np.bincount(y_train, minlength=n_classes)
    out = {}

    zero = np.full_like(y_test, counts.argmax())
    out["zero_rule"] = {"accuracy": accuracy_score(y_test, zero),
                        "macro_f1": f1_score(y_test, zero, average="macro", zero_division=0)}

    p = counts / counts.sum()
    scores = [f1_score(y_test, rng.choice(n_classes, len(y_test), p=p),
                       average="macro", zero_division=0) for _ in range(20)]
    rand = rng.choice(n_classes, len(y_test), p=p)
    out["random_label_dist"] = {"accuracy": accuracy_score(y_test, rand),
                                "macro_f1": float(np.mean(scores))}
    return out


# ---------------------------------------------------------------- search
def make_builder(feat_shape, n_classes):
    import keras
    from keras import layers

    def build(hp):
        inp = layers.Input(shape=feat_shape)
        if hp.Choice("pool", ["gap", "flatten"]) == "gap":
            x = layers.GlobalAveragePooling2D()(inp)
        else:
            x = layers.Flatten()(inp)          # keeps coarse spatial layout
        x = layers.BatchNormalization()(x)
        for i in range(hp.Int("n_layers", 0, 2)):
            x = layers.Dense(hp.Int(f"units_{i}", 64, 512, step=64), activation="relu")(x)
            x = layers.Dropout(hp.Float(f"dropout_{i}", 0.0, 0.6, step=0.1))(x)
        out = layers.Dense(n_classes, activation="softmax")(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float("lr", 1e-4, 1e-2, sampling="log")),
            loss="categorical_crossentropy",
            metrics=[keras.metrics.F1Score(average="macro", name="f1_score")])
        return model
    return build


def run_search(kind, builder, data, trials, epochs, class_weight, out_dir, seed):
    import keras, keras_tuner as kt
    Xtr, Ytr, Xva, Yva = data
    tuner_cls = kt.BayesianOptimization if kind == "bayesian" else kt.RandomSearch
    kwargs = dict(objective=kt.Objective("val_f1_score", "max"),
                  max_trials=trials, seed=seed, overwrite=True,
                  directory=str(out_dir / "tuner"), project_name=kind)
    if kind == "random":
        kwargs["seed"] = seed
    tuner = tuner_cls(builder, **kwargs)
    tuner.search(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs,
                 batch_size=64, class_weight=class_weight, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(
                     "val_f1_score", mode="max", patience=4, restore_best_weights=True)])

    history, best = [], -1.0
    for t in sorted(tuner.oracle.trials.values(), key=lambda t: int(t.trial_id)):
        score = t.score if t.score is not None else 0.0
        best = max(best, score)
        history.append(best)
    return tuner, history


# ---------------------------------------------------------------- slices
def camera_slices(test_df, labels):
    """Per-camera-direction breakdown, worst macro F1 first.

    macro_f1 alone is unstable on this data: several cameras have only one or two
    test images of a class, and a single error there swings their macro F1 by 1/n
    of a class (~0.33 for 3 classes). The per-class support columns and min_class_n
    make that visible instead of leaving it to be inferred.
    """
    from sklearn.metrics import f1_score

    def one(g):
        row = {"n": len(g),
               "accuracy": float((g.true == g.pred).mean()),
               "macro_f1": f1_score(g.true, g.pred, average="macro", zero_division=0)}
        per = f1_score(g.true, g.pred, average=None,
                       labels=range(len(labels)), zero_division=0)
        supports = []
        for i, lab in enumerate(labels):
            n_i = int((g.true == i).sum())
            supports.append(n_i)
            row[f"n_{lab}"] = n_i
            row[f"f1_{lab}"] = per[i]
        row["min_class_n"] = min(supports)
        return pd.Series(row)

    return (test_df.groupby("camera_dir")
            .apply(one, include_groups=False)
            .sort_values("macro_f1"))


# ---------------------------------------------------------------- plots
def plot_search(histories, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (name, hist), color in zip(histories.items(), [BLUE, ORANGE]):
        xs = range(1, len(hist) + 1)
        ax.plot(xs, hist, color=color, linewidth=2, label=name)
        ax.annotate(f"{name}  {hist[-1]:.3f}", (len(hist), hist[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=MUTED, fontsize=9, va="center")
    n = max(len(h) for h in histories.values())
    ax.set_xlim(1, n * 1.22)                       # room for the direct labels
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("trial"); ax.set_ylabel("best val macro F1 so far")
    ax.set_title("AutoML search: best-so-far under an equal trial budget",
                 color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    # legend above the axes: direct labels sit at the line ends, so an in-plot
    # legend collides with them
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_confusion(cm, labels, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2.4, 1.4 * len(labels) + 2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)          # sequential, one hue
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test)", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import keras
    from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                                 classification_report)
    from sklearn.utils.class_weight import compute_class_weight

    keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.artifacts / f"run_{args.classes}cls"
    out.mkdir(parents=True, exist_ok=True)

    labels = CLASSES[args.classes]
    df = pd.read_csv(args.artifacts / "manifest.csv")
    X = np.load(args.artifacts / "features.npy").astype(np.float32)
    y = df[f"class_{args.classes}"].map({c: i for i, c in enumerate(labels)}).values

    idx = {s: (df.split == s).values for s in ("train", "val", "test")}
    Y = keras.utils.to_categorical(y, len(labels))

    res = {"n_classes": args.classes, "labels": labels,
           "baselines": baselines(y[idx["train"]], y[idx["test"]], len(labels), rng)}
    print("\nbaselines:", json.dumps(res["baselines"], indent=2))

    cw = compute_class_weight("balanced", classes=np.arange(len(labels)),
                              y=y[idx["train"]])
    class_weight = dict(enumerate(cw))
    builder = make_builder(X.shape[1:], len(labels))
    data = (X[idx["train"]], Y[idx["train"]], X[idx["val"]], Y[idx["val"]])

    tuners, histories = {}, {}
    for kind in ("bayesian", "random"):
        print(f"\n=== {kind} search, {args.trials} trials ===")
        tuners[kind], histories[kind] = run_search(
            kind, builder, data, args.trials, args.epochs, class_weight, out, args.seed)
        print(f"  best val macro F1: {histories[kind][-1]:.4f}")
    res["search"] = {k: {"best_val_macro_f1": h[-1], "history": h}
                     for k, h in histories.items()}
    plot_search(histories, out / "search_comparison.png")

    winner = max(histories, key=lambda k: histories[k][-1])
    best_hp = tuners[winner].get_best_hyperparameters(1)[0]
    res["winner"] = {"search": winner, "hyperparameters": best_hp.values}
    print(f"\nwinner: {winner} -> {best_hp.values}")

    # retrain on train+val, evaluate once on test
    fit_idx = idx["train"] | idx["val"]
    model = builder(best_hp)
    model.fit(X[fit_idx], Y[fit_idx], epochs=args.epochs, batch_size=64,
              class_weight=class_weight, verbose=2)

    pred = model.predict(X[idx["test"]], verbose=0).argmax(1)
    true = y[idx["test"]]
    res["test"] = {"accuracy": float(accuracy_score(true, pred)),
                   "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0))}
    print("\ntest:", res["test"])
    print(classification_report(true, pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(true, pred, labels=range(len(labels)))
    plot_confusion(cm, labels, out / "confusion_matrix.png")
    res["confusion_matrix"] = cm.tolist()

    # per-camera slice: reveals cameras where the model quietly fails
    test_df = df[idx["test"]].copy(); test_df["pred"] = pred; test_df["true"] = true
    slices = camera_slices(test_df, labels)
    slices.to_csv(out / "per_camera.csv")
    cols = ["n", "accuracy", "macro_f1", "min_class_n"]
    print("\nworst cameras:\n", slices[cols].head(5))
    thin = slices[slices.min_class_n <= 2]
    if len(thin):
        print(f"\nNOTE: {len(thin)} camera(s) have <=2 test images of some class - "
              f"their macro F1 is unstable: {', '.join(thin.index)}")

    model.save(out / "model.keras")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    shutil.rmtree(out / "tuner", ignore_errors=True)
    print(f"\nwrote {out}/  (model.keras, results.json, 2 png, per_camera.csv)")


if __name__ == "__main__":
    main()
