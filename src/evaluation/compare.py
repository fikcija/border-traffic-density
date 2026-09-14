"""
Collect results across backbones into one table and one chart.

    python src/compare.py --classes 3

Scans artifacts/<backbone>/run_<n>cls/ and artifacts/<backbone>/finetune_<n>cls/,
so it picks up whatever you have run - a backbone with no fine-tune stage simply
shows a blank there.
"""
import argparse, json
from pathlib import Path

import pandas as pd

from train import BLUE, ORANGE, INK, MUTED, GRID


def collect(artifacts: Path, n_classes: int) -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        frozen = d / f"run_{n_classes}cls" / "results.json"
        if not frozen.exists():
            continue
        fr = json.loads(frozen.read_text())
        row = {
            "backbone": d.name,
            "frozen_macro_f1": fr["test"]["macro_f1"],
            "frozen_accuracy": fr["test"]["accuracy"],
            "search_winner": fr["winner"]["search"],
            "bayesian_val": fr["search"]["bayesian"]["best_val_macro_f1"],
            "random_val": fr["search"]["random"]["best_val_macro_f1"],
        }
        tuned = d / f"finetune_{n_classes}cls" / "results.json"
        if tuned.exists():
            ft = json.loads(tuned.read_text())
            row["finetuned_macro_f1"] = ft["test"]["macro_f1"]
            row["finetuned_accuracy"] = ft["test"]["accuracy"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"no results found under {artifacts}/*/run_{n_classes}cls/")
    return pd.DataFrame(rows).sort_values(
        "finetuned_macro_f1" if "finetuned_macro_f1" in rows[0] else "frozen_macro_f1",
        ascending=False)


def plot(df: pd.DataFrame, baseline: float, path: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = df.iloc[::-1]                       # best at top for a horizontal chart
    has_ft = "finetuned_macro_f1" in d and d.finetuned_macro_f1.notna().any()
    ys = np.arange(len(d))
    h = 0.34 if has_ft else 0.5
    gap = 0.02                              # surface gap between paired bars

    fig, ax = plt.subplots(figsize=(8, 0.85 * len(d) + 2.1))
    ax.barh(ys + (h / 2 + gap if has_ft else 0), d.frozen_macro_f1, height=h,
            color=BLUE, label="frozen features")
    if has_ft:
        ax.barh(ys - h / 2 - gap, d.finetuned_macro_f1.fillna(0), height=h,
                color=ORANGE, label="fine-tuned")

    for y, row in zip(ys, d.itertuples()):
        ax.text(row.frozen_macro_f1 + 0.008, y + (h / 2 + gap if has_ft else 0),
                f"{row.frozen_macro_f1:.3f}", va="center", fontsize=9, color=MUTED)
        if has_ft and pd.notna(getattr(row, "finetuned_macro_f1", None)):
            ax.text(row.finetuned_macro_f1 + 0.008, y - h / 2 - gap,
                    f"{row.finetuned_macro_f1:.3f}", va="center", fontsize=9,
                    color=MUTED)

    ax.axvline(baseline, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"zero-rule baseline {baseline:.2f}", (baseline, 1.01),
                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                fontsize=9, color=MUTED)

    ax.set_yticks(ys, d.backbone)
    ax.set_xlabel("test macro F1")
    ax.set_xlim(0, min(1.0, max(d.frozen_macro_f1.max(),
                                d.get("finetuned_macro_f1", d.frozen_macro_f1).max())
                       * 1.18))
    ax.set_title("Backbone comparison", color=INK, fontsize=11, loc="left", pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    if has_ft:
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--classes", type=int, choices=[3, 4], default=3)
    args = ap.parse_args()

    df = collect(args.artifacts, args.classes)
    first = json.loads(next(args.artifacts.glob(
        f"*/run_{args.classes}cls/results.json")).read_text())
    baseline = first["baselines"]["zero_rule"]["macro_f1"]

    out = args.artifacts / f"comparison_{args.classes}cls"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "backbones.csv", index=False)
    plot(df, baseline, out / "backbones.png")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\nzero-rule baseline macro F1: {baseline:.4f}\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out}/backbones.csv and backbones.png")


if __name__ == "__main__":
    main()
