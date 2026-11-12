"""
07_generate_figures.py
Generate publication-ready figures from experiment results.
Font: CMU Serif (Computer Modern Unicode), 300 DPI.

Figures produced:
  F1  — Overall model comparison (Accuracy, Macro-F1, Weighted-F1) — bar chart
  F2  — Per-aspect Macro-F1: 3 models × 7 aspects — grouped bar chart
  F3  — Radar chart: per-aspect Macro-F1 per model
  F4  — IndoBERT learning curve: best config (val Macro-F1 + train loss)
  F5  — IndoBERT hyperparameter heatmap: LR × epochs, colour = val Macro-F1
  F6  — Per-class F1: positive/neutral/negative per model — grouped bar chart
  F7  — Significance test discordant pairs — bar chart
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ── register CMU Serif ───────────────────────────────────────────────────────
_MPLDATA = os.path.join(
    os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"
)
for _variant in ["CMUSerif-Roman.otf", "CMUSerif-Bold.otf",
                 "CMUSerif-Italic.otf", "CMUSerif-BoldItalic.otf"]:
    _fp = os.path.join(_MPLDATA, _variant)
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)

plt.rcParams.update({
    "font.family":        "CMU Serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

# ── paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "outputs", "metrics")
LOGS    = os.path.join(ROOT, "outputs", "logs")
FIGS    = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGS, exist_ok=True)

# ── palette & labels ─────────────────────────────────────────────────────────
COLORS       = ["#5B8DB8", "#F4A261", "#2A9D8F"]
MODEL_LABELS = ["Lexicon (InSet)", "TF-IDF + SVM", "IndoBERT-base-p2"]
MODEL_KEYS   = ["lexicon", "svm", "indobert"]

ASPECTS = [
    "fasilitas", "harga", "kebersihan",
    "kenyamanan_kamar", "lokasi", "overall_sentiment", "pelayanan",
]
ASPECT_LABELS = {
    "fasilitas":        "Fasilitas",
    "harga":            "Harga",
    "kebersihan":       "Kebersihan",
    "kenyamanan_kamar": "Kenyamanan Kamar",
    "lokasi":           "Lokasi",
    "overall_sentiment":"Overall Sentiment",
    "pelayanan":        "Pelayanan",
}

# ── helpers ──────────────────────────────────────────────────────────────────
def load_overall(split="test"):
    rows = []
    for key in MODEL_KEYS:
        df = pd.read_csv(os.path.join(METRICS, f"{key}_{split}_overall.csv"))
        rows.append(df.iloc[0].to_dict())
    return pd.DataFrame(rows)

def load_per_aspect(split="test"):
    return {
        key: pd.read_csv(
            os.path.join(METRICS, f"{key}_{split}_per_aspect.csv")
        ).set_index("aspect")
        for key in MODEL_KEYS
    }

def load_per_class(split="test"):
    return {
        key: pd.read_csv(
            os.path.join(METRICS, f"{key}_{split}_per_class.csv")
        ).set_index("label")
        for key in MODEL_KEYS
    }

def savefig(name):
    path = os.path.join(FIGS, name)
    plt.savefig(path)
    plt.close()
    print(f"  saved → {path}")

def bar_label(ax, bars, fmt="{:.4f}", fontsize=8, pad=0.005, rotation=0):
    """Add value labels above each bar, no overlap with ceiling."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, rotation=rotation,
        )


# ── F1: Overall model comparison ─────────────────────────────────────────────
def fig_overall_comparison():
    df      = load_overall("test")
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    mlabels = ["Accuracy", "Macro-F1", "Weighted-F1"]
    x, w    = np.arange(len(metrics)), 0.22

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (key, label, color) in enumerate(zip(MODEL_KEYS, MODEL_LABELS, COLORS)):
        row  = df[df["model"] == key].iloc[0]
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, width=w,
                      label=label, color=color, edgecolor="white", linewidth=0.6)
        bar_label(ax, bars, fmt="{:.4f}", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance on Test Set ($n$=1,792)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig("fig1_overall_comparison.png")


# ── F2: Per-aspect Macro-F1 grouped bar ──────────────────────────────────────
def fig_per_aspect_macro_f1():
    dfs  = load_per_aspect("test")
    x, w = np.arange(len(ASPECTS)), 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (key, label, color) in enumerate(zip(MODEL_KEYS, MODEL_LABELS, COLORS)):
        vals = [dfs[key].loc[a, "macro_f1"] if a in dfs[key].index else 0.0
                for a in ASPECTS]
        bars = ax.bar(x + (i - 1) * w, vals, width=w,
                      label=label, color=color, edgecolor="white", linewidth=0.6)
        bar_label(ax, bars, fmt="{:.2f}", fontsize=7.5, pad=0.006, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([ASPECT_LABELS[a] for a in ASPECTS], fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Per-Aspect Macro-F1 on Test Set")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig("fig2_per_aspect_macro_f1.png")


# ── F3: Radar chart ───────────────────────────────────────────────────────────
def fig_radar():
    dfs    = load_per_aspect("test")
    n      = len(ASPECTS)
    angles = [k / n * 2 * np.pi for k in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7.5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([ASPECT_LABELS[a] for a in ASPECTS], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    for key, label, color in zip(MODEL_KEYS, MODEL_LABELS, COLORS):
        vals = [dfs[key].loc[a, "macro_f1"] if a in dfs[key].index else 0.0
                for a in ASPECTS]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=label, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_title("Per-Aspect Macro-F1 — Radar", pad=22, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), framealpha=0.9)
    fig.tight_layout()
    savefig("fig3_radar_per_aspect.png")


# ── F4: IndoBERT learning curve (best config) ─────────────────────────────────
def fig_learning_curve():
    epoch_log = pd.read_csv(os.path.join(LOGS, "indobert_epoch_log.csv"))
    with open(os.path.join(LOGS, "indobert_best_params.json")) as f:
        best = json.load(f)["best_params"]

    best_combo = epoch_log[
        (np.isclose(epoch_log["learning_rate"], best["learning_rate"])) &
        (epoch_log["batch_size"] == best["batch_size"]) &
        (epoch_log["max_length"] == best["max_length"])
    ].sort_values("epoch")

    if best_combo.empty:
        print("  [F4] best combo rows not found — skipping")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    ax2 = ax1.twinx()

    l1, = ax1.plot(best_combo["epoch"], best_combo["val_macro_f1"],
                   color=COLORS[2], marker="o", linewidth=2, label="Val Macro-F1")
    l2, = ax2.plot(best_combo["epoch"], best_combo["train_loss"],
                   color=COLORS[1], marker="s", linestyle="--", linewidth=2,
                   label="Train Loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Macro-F1", color=COLORS[2])
    ax2.set_ylabel("Train Loss",   color=COLORS[1])
    ax1.tick_params(axis="y", labelcolor=COLORS[2])
    ax2.tick_params(axis="y", labelcolor=COLORS[1])
    ax1.set_xticks(best_combo["epoch"].unique())

    # annotate last point
    last = best_combo.iloc[-1]
    ax1.annotate(f"{last['val_macro_f1']:.4f}",
                 xy=(last["epoch"], last["val_macro_f1"]),
                 xytext=(last["epoch"] - 0.4, last["val_macro_f1"] + 0.005),
                 fontsize=9, color=COLORS[2])

    ax1.legend(handles=[l1, l2], loc="center right", framealpha=0.9)
    cfg = (f"lr={best['learning_rate']}, bs={best['batch_size']}, "
           f"epochs={best['epochs']}, warmup={best['warmup_ratio']}, "
           f"maxlen={best['max_length']}")
    ax1.set_title(f"IndoBERT Learning Curve — Best Config\n({cfg})", fontsize=11)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    savefig("fig4_indobert_learning_curve.png")


# ── F5: Hyperparameter heatmap (LR × epochs per warmup×maxlen) ───────────────
def fig_hparam_heatmap():
    summary = pd.read_csv(os.path.join(LOGS, "indobert_tuning_summary.csv"))
    with open(os.path.join(LOGS, "indobert_best_params.json")) as f:
        best = json.load(f)["best_params"]

    combos = (summary[["warmup_ratio", "max_length"]]
              .drop_duplicates()
              .sort_values(["max_length", "warmup_ratio"])
              .values)
    n     = len(combos)
    ncols = 2
    nrows = (n + 1) // ncols

    lrs    = sorted(summary["learning_rate"].unique())
    epochs = sorted(summary["epochs"].unique())

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(12, 4.2 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for idx, (warmup, maxlen) in enumerate(combos):
        sub   = summary[(summary["warmup_ratio"] == warmup) &
                        (summary["max_length"]   == maxlen)]
        pivot = (sub.pivot_table(index="learning_rate", columns="epochs",
                                 values="best_val_macro_f1", aggfunc="max")
                    .reindex(index=lrs, columns=epochs))

        ax = axes[idx]
        im = ax.imshow(pivot.values, cmap="YlGn", vmin=0.74, vmax=0.83, aspect="auto")
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels(epochs, fontsize=10)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs], fontsize=10)
        ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel("Learning Rate", fontsize=10)
        ax.set_title(f"warmup={warmup},  max_len={int(maxlen)}", fontsize=11)

        for r in range(len(lrs)):
            for c in range(len(epochs)):
                val = pivot.values[r, c]
                if np.isnan(val):
                    continue
                is_best = (
                    np.isclose(lrs[r], best["learning_rate"]) and
                    epochs[c] == best["epochs"] and
                    np.isclose(warmup, best["warmup_ratio"]) and
                    int(maxlen) == best["max_length"]
                )
                ax.text(c, r, f"{val:.4f}",
                        ha="center", va="center", fontsize=8.5,
                        color="darkred" if is_best else "black",
                        fontweight="bold" if is_best else "normal")

        plt.colorbar(im, ax=ax, label="Val Macro-F1", shrink=0.85)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "IndoBERT Grid Search — Val Macro-F1  (bold red = best config)",
        fontsize=13, y=1.01
    )
    savefig("fig5_hparam_heatmap.png")


# ── F6: Per-class F1 ──────────────────────────────────────────────────────────
def fig_per_class_f1():
    dfs    = load_per_class("test")
    labels = ["negative", "neutral", "positive"]
    x, w   = np.arange(len(labels)), 0.22

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (key, mlabel, color) in enumerate(zip(MODEL_KEYS, MODEL_LABELS, COLORS)):
        vals = [dfs[key].loc[lbl, "f1"] if lbl in dfs[key].index else 0.0
                for lbl in labels]
        bars = ax.bar(x + (i - 1) * w, vals, width=w,
                      label=mlabel, color=color, edgecolor="white", linewidth=0.6)
        bar_label(ax, bars, fmt="{:.3f}", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Negative", "Neutral", "Positive"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1-Score on Test Set")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig("fig6_per_class_f1.png")


# ── F7: McNemar discordant pairs ──────────────────────────────────────────────
def fig_significance():
    df          = pd.read_csv(os.path.join(METRICS, "significance_mcnemar_holm.csv"))
    comparisons = [f"{r['model_a']} vs {r['model_b']}" for _, r in df.iterrows()]
    b_vals      = df["b"].tolist()   # model_a correct, model_b wrong
    c_vals      = df["c"].tolist()   # model_b correct, model_a wrong

    x, w = np.arange(len(comparisons)), 0.3

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_b = ax.bar(x - w / 2, b_vals, width=w,
                    label="Model A correct, B wrong",
                    color=COLORS[0], edgecolor="white")
    bars_c = ax.bar(x + w / 2, c_vals, width=w,
                    label="Model B correct, A wrong",
                    color=COLORS[2], edgecolor="white")

    for bar, v in list(zip(bars_b, b_vals)) + list(zip(bars_c, c_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(v), ha="center", va="bottom", fontsize=9.5)

    # significance annotation above each pair
    y_top = max(max(b_vals), max(c_vals))
    for i, row in df.iterrows():
        ax.text(i, y_top + 18,
                f"$p\\approx${row['pvalue_holm']:.3f}  *",
                ha="center", va="bottom", fontsize=9, color="darkred")

    ax.set_xticks(x)
    ax.set_xticklabels(comparisons, fontsize=10)
    ax.set_ylim(0, y_top + 60)
    ax.set_ylabel("Number of Discordant Pairs")
    ax.set_title("McNemar's Exact Test — Discordant Pairs\n(Holm-corrected, $\\alpha$=0.05)")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig("fig7_significance_mcnemar.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures (font: CMU Serif, 300 DPI)...")
    fig_overall_comparison();   print("[F1] Overall comparison       ✓")
    fig_per_aspect_macro_f1();  print("[F2] Per-aspect Macro-F1      ✓")
    fig_radar();                print("[F3] Radar chart              ✓")
    fig_learning_curve();       print("[F4] IndoBERT learning curve  ✓")
    fig_hparam_heatmap();       print("[F5] Hyperparameter heatmap   ✓")
    fig_per_class_f1();         print("[F6] Per-class F1             ✓")
    fig_significance();         print("[F7] Significance test        ✓")
    print(f"\nAll figures saved to: outputs/figures/")
