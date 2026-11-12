"""
05_generate_figures.py
Generate publication figures for RNN comparison paper.
Font: CMU Serif, 300 DPI.

Figures (per embedding mode × experiment):
  F1a — Learning curve frozen   — aspect
  F1b — Learning curve frozen   — overall
  F1c — Learning curve unfrozen — aspect
  F1d — Learning curve unfrozen — overall
  F2a — Confusion matrix frozen  (aspect & overall side-by-side)
  F2b — Confusion matrix unfrozen (aspect & overall side-by-side)

Usage:
  python rnn_absa/05_generate_figures.py            # semua (default)
  python rnn_absa/05_generate_figures.py --freeze   # hanya frozen
  python rnn_absa/05_generate_figures.py --unfreeze # hanya unfrozen
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')

# ── CMU Serif ──────────────────────────────────────────────────────────────────
_MPLDATA = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
for _v in ['CMUSerif-Roman.otf', 'CMUSerif-Bold.otf',
           'CMUSerif-Italic.otf', 'CMUSerif-BoldItalic.otf']:
    _fp = os.path.join(_MPLDATA, _v)
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)

plt.rcParams.update({
    'font.family':        'CMU Serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
})

ROOT     = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(ROOT, 'configs', 'experiment.yaml')

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

EMB_FREEZE = cfg['embedding']['freeze']
SEEDS      = cfg['experiment']['seeds']
MODELS     = cfg['models']
MLABELS    = {'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU',
              'birnn': 'BiRNN', 'bilstm': 'BiLSTM', 'bigru': 'BiGRU'}
EXPS       = ['aspect', 'overall']
EXP_LABEL  = {'aspect': 'Aspect-Level ABSA', 'overall': 'Overall Sentiment'}
LABELS     = cfg['data']['labels']            # positive, negative, neutral

# 6 warna: unidirectional (terang) → bidirectional (gelap)
COLORS = {
    'rnn':    '#5B8DB8',
    'lstm':   '#E9A84A',
    'gru':    '#6BBF85',
    'birnn':  '#264653',
    'bilstm': '#C47A28',
    'bigru':  '#1D6A5F',
}
LINE_STYLE = {
    'rnn':    (None, None),
    'lstm':   (None, None),
    'gru':    (None, None),
    'birnn':  ([6, 2], None),
    'bilstm': ([6, 2], None),
    'bigru':  ([6, 2], None),
}

# set at runtime
LOG_DIR       = ''
MET_DIR       = ''
PRED_DIR      = ''
FIG_DIR       = ''
EMB_MODE_LABEL = ''   # 'Frozen' or 'Unfrozen'


def set_dirs(freeze_embedding: bool):
    global LOG_DIR, MET_DIR, PRED_DIR, FIG_DIR, EMB_MODE_LABEL
    emb_mode      = 'frozen' if freeze_embedding else 'unfrozen'
    EMB_MODE_LABEL = 'Frozen' if freeze_embedding else 'Unfrozen'
    LOG_DIR  = os.path.join(ROOT, 'outputs', 'logs',        emb_mode)
    MET_DIR  = os.path.join(ROOT, 'outputs', 'metrics',     emb_mode)
    PRED_DIR = os.path.join(ROOT, 'outputs', 'predictions', emb_mode)
    FIG_DIR  = os.path.join(ROOT, 'outputs', 'figures',     emb_mode)
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'Embedding mode : {"frozen" if freeze_embedding else "unfrozen"}')
    print(f'Figures dir    : {FIG_DIR}')


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f'  saved → {path}')


# ── helpers ───────────────────────────────────────────────────────────────────

def load_epoch_logs(experiment: str) -> dict:
    """Return {arch: DataFrame(epoch, train_loss_mean, train_loss_std,
                                       val_f1_mean, val_f1_std)}"""
    result = {}
    exp_log_dir = os.path.join(LOG_DIR, experiment)
    for arch in MODELS:
        seed_dfs = []
        for seed in SEEDS:
            path = os.path.join(exp_log_dir, f'{arch}_seed{seed}_epoch_log.csv')
            if os.path.exists(path):
                seed_dfs.append(pd.read_csv(path).set_index('epoch'))
        if not seed_dfs:
            continue

        max_ep = max(df.index.max() for df in seed_dfs)
        padded = []
        for df in seed_dfs:
            last = df.iloc[-1]
            full = df.reindex(range(1, max_ep + 1))
            full = full.ffill()          # pad with last value after early stop
            padded.append(full)

        stacked_loss = np.stack([df['train_loss'].values for df in padded])
        stacked_f1   = np.stack([df['val_macro_f1'].values for df in padded])
        epochs = np.arange(1, max_ep + 1)

        result[arch] = {
            'epochs':          epochs,
            'loss_mean':       stacked_loss.mean(axis=0),
            'loss_std':        stacked_loss.std(axis=0),
            'val_f1_mean':     stacked_f1.mean(axis=0),
            'val_f1_std':      stacked_f1.std(axis=0),
        }
    return result


def load_best_arch_cms(experiment: str, label_order: list):
    """Return (arch_name, cm_avg) — confusion matrix averaged across all seeds
    for the best architecture (highest test mean macro-F1)."""
    summary_path = os.path.join(MET_DIR, experiment, 'multiseed_summary.csv')
    if not os.path.exists(summary_path):
        return None, None
    summary   = pd.read_csv(summary_path)
    best_row  = summary.loc[summary['test_mean_macro_f1'].idxmax()]
    best_arch = best_row['arch']

    cms = []
    for seed in SEEDS:
        pred_path = os.path.join(PRED_DIR, f'pred_{best_arch}_{experiment}_seed{seed}_test.csv')
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        cms.append(confusion_matrix(df['sentiment'], df['predicted_sentiment'],
                                    labels=label_order))
    if not cms:
        return best_arch, None

    cm_avg = np.mean(np.stack(cms, axis=0), axis=0)
    return best_arch, cm_avg


# ── F1: Learning curve ────────────────────────────────────────────────────────

def fig_learning_curve(experiment: str):
    logs = load_epoch_logs(experiment)
    if not logs:
        print(f'  [F1-{experiment}] epoch logs tidak ditemukan — skip')
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax_loss, ax_f1 = axes

    for arch, d in logs.items():
        ep    = d['epochs']
        color = COLORS[arch]
        label = MLABELS[arch]
        dash, _= LINE_STYLE[arch]
        lw = 1.8

        # train loss
        ax_loss.plot(ep, d['loss_mean'], color=color, linewidth=lw,
                     dashes=dash if dash else [], label=label)
        ax_loss.fill_between(ep,
                             d['loss_mean'] - d['loss_std'],
                             d['loss_mean'] + d['loss_std'],
                             color=color, alpha=0.22)

        # val macro-F1
        ax_f1.plot(ep, d['val_f1_mean'], color=color, linewidth=lw,
                   dashes=dash if dash else [], label=label)
        ax_f1.fill_between(ep,
                           d['val_f1_mean'] - d['val_f1_std'],
                           d['val_f1_mean'] + d['val_f1_std'],
                           color=color, alpha=0.22)

    for ax, ylabel, title in [
        (ax_loss, 'Training Loss',       'Training Loss'),
        (ax_f1,   'Validation Macro-F1', 'Validation Macro-F1'),
    ]:
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # grid penuh (horizontal + vertikal) seperti contoh
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.55, color='#bbbbbb')
        ax.set_axisbelow(True)
        # semua spines visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
        ax.text(0.02, 0.03, f'Shaded: mean ± 1 std ({len(SEEDS)} seeds)',
                transform=ax.transAxes, fontsize=8, color='gray', va='bottom')

    handles, lbls = ax_f1.get_legend_handles_labels()
    ax_f1.legend(handles, lbls, loc='lower right',
                 frameon=True, framealpha=0.85, edgecolor='#cccccc',
                 fontsize=8.5, ncol=1)

    exp_title = EXP_LABEL[experiment]
    fig.suptitle(f'Learning Curves — {exp_title} ({EMB_MODE_LABEL})', fontsize=11, y=1.02)
    fig.tight_layout(w_pad=2.5)
    savefig(f'fig1_learning_curve_{experiment}.png')


# ── F2: Confusion matrix ──────────────────────────────────────────────────────

def fig_confusion_matrix():
    fig, axes   = plt.subplots(1, 2, figsize=(11, 4.5))
    label_order = ['positive', 'negative', 'neutral']
    tick_labels = ['Positive', 'Negative', 'Neutral']
    n_seeds     = len(SEEDS)

    for ax, exp in zip(axes, EXPS):
        arch, cm_avg = load_best_arch_cms(exp, label_order)
        if cm_avg is None:
            ax.set_title(f'{EXP_LABEL[exp]}\n(data tidak ditemukan)')
            ax.axis('off')
            continue

        cm_norm = cm_avg / cm_avg.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

        # ticks & labels — tidak dirotasi, seperti contoh
        ax.set_xticks(range(len(label_order)))
        ax.set_yticks(range(len(label_order)))
        ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=10)
        ax.set_yticklabels(tick_labels, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)

        # title dua baris: model + keterangan run
        ax.set_title(
            f'{MLABELS[arch]} — {EXP_LABEL[exp]}\n'
            f'({EMB_MODE_LABEL}, Avg. {n_seeds} Seeds)',
            fontsize=12, pad=10
        )

        # spines visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')

        # nilai di tiap cell (hanya desimal, tanpa raw count)
        for i in range(len(label_order)):
            for j in range(len(label_order)):
                val   = cm_norm[i, j]
                color = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center', fontsize=11,
                        fontweight='normal', color=color)

        # colorbar
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9)

    fig.tight_layout(w_pad=3.5)
    savefig('fig2_confusion_matrix.png')


def run_for_mode(freeze_embedding: bool):
    set_dirs(freeze_embedding)
    print('\nGenerating figures (CMU Serif, 300 DPI)...')
    for exp in EXPS:
        fig_learning_curve(exp)
        print(f'[F1-{exp}] done')
    fig_confusion_matrix()
    print('[F2] confusion matrix done')
    print(f'All figures saved to: {FIG_DIR}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    emb_group = parser.add_mutually_exclusive_group()
    emb_group.add_argument('--freeze',   dest='mode', action='store_const', const='freeze')
    emb_group.add_argument('--unfreeze', dest='mode', action='store_const', const='unfreeze')
    parser.set_defaults(mode='both')
    args = parser.parse_args()

    modes = {
        'freeze':   [True],
        'unfreeze': [False],
        'both':     [True, False],
    }[args.mode]

    for freeze in modes:
        run_for_mode(freeze)
