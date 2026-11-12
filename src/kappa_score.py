"""
Inter-annotator agreement: Cohen's Kappa (pairwise) & Fleiss' Kappa (multi-rater)
per aspek ABSA.

Output:
  outputs/kappa_cohens_pairwise.csv   - Cohen's Kappa tiap pasangan annotator × aspek
  outputs/kappa_fleiss.csv            - Fleiss' Kappa per aspek
  outputs/figures/kappa_heatmap.png   - heatmap Cohen's Kappa
  outputs/figures/kappa_fleiss_bar.png- bar chart Fleiss' Kappa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

OUTPUT_DIR = 'outputs'
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

ASPECT_MAP = {
    'kebersihan':       'Cleanliness',
    'pelayanan':        'Service',
    'fasilitas':        'Facilities',
    'lokasi':           'Location',
    'harga':            'Price',
    'kenyamanan_kamar': 'Room Comfort',
    'overall_sentiment':'Overall Sentiment',
}
ASPECTS_ID = list(ASPECT_MAP.keys())
ASPECTS_EN = list(ASPECT_MAP.values())

# interpretasi kappa (Landis & Koch 1977)
def kappa_label(k):
    if k < 0:      return 'Poor'
    if k < 0.20:   return 'Slight'
    if k < 0.40:   return 'Fair'
    if k < 0.60:   return 'Moderate'
    if k < 0.80:   return 'Substantial'
    return 'Almost Perfect'


def load_annotations():
    df = pd.read_csv('data/raw/annotations.csv')
    # gunakan annotator id (kolom annotator berisi user_id integer dari bulk export)
    df['annotator'] = df['annotator'].astype(str)
    return df


def pivot_per_aspect(df, aspect_col):
    """
    Pivot ke format wide: index=task_id, columns=annotator, values=label.
    Hanya ambil task yang punya setidaknya 2 annotator untuk aspek ini.
    """
    sub = df[['task_id', 'annotator', aspect_col]].dropna(subset=[aspect_col])
    # satu annotator bisa punya duplikat per task? ambil yang pertama
    sub = sub.drop_duplicates(subset=['task_id', 'annotator'])
    wide = sub.pivot(index='task_id', columns='annotator', values=aspect_col)
    # hanya baris yang punya >= 2 annotator
    wide = wide[wide.notna().sum(axis=1) >= 2]
    return wide


# ── Cohen's Kappa pairwise ────────────────────────────────────────────────────
def compute_cohens_kappa(df):
    rows = []
    annotators = sorted(df['annotator'].unique())
    pairs = list(combinations(annotators, 2))

    for asp_id, asp_en in ASPECT_MAP.items():
        wide = pivot_per_aspect(df, asp_id)
        for a1, a2 in pairs:
            if a1 not in wide.columns or a2 not in wide.columns:
                continue
            sub = wide[[a1, a2]].dropna()
            if len(sub) < 10:
                continue
            try:
                kappa = cohen_kappa_score(sub[a1], sub[a2])
            except Exception:
                kappa = np.nan
            rows.append({
                'aspect': asp_en,
                'annotator_1': a1,
                'annotator_2': a2,
                'n_tasks': len(sub),
                'cohens_kappa': round(kappa, 4),
                'interpretation': kappa_label(kappa),
            })

    df_kappa = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'kappa_cohens_pairwise.csv')
    df_kappa.to_csv(path, index=False)
    print(f'Saved: {path}')
    return df_kappa


# ── Fleiss' Kappa ─────────────────────────────────────────────────────────────
def compute_fleiss_kappa(df):
    rows = []
    for asp_id, asp_en in ASPECT_MAP.items():
        wide = pivot_per_aspect(df, asp_id)
        # aggregate_raters butuh array (n_tasks, n_raters) dengan nilai kategorik
        # pakai kolom yang ada isinya, fill NaN dengan nilai paling umum
        rater_cols = wide.columns.tolist()
        mat = wide[rater_cols].copy()

        # hanya baris lengkap (semua rater mengisi) untuk Fleiss yang strict
        mat_complete = mat.dropna()
        if len(mat_complete) < 10:
            rows.append({'aspect': asp_en, 'n_tasks': len(mat_complete),
                         'fleiss_kappa': np.nan, 'interpretation': 'Insufficient data'})
            continue

        try:
            # aggregate_raters: konversi ke integer encoding
            agg, _ = aggregate_raters(mat_complete.values)
            fk = fleiss_kappa(agg)
        except Exception as e:
            fk = np.nan

        rows.append({
            'aspect': asp_en,
            'n_tasks': len(mat_complete),
            'fleiss_kappa': round(fk, 4),
            'interpretation': kappa_label(fk),
        })

    df_fleiss = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'kappa_fleiss.csv')
    df_fleiss.to_csv(path, index=False)
    print(f'Saved: {path}')
    print('\nFleiss Kappa per aspect:')
    print(df_fleiss.to_string(index=False))
    return df_fleiss


# ── Figure: Cohen's Kappa heatmap ─────────────────────────────────────────────
def plot_cohens_heatmap(df_kappa):
    if df_kappa.empty:
        return
    pairs = df_kappa[['annotator_1', 'annotator_2']].drop_duplicates()
    pair_labels = [f"{r.annotator_1} vs {r.annotator_2}" for _, r in pairs.iterrows()]
    matrix = np.full((len(pair_labels), len(ASPECTS_EN)), np.nan)

    for i, (_, row) in enumerate(pairs.iterrows()):
        for j, asp in enumerate(ASPECTS_EN):
            match = df_kappa[(df_kappa['annotator_1'] == row.annotator_1) &
                             (df_kappa['annotator_2'] == row.annotator_2) &
                             (df_kappa['aspect'] == asp)]
            if not match.empty:
                matrix[i, j] = match['cohens_kappa'].values[0]

    fig, ax = plt.subplots(figsize=(11, max(4, len(pair_labels) * 0.7 + 1.5)))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-0.1, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(ASPECTS_EN)))
    ax.set_xticklabels(ASPECTS_EN, rotation=25, ha='right', fontsize=10)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)
    for i in range(len(pair_labels)):
        for j in range(len(ASPECTS_EN)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                        color='black' if 0.2 < val < 0.85 else 'white')
    plt.colorbar(im, ax=ax, label="Cohen's Kappa")
    ax.set_title("Cohen's Kappa — Pairwise Inter-annotator Agreement", fontweight='bold', pad=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'kappa_cohens_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── Figure: Fleiss' Kappa bar chart ───────────────────────────────────────────
def plot_fleiss_bar(df_fleiss):
    INTERP_COLORS = {
        'Poor': '#EF5350', 'Slight': '#FF7043', 'Fair': '#FFA726',
        'Moderate': '#FFEE58', 'Substantial': '#66BB6A', 'Almost Perfect': '#26A69A',
        'Insufficient data': '#BDBDBD',
    }
    df_plot = df_fleiss.dropna(subset=['fleiss_kappa']).copy()
    colors = [INTERP_COLORS.get(i, '#888') for i in df_plot['interpretation']]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df_plot['aspect'], df_plot['fleiss_kappa'], color=colors, width=0.55)
    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{row['fleiss_kappa']:.3f}\n({row['interpretation']})",
                ha='center', va='bottom', fontsize=8.5)

    ax.axhline(0.0,  color='#EF5350', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0.20, color='#FFA726', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0.40, color='#FFEE58', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0.60, color='#66BB6A', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0.80, color='#26A69A', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_ylim(-0.05, min(1.05, df_plot['fleiss_kappa'].max() + 0.18))
    ax.set_xticklabels(df_plot['aspect'], rotation=20, ha='right')
    ax.set_ylabel("Fleiss' Kappa")
    ax.set_title("Fleiss' Kappa per Aspect (Landis & Koch 1977)", fontweight='bold', pad=12)

    legend_items = [mpatches.Patch(color=c, label=l) for l, c in INTERP_COLORS.items()
                    if l != 'Insufficient data']
    ax.legend(handles=legend_items, title='Interpretation', bbox_to_anchor=(1.01, 1),
              loc='upper left', frameon=False, fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'kappa_fleiss_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading annotations...')
    df = load_annotations()
    print(f'  {len(df)} annotation rows | {df["annotator"].nunique()} annotators\n')

    print("Computing Cohen's Kappa (pairwise)...")
    df_kappa = compute_cohens_kappa(df)

    print("\nComputing Fleiss' Kappa...")
    df_fleiss = compute_fleiss_kappa(df)

    print('\nGenerating figures...')
    plot_cohens_heatmap(df_kappa)
    plot_fleiss_bar(df_fleiss)

    print('\nSelesai.')


if __name__ == '__main__':
    main()
