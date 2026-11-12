"""
Exploratory Data Analysis — Hotel Malang ABSA Dataset
Output disimpan di folder outputs/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from collections import Counter

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
LABEL_MAP = {
    'Positif':           'Positive',
    'Negatif':           'Negative',
    'Netral':            'Neutral',
    'Tidak Disebutkan':  'Not Mentioned',
}
ASPECTS = list(ASPECT_MAP.keys())
LABEL_ORDER = ['Positive', 'Negative', 'Neutral', 'Not Mentioned']
LABEL_COLORS = {
    'Positive':      '#4CAF50',
    'Negative':      '#F44336',
    'Neutral':       '#FF9800',
    'Not Mentioned': '#9E9E9E',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_data():
    df = pd.read_csv('data/raw/dataset.csv')
    df_ann = pd.read_csv('data/raw/annotations.csv')

    # rename kolom ke English
    df = df.rename(columns={**ASPECT_MAP, 'Kecamatan': 'District', 'Nama Hotel': 'Hotel Name', 'Review': 'Review'})
    for asp_en in ASPECT_MAP.values():
        if asp_en in df.columns:
            df[asp_en] = df[asp_en].map(LABEL_MAP).fillna(df[asp_en])

    return df, df_ann


# ── 1. Summary stats ─────────────────────────────────────────────────────────
def save_summary(df, df_ann):
    lines = []
    lines.append('=== DATASET SUMMARY ===')
    lines.append(f'Total reviews       : {len(df)}')
    lines.append(f'Unique hotels       : {df["Hotel Name"].nunique()}')
    lines.append(f'Unique districts    : {df["District"].nunique()}')
    lines.append(f'Total annotations   : {len(df_ann)}')
    lines.append(f'Avg annotators/task : {df["n_annotators"].mean():.2f}')
    lines.append('')

    lines.append('=== REVIEW LENGTH (characters) ===')
    lengths = df['Review'].str.len()
    lines.append(f'Min    : {lengths.min()}')
    lines.append(f'Max    : {lengths.max()}')
    lines.append(f'Mean   : {lengths.mean():.1f}')
    lines.append(f'Median : {lengths.median():.1f}')
    lines.append('')

    lines.append('=== LABEL DISTRIBUTION PER ASPECT ===')
    for asp_id, asp_en in ASPECT_MAP.items():
        lines.append(f'\n{asp_en}:')
        vc = df[asp_en].value_counts()
        for label, cnt in vc.items():
            lines.append(f'  {label:<18}: {cnt:>5} ({cnt/len(df)*100:.1f}%)')

    lines.append('\n=== REVIEWS PER DISTRICT ===')
    for district, cnt in df['District'].value_counts().items():
        lines.append(f'  {district:<20}: {cnt}')

    path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {path}')


# ── 2. Overall sentiment distribution ────────────────────────────────────────
def plot_overall_sentiment(df):
    vc = df['Overall Sentiment'].value_counts()
    colors = [LABEL_COLORS.get(l, '#888') for l in vc.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(vc.index, vc.values, color=colors, width=0.5)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax.set_title('Overall Sentiment Distribution', fontweight='bold', pad=12)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'overall_sentiment_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── 3. Label distribution per aspect (stacked bar) ───────────────────────────
def plot_aspect_distribution(df):
    aspect_cols = [v for v in ASPECT_MAP.values() if v != 'Overall Sentiment']
    data = {}
    for label in LABEL_ORDER:
        data[label] = [(df[asp] == label).sum() for asp in aspect_cols]

    x = np.arange(len(aspect_cols))
    width = 0.55
    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = np.zeros(len(aspect_cols))
    for label in LABEL_ORDER:
        vals = np.array(data[label])
        ax.bar(x, vals, width, label=label, bottom=bottoms, color=LABEL_COLORS[label])
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(aspect_cols, rotation=20, ha='right')
    ax.set_title('Label Distribution per Aspect', fontweight='bold', pad=12)
    ax.set_ylabel('Count')
    ax.legend(title='Label', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'aspect_label_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── 4. Review length distribution ────────────────────────────────────────────
def plot_review_length(df):
    lengths = df['Review'].str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=50, color='#5C6BC0', edgecolor='white', linewidth=0.4)
    ax.axvline(lengths.median(), color='#E53935', linestyle='--', linewidth=1.5, label=f'Median: {lengths.median():.0f}')
    ax.set_title('Review Length Distribution (characters)', fontweight='bold', pad=12)
    ax.set_xlabel('Number of Characters')
    ax.set_ylabel('Count')
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'review_length_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── 5. Top hotels by review count ────────────────────────────────────────────
def plot_top_hotels(df, top_n=20):
    vc = df['Hotel Name'].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(vc.index[::-1], vc.values[::-1], color='#26A69A')
    ax.set_title(f'Top {top_n} Hotels by Review Count', fontweight='bold', pad=12)
    ax.set_xlabel('Number of Reviews')
    for i, (val, name) in enumerate(zip(vc.values[::-1], vc.index[::-1])):
        ax.text(val + 0.3, i, str(val), va='center', fontsize=9)
    ax.set_yticklabels([n[:45] + '…' if len(n) > 45 else n for n in vc.index[::-1]], fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'top_hotels_by_review_count.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── 6. Reviews per district ───────────────────────────────────────────────────
def plot_district_distribution(df):
    vc = df['District'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(vc.index, vc.values, color='#AB47BC', width=0.5)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha='center', va='bottom', fontsize=10)
    ax.set_title('Reviews per District', fontweight='bold', pad=12)
    ax.set_xlabel('District')
    ax.set_ylabel('Count')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'reviews_per_district.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── 7. Aspect sentiment heatmap (% Positive) ─────────────────────────────────
def plot_aspect_heatmap(df):
    districts = df['District'].unique()
    aspect_cols = list(ASPECT_MAP.values())

    matrix = []
    for dist in districts:
        sub = df[df['District'] == dist]
        row = []
        for asp in aspect_cols:
            pct = (sub[asp] == 'Positive').sum() / len(sub) * 100
            row.append(pct)
        matrix.append(row)

    mat = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(aspect_cols)))
    ax.set_xticklabels(aspect_cols, rotation=25, ha='right', fontsize=10)
    ax.set_yticks(range(len(districts)))
    ax.set_yticklabels(districts, fontsize=10)
    for i in range(len(districts)):
        for j in range(len(aspect_cols)):
            ax.text(j, i, f'{mat[i, j]:.0f}%', ha='center', va='center',
                    fontsize=9, color='black')
    plt.colorbar(im, ax=ax, label='% Positive')
    ax.set_title('% Positive Sentiment per Aspect by District', fontweight='bold', pad=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'aspect_positive_heatmap_by_district.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── 8. Inter-annotator agreement CSV ─────────────────────────────────────────
def save_annotator_agreement(df_ann):
    df_ann = df_ann.copy()
    for asp_id, asp_en in ASPECT_MAP.items():
        if asp_id in df_ann.columns:
            df_ann[asp_en] = df_ann[asp_id].map(LABEL_MAP).fillna(df_ann[asp_id])
            df_ann = df_ann.drop(columns=[asp_id])

    aspect_en_cols = list(ASPECT_MAP.values())
    rows = []
    for task_id, grp in df_ann.groupby('task_id'):
        row = {'task_id': task_id, 'n_annotators': len(grp)}
        for asp in aspect_en_cols:
            if asp not in grp.columns:
                continue
            labels = grp[asp].dropna().tolist()
            row[f'{asp}_agreement'] = len(set(labels)) == 1 if labels else None
        rows.append(row)

    df_agree = pd.DataFrame(rows)
    agree_cols = [c for c in df_agree.columns if c.endswith('_agreement')]
    summary = df_agree[agree_cols].mean() * 100
    summary.index = [c.replace('_agreement', '') for c in summary.index]
    summary.name = 'agreement_pct'
    summary_df = summary.reset_index().rename(columns={'index': 'aspect'})

    path_detail = os.path.join(OUTPUT_DIR, 'annotator_agreement_detail.csv')
    path_summary = os.path.join(OUTPUT_DIR, 'annotator_agreement_summary.csv')
    df_agree.to_csv(path_detail, index=False)
    summary_df.to_csv(path_summary, index=False)
    print(f'Saved: {path_detail}')
    print(f'Saved: {path_summary}')
    print('\nInter-annotator agreement (% full agreement):')
    print(summary_df.to_string(index=False))


# ── 9. Label distribution CSV ─────────────────────────────────────────────────
def save_label_distribution_csv(df):
    rows = []
    for asp_en in ASPECT_MAP.values():
        if asp_en not in df.columns:
            continue
        vc = df[asp_en].value_counts()
        for label, cnt in vc.items():
            rows.append({'aspect': asp_en, 'label': label, 'count': cnt,
                         'percentage': round(cnt / len(df) * 100, 2)})
    df_dist = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'label_distribution.csv')
    df_dist.to_csv(path, index=False)
    print(f'Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading data...')
    df, df_ann = load_data()
    print(f'Dataset: {len(df)} rows | Annotations: {len(df_ann)} rows\n')

    save_summary(df, df_ann)
    save_label_distribution_csv(df)
    save_annotator_agreement(df_ann)

    plot_overall_sentiment(df)
    plot_aspect_distribution(df)
    plot_review_length(df)
    plot_top_hotels(df)
    plot_district_distribution(df)
    plot_aspect_heatmap(df)

    print('\nEDA selesai. Semua output tersimpan di folder outputs/')


if __name__ == '__main__':
    main()
