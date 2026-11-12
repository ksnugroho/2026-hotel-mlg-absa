"""
Memuat dataset.csv dan membangun review-aspect pairs sesuai experiment design.

Label mapping:
  Positif          → positive
  Negatif          → negative
  Netral           → neutral
  Tidak Disebutkan → dikecualikan (bukan target prediksi)
"""

import pandas as pd
import yaml
from pathlib import Path

LABEL_MAP = {
    'Positif': 'positive',
    'Negatif': 'negative',
    'Netral':  'neutral',
}
EXCLUDE_LABEL = 'Tidak Disebutkan'

ASPECTS = [
    'kebersihan', 'pelayanan', 'fasilitas',
    'lokasi', 'harga', 'kenyamanan_kamar', 'overall_sentiment',
]


def load_config(config_path: str = 'configs/experiment.yaml') -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_review_aspect_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ubah format wide (satu baris per ulasan) ke long (satu baris per pasangan
    ulasan-aspek). Baris dengan label 'Tidak Disebutkan' dikecualikan.
    """
    valid_aspects = [a for a in ASPECTS if a in df.columns]
    rows = []
    for _, row in df.iterrows():
        for aspect in valid_aspects:
            label_id = row[aspect]
            if pd.isna(label_id) or label_id == EXCLUDE_LABEL:
                continue
            label_en = LABEL_MAP.get(label_id)
            if label_en is None:
                continue
            rows.append({
                'sample_id':   f"{row['task_id']}_{aspect}",
                'review_id':   row['task_id'],
                'hotel_name':  row['Nama Hotel'],
                'district':    row['Kecamatan'],
                'review_text': row['Review'],
                'aspect':      aspect,
                'sentiment':   label_en,
                'input_text':  f"[ASPEK] {aspect} [ULASAN] {row['Review']}",
            })
    return pd.DataFrame(rows)


def load_pairs(dataset_path: str = 'data/raw/dataset.csv') -> pd.DataFrame:
    df_raw = pd.read_csv(dataset_path)
    df_pairs = build_review_aspect_pairs(df_raw)
    return df_pairs


if __name__ == '__main__':
    df = load_pairs()
    print(f'Total review-aspect pairs : {len(df)}')
    print(f'Unique reviews            : {df["review_id"].nunique()}')
    print(f'\nDistribusi aspek:')
    print(df['aspect'].value_counts())
    print(f'\nDistribusi sentimen:')
    print(df['sentiment'].value_counts())
