"""
Tahap 1-3: Build review-aspect pairs, split 70:15:15, validasi distribusi.
Output: data/processed/{train,validation,test}.csv + data/manifests/split_manifest.json
"""

import sys
sys.path.insert(0, '.')

from src.data.load_data   import load_pairs, load_config
from src.data.split_data  import split_dataset, save_splits
from src.data.validate_data import print_report

cfg = load_config('configs/experiment.yaml')

print('=== 01. PREPARE DATA ===')
print('\n[1] Membangun review-aspect pairs...')
df = load_pairs('data/raw/dataset.csv')
print(f'    Total pairs: {len(df)} | Unique reviews: {df["review_id"].nunique()}')
print(df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0))

print('\n[2] Split 70:15:15 berdasarkan review_id...')
split_cfg = cfg['experiment']['split']
df_train, df_val, df_test = split_dataset(
    df,
    train_ratio = split_cfg['train'],
    val_ratio   = split_cfg['validation'],
    test_ratio  = split_cfg['test'],
    group_col   = split_cfg['group_column'],
    seed        = cfg['experiment']['seed'],
)
save_splits(df_train, df_val, df_test, cfg)

print('\n[3] Validasi distribusi...')
print_report(df_train, df_val, df_test)

print('\n[OK] Data siap.')
