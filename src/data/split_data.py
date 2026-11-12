"""
Split review-aspect pairs menjadi train/validation/test dengan proporsi 70:15:15.
Split dilakukan berdasarkan review_id untuk mencegah data leakage.
"""

import json
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def load_config(config_path: str = 'configs/experiment.yaml') -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    group_col:   str   = 'review_id',
    seed:        int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split berdasarkan group (review_id) agar seluruh aspek dari satu ulasan
    berada pada subset yang sama.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    groups = df[group_col].values

    # step 1: pisahkan test dari (train+val)
    test_size = test_ratio
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=groups))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    # step 2: pisahkan val dari train+val
    val_relative = val_ratio / (train_ratio + val_ratio)
    groups_tv = df_trainval[group_col].values
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_idx, val_idx = next(gss_val.split(df_trainval, groups=groups_tv))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, df_test


def save_splits(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    config:   dict,
) -> None:
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/manifests').mkdir(parents=True, exist_ok=True)

    df_train.to_csv('data/processed/train.csv',      index=False)
    df_val.to_csv(  'data/processed/validation.csv', index=False)
    df_test.to_csv( 'data/processed/test.csv',       index=False)

    manifest = {
        'seed':       config['experiment']['seed'],
        'proportions': config['experiment']['split'],
        'sizes': {
            'train':      len(df_train),
            'validation': len(df_val),
            'test':       len(df_test),
        },
        'unique_reviews': {
            'train':      int(df_train['review_id'].nunique()),
            'validation': int(df_val['review_id'].nunique()),
            'test':       int(df_test['review_id'].nunique()),
        },
        'review_ids': {
            'train':      df_train['review_id'].unique().tolist(),
            'validation': df_val['review_id'].unique().tolist(),
            'test':       df_test['review_id'].unique().tolist(),
        },
    }
    with open('data/manifests/split_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print('Split disimpan:')
    for name, df in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
        print(f'  {name:12}: {len(df):5} pairs | {df["review_id"].nunique()} reviews')
