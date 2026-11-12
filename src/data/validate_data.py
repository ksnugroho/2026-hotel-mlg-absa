"""
Validasi distribusi dataset setelah split.
"""

import pandas as pd
import json


def check_leakage(df_train, df_val, df_test) -> bool:
    ids_train = set(df_train['review_id'])
    ids_val   = set(df_val['review_id'])
    ids_test  = set(df_test['review_id'])

    ok = True
    if ids_train & ids_val:
        print(f'LEAKAGE: {len(ids_train & ids_val)} review_id overlap train-val')
        ok = False
    if ids_train & ids_test:
        print(f'LEAKAGE: {len(ids_train & ids_test)} review_id overlap train-test')
        ok = False
    if ids_val & ids_test:
        print(f'LEAKAGE: {len(ids_val & ids_test)} review_id overlap val-test')
        ok = False
    if ok:
        print('Leakage check: OK — tidak ada review_id yang overlap antar subset.')
    return ok


def distribution_report(df_train, df_val, df_test) -> pd.DataFrame:
    rows = []
    for name, df in [('train', df_train), ('validation', df_val), ('test', df_test)]:
        for aspect in df['aspect'].unique():
            sub = df[df['aspect'] == aspect]
            for label in ['positive', 'negative', 'neutral']:
                cnt = (sub['sentiment'] == label).sum()
                rows.append({
                    'subset': name, 'aspect': aspect,
                    'label': label, 'count': cnt,
                    'pct': round(cnt / len(sub) * 100, 1) if len(sub) > 0 else 0,
                })
    return pd.DataFrame(rows)


def print_report(df_train, df_val, df_test) -> None:
    print('\n=== DISTRIBUSI SPLIT ===')
    for name, df in [('TRAIN', df_train), ('VAL', df_val), ('TEST', df_test)]:
        print(f'\n{name} ({len(df)} pairs, {df["review_id"].nunique()} reviews):')
        print(df['sentiment'].value_counts().to_string())
        print('Per aspek:')
        for asp in sorted(df['aspect'].unique()):
            sub = df[df['aspect'] == asp]
            dist = sub['sentiment'].value_counts().to_dict()
            print(f'  {asp:<20}: {dist}')

    check_leakage(df_train, df_val, df_test)
