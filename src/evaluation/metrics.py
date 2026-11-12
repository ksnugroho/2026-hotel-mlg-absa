"""
Hitung metrik evaluasi: accuracy, precision, recall, F1 (macro & weighted & per-class).
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

LABELS = ['negative', 'neutral', 'positive']


def compute_metrics(y_true, y_pred, labels=LABELS) -> dict:
    return {
        'accuracy':          round(accuracy_score(y_true, y_pred), 4),
        'macro_precision':   round(precision_score(y_true, y_pred, average='macro',    labels=labels, zero_division=0), 4),
        'macro_recall':      round(recall_score(   y_true, y_pred, average='macro',    labels=labels, zero_division=0), 4),
        'macro_f1':          round(f1_score(       y_true, y_pred, average='macro',    labels=labels, zero_division=0), 4),
        'weighted_f1':       round(f1_score(       y_true, y_pred, average='weighted', labels=labels, zero_division=0), 4),
    }


def compute_per_class(y_true, y_pred, labels=LABELS) -> pd.DataFrame:
    rows = []
    for label in labels:
        rows.append({
            'label':     label,
            'precision': round(precision_score(y_true, y_pred, labels=[label], average='micro', zero_division=0), 4),
            'recall':    round(recall_score(   y_true, y_pred, labels=[label], average='micro', zero_division=0), 4),
            'f1':        round(f1_score(       y_true, y_pred, labels=[label], average='micro', zero_division=0), 4),
            'support':   int(sum(1 for t in y_true if t == label)),
        })
    return pd.DataFrame(rows)


def compute_per_aspect(df_test: pd.DataFrame, pred_col: str, labels=LABELS) -> pd.DataFrame:
    rows = []
    for aspect in sorted(df_test['aspect'].unique()):
        sub = df_test[df_test['aspect'] == aspect]
        y_true = sub['sentiment'].values
        y_pred = sub[pred_col].values
        m = compute_metrics(y_true, y_pred, labels)
        rows.append({'aspect': aspect, 'n_samples': len(sub), **m})
    return pd.DataFrame(rows)


def full_report(
    df_test:    pd.DataFrame,
    model_name: str,
    pred_col:   str,
    out_dir:    str = 'outputs/metrics',
    split:      str = 'test',
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    y_true = df_test['sentiment'].values
    y_pred = df_test[pred_col].values
    ts     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    overall    = compute_metrics(y_true, y_pred)
    per_class  = compute_per_class(y_true, y_pred)
    per_aspect = compute_per_aspect(df_test, pred_col)
    clf_report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)

    prefix = f'{out_dir}/{model_name}_{split}'

    # overall metrics → CSV + JSON
    overall_row = {'model': model_name, 'split': split, 'timestamp': ts,
                   'n_samples': len(y_true), **overall}
    pd.DataFrame([overall_row]).to_csv(f'{prefix}_overall.csv', index=False)
    with open(f'{prefix}_overall.json', 'w') as f:
        json.dump(overall_row, f, indent=2)

    # per-class & per-aspect
    per_class.to_csv( f'{prefix}_per_class.csv',  index=False)
    per_aspect.to_csv(f'{prefix}_per_aspect.csv', index=False)

    # sklearn classification report (teks lengkap)
    with open(f'{prefix}_classification_report.txt', 'w') as f:
        f.write(f'Model  : {model_name}\n')
        f.write(f'Split  : {split}\n')
        f.write(f'Time   : {ts}\n')
        f.write(f'Samples: {len(y_true)}\n\n')
        f.write(clf_report)

    print(f'\n=== {model_name.upper()} [{split}] ===')
    for k, v in overall.items():
        print(f'  {k:<20}: {v}')

    return overall
