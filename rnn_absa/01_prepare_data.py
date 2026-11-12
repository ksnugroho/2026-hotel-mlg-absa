"""
01_prepare_data.py
Validate and prepare data for RNN experiments.
Reuses ../data/processed/ splits — no re-splitting needed.

Outputs:
  rnn_absa/outputs/logs/data_summary.json
"""

import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CFG_PATH  = os.path.join(os.path.dirname(__file__), 'configs', 'experiment.yaml')
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
LOG_DIR   = os.path.join(os.path.dirname(__file__), 'outputs', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

ASPECTS = cfg['data']['aspects']
OVERALL = cfg['data']['overall_aspect']
LABELS  = cfg['data']['labels']

summary = {}
for split in ['train', 'validation', 'test']:
    path = os.path.join(DATA_DIR, f'{split}.csv')
    assert os.path.exists(path), f"Missing: {path}"
    df = pd.read_csv(path)

    # aspect experiment (exclude overall_sentiment)
    df_asp = df[df['aspect'] != OVERALL].copy()
    df_asp = df_asp[df_asp['sentiment'].isin(LABELS)]

    # overall experiment
    df_ov = df[df['aspect'] == OVERALL].copy()
    df_ov = df_ov[df_ov['sentiment'].isin(LABELS)]

    summary[split] = {
        'aspect_exp': {
            'total': len(df_asp),
            'per_aspect': df_asp.groupby('aspect').size().to_dict(),
            'per_label':  df_asp.groupby('sentiment').size().to_dict(),
        },
        'overall_exp': {
            'total': len(df_ov),
            'per_label': df_ov.groupby('sentiment').size().to_dict(),
        },
    }
    print(f"[{split}] aspect={len(df_asp):,}  overall={len(df_ov):,}")

with open(os.path.join(LOG_DIR, 'data_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\nData validation passed. Summary saved to outputs/logs/data_summary.json")
