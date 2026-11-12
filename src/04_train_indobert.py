"""
Tahap 7: Fine-tuning IndoBERT-base-p2, grid search hyperparameter, prediksi val+test.
Output: outputs/models/indobert_best/
        outputs/logs/indobert_best_params.json + indobert_tuning_summary.csv + indobert_epoch_log.csv
        outputs/metrics/indobert_{validation,test}_overall.csv + per_class + per_aspect + report
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path
from src.models.indobert_model import IndoBERTABSA
from src.evaluation.metrics import full_report

Path('outputs/predictions').mkdir(parents=True, exist_ok=True)

print('=== 04. TRAIN INDOBERT ===')
df_train = pd.read_csv('data/processed/train.csv')
df_val   = pd.read_csv('data/processed/validation.csv')
df_test  = pd.read_csv('data/processed/test.csv')

model = IndoBERTABSA(
    checkpoint  = 'indobenchmark/indobert-base-p2',
    config_path = 'configs/indobert.yaml',
    seed        = 42,
)

print(f'\nFine-tuning pada {len(df_train)} train samples...')
model.tune(df_train, df_val, log_dir='outputs/logs')

best_maxlen = model.best_params['max_length']
for split, df in [('validation', df_val), ('test', df_test)]:
    df = df.copy()
    df['y_pred_indobert'] = model.predict(
        df, model_path='outputs/models/indobert_best', max_length=best_maxlen)

    # simpan prediksi
    out = f'outputs/predictions/pred_indobert_{split}.csv'
    df[['sample_id', 'review_id', 'aspect', 'sentiment', 'y_pred_indobert']].to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # simpan semua metrik
    full_report(df, model_name='indobert', pred_col='y_pred_indobert', split=split)

print('\n[OK] IndoBERT selesai.')
