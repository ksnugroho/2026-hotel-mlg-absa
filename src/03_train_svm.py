"""
Tahap 6: Fit TF-IDF, grid search SVM pada validasi, prediksi val+test.
Output: outputs/models/svm_pipeline.pkl
        outputs/logs/svm_best_params.json + svm_grid_search_results.csv
        outputs/metrics/svm_{validation,test}_overall.csv + per_class + per_aspect + report
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path
from src.models.svm_model import SVMModel
from src.evaluation.metrics import full_report

Path('outputs/predictions').mkdir(parents=True, exist_ok=True)
Path('outputs/metrics').mkdir(parents=True, exist_ok=True)

print('=== 03. TRAIN SVM ===')
df_train = pd.read_csv('data/processed/train.csv')
df_val   = pd.read_csv('data/processed/validation.csv')
df_test  = pd.read_csv('data/processed/test.csv')

model = SVMModel('configs/svm.yaml')

print(f'\nTuning pada {len(df_train)} train + {len(df_val)} val samples...')
df_results = model.tune(df_train, df_val)
df_results.to_csv('outputs/metrics/svm_grid_search_results.csv', index=False)
model.save('outputs/models/svm_pipeline.pkl')

for split, df in [('validation', df_val), ('test', df_test)]:
    df = df.copy()
    df['y_pred_svm'] = model.predict(df)

    # simpan prediksi
    out = f'outputs/predictions/pred_svm_{split}.csv'
    df[['sample_id', 'review_id', 'aspect', 'sentiment', 'y_pred_svm']].to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # simpan semua metrik
    full_report(df, model_name='svm', pred_col='y_pred_svm', split=split)

print('\n[OK] SVM selesai.')
