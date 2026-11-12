"""
Tahap 9-11: Evaluasi ketiga model pada test set yang sama.
- Metrik keseluruhan + per aspek
- Confusion matrix
- McNemar's test + koreksi Holm
Output: outputs/metrics/ + outputs/figures/
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path
from src.evaluation.metrics           import full_report
from src.evaluation.confusion_matrix_plot import plot_all_models
from src.evaluation.significance_test import run_significance_tests

Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
Path('outputs/figures').mkdir(parents=True, exist_ok=True)

print('=== 05. EVALUATE TEST SET ===')

# Load test + prediksi ketiga model
df_test = pd.read_csv('data/processed/test.csv')
pred_lex  = pd.read_csv('outputs/predictions/pred_lexicon_test.csv')
pred_svm  = pd.read_csv('outputs/predictions/pred_svm_test.csv')
pred_bert = pd.read_csv('outputs/predictions/pred_indobert_test.csv')

df_test = df_test.merge(pred_lex[ ['sample_id','y_pred_lexicon']],  on='sample_id')
df_test = df_test.merge(pred_svm[ ['sample_id','y_pred_svm']],      on='sample_id')
df_test = df_test.merge(pred_bert[['sample_id','y_pred_indobert']], on='sample_id')

# Simpan prediksi gabungan (Section 8.1)
combined_cols = ['sample_id','review_id','aspect','sentiment',
                 'y_pred_lexicon','y_pred_svm','y_pred_indobert']
df_test[combined_cols].to_csv('outputs/predictions/predictions_combined_test.csv', index=False)

# Metrik per model
print('\n--- Metrik Keseluruhan ---')
results = {}
for model_name, pred_col in [
    ('lexicon',  'y_pred_lexicon'),
    ('svm',      'y_pred_svm'),
    ('indobert', 'y_pred_indobert'),
]:
    results[model_name] = full_report(df_test, model_name, pred_col)

# Tabel perbandingan
df_compare = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'model'})
df_compare.to_csv('outputs/metrics/comparison_overall.csv', index=False)
print('\nPerbandingan keseluruhan:')
print(df_compare.to_string(index=False))

# Confusion matrix
print('\n--- Confusion Matrix ---')
plot_all_models(df_test, {
    'lexicon':  df_test['y_pred_lexicon'].values,
    'svm':      df_test['y_pred_svm'].values,
    'indobert': df_test['y_pred_indobert'].values,
})

# McNemar + Holm
print('\n--- Significance Test ---')
run_significance_tests(
    y_true          = df_test['sentiment'].values,
    y_pred_lexicon  = df_test['y_pred_lexicon'].values,
    y_pred_svm      = df_test['y_pred_svm'].values,
    y_pred_indobert = df_test['y_pred_indobert'].values,
    alpha   = 0.05,
    out_dir = 'outputs/metrics',
)

print('\n[OK] Evaluasi test selesai.')
