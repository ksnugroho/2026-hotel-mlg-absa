"""
Tahap 13: Inferensi seluruh dataset menggunakan model terbaik.
Jalankan setelah 05_evaluate_test.py dan model terbaik telah dipilih.

Edit MODEL_BEST di bawah sesuai hasil evaluasi.
Output: outputs/predictions/full_inference.csv
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path

# ── Pilih model terbaik setelah 05_evaluate_test.py ──────────────────────────
MODEL_BEST = 'indobert'   # ganti: 'lexicon' | 'svm' | 'indobert'
# ─────────────────────────────────────────────────────────────────────────────

Path('outputs/predictions').mkdir(parents=True, exist_ok=True)

print(f'=== 06. INFER FULL DATASET (model: {MODEL_BEST}) ===')

df_raw = pd.read_csv('data/raw/dataset.csv')
from src.data.load_data import build_review_aspect_pairs
df_all = build_review_aspect_pairs(df_raw)
print(f'Total pasangan untuk inferensi: {len(df_all)}')

if MODEL_BEST == 'lexicon':
    from src.models.lexicon_model import LexiconABSA
    model = LexiconABSA()
    df_all['predicted_sentiment'] = model.predict(df_all)
    df_all['model_version']       = 'lexicon_inset_v1'
    df_all['confidence']          = None

elif MODEL_BEST == 'svm':
    from src.models.svm_model import SVMModel
    model = SVMModel.load('outputs/models/svm_pipeline.pkl')
    df_all['predicted_sentiment'] = model.predict(df_all)
    df_all['model_version']       = 'svm_tfidf_v1'
    df_all['confidence']          = None

elif MODEL_BEST == 'indobert':
    import yaml, torch
    from src.models.indobert_model import IndoBERTABSA
    with open('configs/indobert.yaml') as f:
        cfg = yaml.safe_load(f)
    model = IndoBERTABSA(checkpoint='outputs/models/indobert_best')
    best_maxlen = cfg['tokenizer']['max_length'][0]  # pakai default
    df_all['predicted_sentiment'] = model.predict(
        df_all, model_path='outputs/models/indobert_best', max_length=best_maxlen)
    df_all['model_version'] = 'indobert-base-p2_finetuned'
    df_all['confidence']    = None

out_cols = ['review_id', 'hotel_name', 'district', 'review_text',
            'aspect', 'predicted_sentiment', 'confidence', 'model_version']
df_out = df_all[out_cols]
df_out.to_csv('outputs/predictions/full_inference.csv', index=False)
print(f'\nInferensi selesai: {len(df_out)} baris')
print(df_out['predicted_sentiment'].value_counts())
print('\nSaved: outputs/predictions/full_inference.csv')
