"""
Tahap 5: Jalankan pendekatan leksikon pada validasi dan test.
Output: outputs/predictions/pred_lexicon_{val,test}.csv
        outputs/metrics/lexicon_{val,test}_overall.csv + per_class + per_aspect + report
        outputs/logs/lexicon_config.json
"""

import sys
sys.path.insert(0, '.')

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.models.lexicon_model import LexiconABSA
from src.evaluation.metrics import full_report

Path('outputs/predictions').mkdir(parents=True, exist_ok=True)
Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
Path('outputs/logs').mkdir(parents=True, exist_ok=True)

print('=== 02. RUN LEXICON ===')
model = LexiconABSA()

# simpan konfigurasi lexicon
lexicon_config = {
    'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'lexicon_base':     'InSet (Indonesian Sentiment Lexicon)',
    'lexicon_domain':   'lexicons/hotel_domain_lexicon.csv',
    'aspect_terms':     'lexicons/aspect_terms.csv',
    'negation_window':  3,
    'no_match_policy':  'neutral',
}
with open('outputs/logs/lexicon_config.json', 'w') as f:
    json.dump(lexicon_config, f, indent=2)

for split in ['validation', 'test']:
    df = pd.read_csv(f'data/processed/{split}.csv')
    print(f'\nPrediksi {split} ({len(df)} pairs)...')
    df['y_pred_lexicon'] = model.predict(df)

    # simpan prediksi
    out = f'outputs/predictions/pred_lexicon_{split}.csv'
    df[['sample_id', 'review_id', 'aspect', 'sentiment', 'y_pred_lexicon']].to_csv(out, index=False)
    print(f'  Saved: {out}')

    # simpan semua metrik (overall + per_class + per_aspect + report)
    full_report(df, model_name='lexicon', pred_col='y_pred_lexicon', split=split)

print('\n[OK] Lexicon selesai.')
