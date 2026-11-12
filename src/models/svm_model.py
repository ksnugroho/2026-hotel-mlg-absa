"""
TF-IDF + SVM pipeline untuk ABSA.
Fit vectorizer hanya dari data latih, transform untuk val/test.
"""

import json
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from src.preprocessing.text_preprocessing import preprocess_for_svm


def load_config(path: str = 'configs/svm.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_input_text(df: pd.DataFrame) -> list[str]:
    """Preprocess review_text saja, pertahankan marker [ASPEK]/[ULASAN]."""
    return [
        f"[ASPEK] {row['aspect']} [ULASAN] {preprocess_for_svm(row['review_text'])}"
        for _, row in df.iterrows()
    ]


def _grid_configs(cfg: dict) -> list[dict]:
    """Buat semua kombinasi hyperparameter dari config."""
    vec_cfg  = cfg['vectorizer']
    cls_cfg  = cfg['classifier']
    keys_v   = ['ngram_range', 'min_df', 'max_df', 'sublinear_tf']
    keys_c   = ['kernel', 'C', 'class_weight']
    vals_v   = [vec_cfg[k] for k in keys_v]
    vals_c   = [cls_cfg[k] for k in keys_c]
    combos   = list(product(*vals_v, *vals_c))
    configs  = []
    for combo in combos:
        configs.append({
            'ngram_range':  combo[0],
            'min_df':       combo[1],
            'max_df':       combo[2],
            'sublinear_tf': combo[3],
            'kernel':       combo[4],
            'C':            combo[5],
            'class_weight': combo[6],
        })
    return configs


class SVMModel:
    def __init__(self, config_path: str = 'configs/svm.yaml'):
        self.config   = load_config(config_path)
        self.pipeline = None
        self.best_params = None

    def _make_pipeline(self, params: dict) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range  = tuple(params['ngram_range']),
                min_df       = params['min_df'],
                max_df       = params['max_df'],
                sublinear_tf = params['sublinear_tf'],
                max_features = self.config['vectorizer'].get('max_features'),
            )),
            ('svm', SVC(
                kernel       = params['kernel'],
                C            = params['C'],
                class_weight = params['class_weight'] or None,
                random_state = 42,
            )),
        ])

    def tune(
        self,
        df_train: pd.DataFrame,
        df_val:   pd.DataFrame,
    ) -> pd.DataFrame:
        """Grid search pada data validasi. Metrik utama: macro_f1."""
        X_train = build_input_text(df_train)
        y_train = df_train['sentiment'].values
        X_val   = build_input_text(df_val)
        y_val   = df_val['sentiment'].values

        configs  = _grid_configs(self.config)
        results  = []
        best_f1  = -1
        best_cfg = None
        best_pl  = None

        print(f'Grid search: {len(configs)} kombinasi...')
        for i, params in enumerate(configs, 1):
            pl = self._make_pipeline(params)
            pl.fit(X_train, y_train)
            y_pred = pl.predict(X_val)
            f1_mac = f1_score(y_val, y_pred, average='macro', zero_division=0)
            results.append({**params, 'val_macro_f1': round(f1_mac, 4)})
            if f1_mac > best_f1:
                best_f1  = f1_mac
                best_cfg = params
                best_pl  = pl
            if i % 10 == 0:
                print(f'  [{i}/{len(configs)}] best macro_f1={best_f1:.4f}')

        self.pipeline    = best_pl
        self.best_params = best_cfg

        df_results = pd.DataFrame(results).sort_values('val_macro_f1', ascending=False)

        # simpan best params + summary ke logs
        Path('outputs/logs').mkdir(parents=True, exist_ok=True)
        log = {
            'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_combos':      len(configs),
            'best_val_macro_f1': round(best_f1, 4),
            'best_params':   best_cfg,
        }
        with open('outputs/logs/svm_best_params.json', 'w') as f:
            json.dump(log, f, indent=2)

        print(f'\nKonfigurasi terbaik (val macro_f1={best_f1:.4f}):')
        print(best_cfg)
        return df_results

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = build_input_text(df)
        return self.pipeline.predict(X)

    def save(self, path: str = 'outputs/models/svm_pipeline.pkl') -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'pipeline': self.pipeline, 'params': self.best_params}, f)
        print(f'SVM pipeline disimpan ke {path}')

    @classmethod
    def load(cls, path: str = 'outputs/models/svm_pipeline.pkl') -> 'SVMModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls.__new__(cls)
        model.pipeline    = data['pipeline']
        model.best_params = data['params']
        return model
