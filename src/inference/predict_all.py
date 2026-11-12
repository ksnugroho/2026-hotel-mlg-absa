"""
Modul inferensi: jalankan model terpilih pada DataFrame baru.

Fungsi utama: predict_all(df, model_name, model_path=None, config_path=None)
  - df          : DataFrame dengan kolom 'input_text', 'review_id', 'aspect'
  - model_name  : 'lexicon' | 'svm' | 'indobert'
  - model_path  : path ke artefak model (pickle atau direktori HF)
  - config_path : path ke config YAML (opsional)
  Mengembalikan df dengan kolom tambahan 'predicted_sentiment'.
"""

import pandas as pd


def predict_all(
    df: pd.DataFrame,
    model_name: str,
    model_path: str | None = None,
    config_path: str | None = None,
    max_length: int = 128,
) -> pd.DataFrame:
    df = df.copy()

    if model_name == 'lexicon':
        from src.models.lexicon_model import LexiconABSA
        model = LexiconABSA()
        df['predicted_sentiment'] = model.predict(df)

    elif model_name == 'svm':
        from src.models.svm_model import SVMModel
        path = model_path or 'outputs/models/svm_pipeline.pkl'
        model = SVMModel.load(path)
        df['predicted_sentiment'] = model.predict(df)

    elif model_name == 'indobert':
        from src.models.indobert_model import IndoBERTABSA
        checkpoint = model_path or 'outputs/models/indobert_best'
        model = IndoBERTABSA(checkpoint=checkpoint)
        df['predicted_sentiment'] = model.predict(
            df, model_path=checkpoint, max_length=max_length)

    else:
        raise ValueError(f"model_name tidak dikenal: '{model_name}'. "
                         "Pilih: 'lexicon' | 'svm' | 'indobert'")

    return df
