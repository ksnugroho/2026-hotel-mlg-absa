"""
04_aggregate_results.py
Aggregate all evaluation metrics into two summary files (frozen / unfrozen).

Reads multiseed_summary.csv (eval metrics) and seed_results.csv (training time)
from each experiment (aspect, overall) and combines them into one wide CSV per
embedding mode.

Metrics per split : accuracy, macro_precision, macro_recall, macro_f1, weighted_f1
Columns per metric: mean, std, best (across seeds)
Training time cols: mean_training_time_s, std_training_time_s, total_training_time_s

Outputs:
  outputs/metrics/frozen/aggregate_results.csv
  outputs/metrics/unfrozen/aggregate_results.csv

Usage:
  python rnn_absa/04_aggregate_results.py
"""

import os

import numpy as np
import pandas as pd

ROOT    = os.path.dirname(os.path.abspath(__file__))
MET_DIR = os.path.join(ROOT, 'outputs', 'metrics')
LOG_DIR = os.path.join(ROOT, 'outputs', 'logs')

EXPS    = ['aspect', 'overall']
SPLITS  = ['train', 'validation', 'test']
METRICS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']
MODES   = ['frozen', 'unfrozen']

_BASE_COLS  = ['emb_mode', 'experiment', 'arch', 'best_seed']
_TIME_COLS  = ['mean_training_time_s', 'std_training_time_s', 'total_training_time_s']
_METRIC_COLS = []
for _split in SPLITS:
    for _metric in METRICS:
        _METRIC_COLS += [
            f'{_split}_mean_{_metric}',
            f'{_split}_std_{_metric}',
            f'{_split}_best_{_metric}',
        ]

OUTPUT_COLS = _BASE_COLS + _TIME_COLS + _METRIC_COLS


def load_eval_summary(emb_mode: str, experiment: str) -> pd.DataFrame | None:
    path = os.path.join(MET_DIR, emb_mode, experiment, 'multiseed_summary.csv')
    if not os.path.exists(path):
        print(f'  [SKIP] tidak ditemukan: {path}')
        return None
    df = pd.read_csv(path)
    df.insert(0, 'experiment', experiment)
    df.insert(0, 'emb_mode', emb_mode)
    return df


def load_training_times(emb_mode: str, experiment: str) -> pd.DataFrame:
    """Baca seed_results.csv lalu hitung mean/std/total training time per arch."""
    rows = []
    log_exp_dir = os.path.join(LOG_DIR, emb_mode, experiment)
    for arch in ['rnn', 'lstm', 'gru', 'birnn', 'bilstm', 'bigru']:
        path = os.path.join(log_exp_dir, f'{arch}_seed_results.csv')
        if not os.path.exists(path):
            continue
        sr = pd.read_csv(path)
        times = sr['training_time_s'].values
        rows.append({
            'arch':                   arch,
            'mean_training_time_s':   round(float(np.mean(times)), 1),
            'std_training_time_s':    round(float(np.std(times)),  1),
            'total_training_time_s':  round(float(np.sum(times)),  1),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['arch', 'mean_training_time_s', 'std_training_time_s', 'total_training_time_s']
    )


def fill_best_cols(df: pd.DataFrame) -> pd.DataFrame:
    """multiseed_summary menyimpan best_macro_f1; kolom best_ lain di-fallback ke mean."""
    for split in SPLITS:
        for metric in METRICS:
            best_col = f'{split}_best_{metric}'
            if best_col not in df.columns:
                mean_col = f'{split}_mean_{metric}'
                df[best_col] = df[mean_col] if mean_col in df.columns else float('nan')
    return df


def aggregate(emb_mode: str):
    frames = []
    for exp in EXPS:
        df = load_eval_summary(emb_mode, exp)
        if df is None:
            continue

        time_df = load_training_times(emb_mode, exp)
        if not time_df.empty:
            df = df.merge(time_df, on='arch', how='left')
        else:
            for col in _TIME_COLS:
                df[col] = float('nan')

        frames.append(df)

    if not frames:
        print(f'  [SKIP] tidak ada data untuk {emb_mode}')
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = fill_best_cols(combined)

    for col in OUTPUT_COLS:
        if col not in combined.columns:
            combined[col] = float('nan')

    combined = combined[OUTPUT_COLS]

    float_cols = [c for c in combined.columns if c not in _BASE_COLS]
    combined[float_cols] = combined[float_cols].round(4)

    out_dir  = os.path.join(MET_DIR, emb_mode)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'aggregate_results.csv')
    combined.to_csv(out_path, index=False)
    print(f'  Tersimpan → {out_path}  ({len(combined)} baris, {len(combined.columns)} kolom)')

    print(f'\n  {"exp":<10} {"arch":<8} {"train_f1":>9} {"val_f1":>8} {"test_f1":>8} {"±std":>7} {"time_mean":>10} {"±std":>7}')
    print(f'  {"─"*70}')
    for _, row in combined.iterrows():
        t = row.get('mean_training_time_s', float('nan'))
        ts = row.get('std_training_time_s', float('nan'))
        print(
            f'  {row["experiment"]:<10} {row["arch"]:<8}'
            f'  {row["train_mean_macro_f1"]:>8.4f}'
            f'  {row["validation_mean_macro_f1"]:>7.4f}'
            f'  {row["test_mean_macro_f1"]:>7.4f}'
            f'  ±{row["test_std_macro_f1"]:>5.4f}'
            f'  {t:>9.1f}s'
            f'  ±{ts:>5.1f}s'
        )


def generate_paper_tables():
    """
    Buat dua tabel siap paper (aspect & overall), masing-masing berisi:
      architecture | embedding | test_mean_accuracy | test_std_accuracy
                   |           | test_mean_macro_f1 | test_std_macro_f1

    Output:
      outputs/metrics/paper_table_aspect.csv
      outputs/metrics/paper_table_overall.csv
    """
    all_frames = []
    for mode in MODES:
        for exp in EXPS:
            path = os.path.join(MET_DIR, mode, exp, 'multiseed_summary.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['emb_mode']   = mode
            df['experiment'] = exp
            all_frames.append(df)

    if not all_frames:
        print('\n[paper tables] tidak ada data.')
        return

    full = pd.concat(all_frames, ignore_index=True)

    ARCH_ORDER = ['rnn', 'lstm', 'gru', 'birnn', 'bilstm', 'bigru']
    ARCH_LABEL = {'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU',
                  'birnn': 'BiRNN', 'bilstm': 'BiLSTM', 'bigru': 'BiGRU'}
    EMB_LABEL  = {'frozen': 'Frozen', 'unfrozen': 'Unfrozen'}

    out_dir = MET_DIR
    os.makedirs(out_dir, exist_ok=True)

    for exp in EXPS:
        rows = []
        subset = full[full['experiment'] == exp].copy()
        for arch in ARCH_ORDER:
            for mode in MODES:
                r = subset[(subset['arch'] == arch) & (subset['emb_mode'] == mode)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append({
                    'architecture':       ARCH_LABEL[arch],
                    'embedding':          EMB_LABEL[mode],
                    'test_mean_accuracy': round(r['test_mean_accuracy'], 4),
                    'test_std_accuracy':  round(r['test_std_accuracy'],  4),
                    'test_mean_macro_f1': round(r['test_mean_macro_f1'], 4),
                    'test_std_macro_f1':  round(r['test_std_macro_f1'],  4),
                })
        tbl = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f'paper_table_{exp}.csv')
        tbl.to_csv(out_path, index=False)

        print(f'\n  Paper Table — {exp.upper()}')
        print(f'  {"arch":<8} {"emb":<10} {"acc mean":>9} {"±std":>7} {"f1 mean":>8} {"±std":>7}')
        print(f'  {"─"*55}')
        for _, row in tbl.iterrows():
            print(
                f'  {row["architecture"]:<8} {row["embedding"]:<10}'
                f'  {row["test_mean_accuracy"]:>8.4f}  ±{row["test_std_accuracy"]:>5.4f}'
                f'  {row["test_mean_macro_f1"]:>7.4f}  ±{row["test_std_macro_f1"]:>5.4f}'
            )
        print(f'  → {out_path}')


if __name__ == '__main__':
    print('=' * 70)
    print('Aggregate Results')
    print('=' * 70)
    for mode in MODES:
        print(f'\n[{mode.upper()}]')
        aggregate(mode)

    print('\n' + '=' * 70)
    print('Paper Tables')
    print('=' * 70)
    generate_paper_tables()

    print('\nSelesai.')
