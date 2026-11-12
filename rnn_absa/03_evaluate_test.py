"""
03_evaluate_test.py
Evaluate all trained models (all seeds) on train, validation, and test sets.

For each model:
  - Load each seed checkpoint once, evaluate on all three splits
  - Report mean ± std for all metrics (accuracy, macro-P/R/F1, weighted-F1) across seeds
  - Save per-seed predictions and full classification reports (val + test)
  - Save arch-level (best-seed by val F1) metrics for 05_generate_figures.py
  - Save combined test predictions for downstream analysis

Outputs per experiment:
  outputs/metrics/{experiment}/
    {arch}_{experiment}_{split}_overall.csv      ← arch-level (best seed)
    {arch}_{experiment}_{split}_per_class.csv
    {arch}_{experiment}_{split}_per_aspect.csv   (aspect exp only)
    {arch}_seed{seed}_{experiment}_{split}_overall.csv  ← per-seed
    multiseed_summary.csv                        ← mean±std all metrics all splits

Usage:
  python rnn_absa/03_evaluate_test.py [--experiment aspect|overall|both]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.text_preprocessing import (
    preprocess_aspect_input, preprocess_overall_input,
)
from models.embedding import FastTextEmbedding, build_vocab
from models.rnn_model import RNNClassifier
from evaluation.metrics import full_report

ROOT     = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(ROOT, 'configs', 'experiment.yaml')
DATA_DIR = os.path.join(ROOT, '..', 'data', 'processed')
OUT_ROOT = os.path.join(ROOT, 'outputs')

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

SEEDS      = cfg['experiment']['seeds']
LABELS     = cfg['data']['labels']
LABEL2IDX  = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL  = {i: l for l, i in LABEL2IDX.items()}
MODELS     = cfg['models']
OVERALL    = cfg['data']['overall_aspect']
HP         = cfg['hyperparameters']
EMB_PATH   = os.path.join(ROOT, cfg['embedding']['path'])
EMB_DIM    = cfg['embedding']['dim']
EMB_FREEZE = cfg['embedding']['freeze']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPLITS       = ['train', 'validation', 'test']
METRIC_NAMES = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(experiment):
    dfs = {}
    for split in SPLITS:
        df = pd.read_csv(os.path.join(DATA_DIR, f'{split}.csv'))
        df = df[df['aspect'] != OVERALL] if experiment == 'aspect' \
             else df[df['aspect'] == OVERALL]
        dfs[split] = df[df['sentiment'].isin(LABELS)].copy()
    return dfs


def tokenize_df(df, experiment):
    if experiment == 'aspect':
        return [preprocess_aspect_input(r['aspect'], r['review_text'])
                for _, r in df.iterrows()]
    return [preprocess_overall_input(r['review_text']) for _, r in df.iterrows()]


def compute_metrics(y_true, y_pred):
    return {
        'accuracy':        float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall':    float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1':        float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1':     float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


@torch.no_grad()
def predict(model, tokens_list, vocab, max_len, batch_size=128):
    model.eval()
    preds = []
    for i in range(0, len(tokens_list), batch_size):
        batch = tokens_list[i:i + batch_size]
        ids   = []
        for tokens in batch:
            t  = tokens[:max_len]
            t  = [vocab.get(tok, 1) for tok in t]
            t += [0] * (max_len - len(t))
            ids.append(t)
        preds.extend(
            model(torch.tensor(ids, dtype=torch.long).to(DEVICE))
            .argmax(dim=-1).cpu().numpy().tolist()
        )
    return preds


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_experiment(experiment, freeze_embedding=True):
    emb_mode = 'frozen' if freeze_embedding else 'unfrozen'
    MDL_DIR  = os.path.join(OUT_ROOT, 'models',      emb_mode)
    MET_DIR  = os.path.join(OUT_ROOT, 'metrics',     emb_mode)
    PRED_DIR = os.path.join(OUT_ROOT, 'predictions', emb_mode)
    os.makedirs(os.path.join(MET_DIR,  experiment), exist_ok=True)
    os.makedirs(PRED_DIR,                           exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Evaluating       : {experiment.upper()}")
    print(f"Embedding mode   : {emb_mode}")
    print(f"{'='*65}")

    dfs    = load_data(experiment)
    tokens = {s: tokenize_df(dfs[s], experiment) for s in SPLITS}

    vocab = build_vocab(tokens['train'] + tokens['validation'],
                        min_freq=1, special_tokens=['<PAD>', '<UNK>'])
    print(f"Vocab size : {len(vocab):,}")

    ft           = FastTextEmbedding(EMB_PATH)
    embed_weight = torch.tensor(ft.build_matrix(vocab), dtype=torch.float32)
    max_len      = HP['max_seq_len']

    exp_met_dir = os.path.join(MET_DIR, experiment)
    os.makedirs(exp_met_dir, exist_ok=True)

    # arch → split → seed → metrics
    all_metrics = {arch: {s: {} for s in SPLITS} for arch in MODELS}
    # arch → split → seed → pred_labels
    all_preds   = {arch: {s: {} for s in SPLITS} for arch in MODELS}

    for arch in MODELS:
        print(f"\n{'─'*45}")
        print(f"  {arch.upper()}")
        print(f"{'─'*45}")

        for seed in SEEDS:
            ckpt_path = os.path.join(MDL_DIR, experiment, f'{arch}_seed{seed}.pt')
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] seed={seed} — checkpoint not found: {ckpt_path}")
                continue

            model = RNNClassifier(
                arch=arch, vocab_size=len(vocab), embed_dim=EMB_DIM,
                hidden_size=HP['hidden_size'], num_layers=HP['num_layers'],
                num_classes=len(LABELS), dropout=HP['dropout'],
                embedding_weight=embed_weight, freeze_embedding=freeze_embedding,
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

            for split in SPLITS:
                df_split    = dfs[split].copy()
                raw_preds   = predict(model, tokens[split], vocab, max_len)
                pred_labels = [IDX2LABEL[p] for p in raw_preds]

                all_preds[arch][split][seed]   = pred_labels
                all_metrics[arch][split][seed] = compute_metrics(
                    df_split['sentiment'], pred_labels
                )

                # save per-seed predictions (all splits)
                df_tmp = df_split.copy()
                df_tmp['predicted_sentiment'] = pred_labels
                df_tmp.to_csv(
                    os.path.join(PRED_DIR,
                                 f'pred_{arch}_{experiment}_seed{seed}_{split}.csv'),
                    index=False
                )

                # full classification report (val + test, not train, to keep output manageable)
                if split in ('validation', 'test'):
                    full_report(df_tmp, model_name=f'{arch}_seed{seed}',
                                pred_col='predicted_sentiment', true_col='sentiment',
                                out_dir=exp_met_dir, split=split, experiment=experiment)

            m = all_metrics[arch]
            print(
                f"  seed={seed}"
                f"  train_f1={m['train'][seed]['macro_f1']:.4f}"
                f"  val_f1={m['validation'][seed]['macro_f1']:.4f}"
                f"  test_f1={m['test'][seed]['macro_f1']:.4f}"
            )

        # ── aggregate: best seed by val Macro-F1 ───────────────────────────────
        val_seeds = sorted(all_metrics[arch]['validation'].keys())
        if not val_seeds:
            print(f"  [SKIP] no completed seeds for {arch}")
            continue

        val_f1s   = [all_metrics[arch]['validation'][s]['macro_f1'] for s in val_seeds]
        best_seed = val_seeds[int(np.argmax(val_f1s))]
        print(f"\n  Best seed = {best_seed}  (val Macro-F1 = {max(val_f1s):.4f})")

        # arch-level (best-seed) full report for val + test → used by 05_generate_figures.py
        for split in ('validation', 'test'):
            if best_seed not in all_preds[arch][split]:
                continue
            df_split = dfs[split].copy()
            df_split['predicted_sentiment'] = all_preds[arch][split][best_seed]
            full_report(df_split, model_name=arch,
                        pred_col='predicted_sentiment', true_col='sentiment',
                        out_dir=exp_met_dir, split=split, experiment=experiment)

        # ── print mean ± std per split ──────────────────────────────────────────
        print(f"\n  {'split':<12} {'metric':<22} {'mean':>8}  {'std':>8}")
        print(f"  {'─'*55}")
        for split in SPLITS:
            seeds_ok = sorted(all_metrics[arch][split].keys())
            if not seeds_ok:
                continue
            for mn in METRIC_NAMES:
                vals = [all_metrics[arch][split][s][mn] for s in seeds_ok]
                print(f"  {split:<12} {mn:<22} {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}")

    # ── combined test predictions (best seed per arch) ─────────────────────────
    df_combined = dfs['test'].copy()
    for arch in MODELS:
        val_seeds = sorted(all_metrics[arch]['validation'].keys())
        if not val_seeds:
            continue
        val_f1s   = [all_metrics[arch]['validation'][s]['macro_f1'] for s in val_seeds]
        best_seed = val_seeds[int(np.argmax(val_f1s))]
        if best_seed in all_preds[arch]['test']:
            df_combined[f'pred_{arch}'] = all_preds[arch]['test'][best_seed]
    df_combined.to_csv(
        os.path.join(PRED_DIR, f'predictions_{experiment}_test.csv'), index=False)

    # ── multiseed_summary.csv ───────────────────────────────────────────────────
    summary_rows = []
    for arch in MODELS:
        val_seeds = sorted(all_metrics[arch]['validation'].keys())
        if not val_seeds:
            continue
        val_f1s   = [all_metrics[arch]['validation'][s]['macro_f1'] for s in val_seeds]
        best_seed = val_seeds[int(np.argmax(val_f1s))]

        row = {'arch': arch, 'best_seed': best_seed}
        for split in SPLITS:
            seeds_ok = sorted(all_metrics[arch][split].keys())
            if not seeds_ok:
                continue
            for mn in METRIC_NAMES:
                vals = [all_metrics[arch][split][s][mn] for s in seeds_ok]
                row[f'{split}_mean_{mn}'] = round(float(np.mean(vals)), 4)
                row[f'{split}_std_{mn}']  = round(float(np.std(vals)),  4)
            row[f'{split}_best_macro_f1'] = round(
                max(all_metrics[arch][split][s]['macro_f1'] for s in seeds_ok), 4)
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(exp_met_dir, 'multiseed_summary.csv'), index=False)
        print(f"\n{'='*65}")
        print(f"Test Set Summary — {experiment.upper()}")
        print(f"{'='*65}")
        disp = ['arch',
                'train_mean_accuracy',   'train_mean_macro_f1',
                'validation_mean_accuracy', 'validation_mean_macro_f1',
                'test_mean_accuracy',    'test_std_accuracy',
                'test_mean_macro_f1',    'test_std_macro_f1',
                'test_mean_weighted_f1', 'best_seed']
        disp = [c for c in disp if c in summary_df.columns]
        print(summary_df[disp].to_string(index=False))

    print(f"\nDone → outputs/metrics/{emb_mode}/{experiment}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='both',
                        choices=['aspect', 'overall', 'both'])
    emb_group = parser.add_mutually_exclusive_group()
    emb_group.add_argument('--freeze', dest='freeze_embedding', action='store_true',
                           help='Use frozen embedding (default from config)')
    emb_group.add_argument('--unfreeze', dest='freeze_embedding', action='store_false',
                           help='Use unfrozen embedding (fine-tuned)')
    parser.set_defaults(freeze_embedding=EMB_FREEZE)
    args = parser.parse_args()

    if args.experiment in ('aspect', 'both'):
        evaluate_experiment('aspect', freeze_embedding=args.freeze_embedding)
    if args.experiment in ('overall', 'both'):
        evaluate_experiment('overall', freeze_embedding=args.freeze_embedding)
