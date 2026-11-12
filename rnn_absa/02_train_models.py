"""
02_train_models.py
Train all 6 RNN architectures with fixed hyperparameters across multiple seeds.

For each model × experiment × seed:
  - Train with early stopping
  - Save model checkpoint + per-seed metrics

Final summary: mean ± std of Macro-F1 across seeds saved to outputs/logs/

Usage:
  python rnn_absa/02_train_models.py [--experiment aspect|overall|both]
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.text_preprocessing import (
    preprocess_aspect_input, preprocess_overall_input,
)
from models.embedding import FastTextEmbedding, build_vocab
from models.rnn_model import RNNClassifier

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(ROOT, 'configs', 'experiment.yaml')
DATA_DIR = os.path.join(ROOT, '..', 'data', 'processed')
LOG_DIR  = os.path.join(ROOT, 'outputs', 'logs')
MDL_DIR  = os.path.join(ROOT, 'outputs', 'models')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

SEEDS     = cfg['experiment']['seeds']
LABELS    = cfg['data']['labels']
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}
MODELS    = cfg['models']
OVERALL   = cfg['data']['overall_aspect']
HP        = cfg['hyperparameters']
EMB_PATH  = os.path.join(ROOT, cfg['embedding']['path'])
EMB_DIM   = cfg['embedding']['dim']
EMB_FREEZE= cfg['embedding']['freeze']
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── dataset ───────────────────────────────────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, tokens_list, labels, vocab, max_len, pad_idx=0):
        self.samples = tokens_list
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.samples[idx][:self.max_len]
        ids    = [self.vocab.get(t, 1) for t in tokens]
        ids   += [self.pad_idx] * (self.max_len - len(ids))
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


def load_data(experiment: str) -> dict:
    dfs = {}
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(os.path.join(DATA_DIR, f'{split}.csv'))
        df = df[df['aspect'] != OVERALL] if experiment == 'aspect' \
             else df[df['aspect'] == OVERALL]
        dfs[split] = df[df['sentiment'].isin(LABELS)].copy()
    return dfs


def tokenize_df(df: pd.DataFrame, experiment: str) -> list:
    if experiment == 'aspect':
        return [preprocess_aspect_input(r['aspect'], r['review_text'])
                for _, r in df.iterrows()]
    return [preprocess_overall_input(r['review_text']) for _, r in df.iterrows()]


# ── training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.extend(model(x.to(DEVICE)).argmax(dim=-1).cpu().numpy())
        trues.extend(y.numpy())
    f1  = float(f1_score(trues, preds, average='macro', zero_division=0))
    acc = float(accuracy_score(trues, preds))
    return f1, acc, preds


def train_one_seed(arch: str, seed: int,
                   train_loader, train_eval_loader, val_loader,
                   vocab_size: int, embed_weight: torch.Tensor,
                   class_weights: torch.Tensor = None,
                   freeze_embedding: bool = True):
    set_seed(seed)
    model = RNNClassifier(
        arch=arch,
        vocab_size=vocab_size,
        embed_dim=EMB_DIM,
        hidden_size=HP['hidden_size'],
        num_layers=HP['num_layers'],
        num_classes=len(LABELS),
        dropout=HP['dropout'],
        embedding_weight=embed_weight,
        freeze_embedding=freeze_embedding,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HP['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    patience  = HP['early_stopping_patience']

    best_f1, best_epoch, no_improve = 0.0, 0, 0
    best_state = None
    epoch_logs = []
    t_train_start = time.time()

    for epoch in range(1, HP['max_epochs'] + 1):
        t_epoch = time.time()
        loss                    = train_epoch(model, train_loader, optimizer, criterion)
        train_f1, train_acc, _  = evaluate(model, train_eval_loader)
        val_f1, val_acc, _      = evaluate(model, val_loader)
        scheduler.step(val_f1)
        current_lr  = optimizer.param_groups[0]['lr']
        epoch_time  = round(time.time() - t_epoch, 2)
        epoch_logs.append({
            'epoch':          epoch,
            'train_loss':     round(loss,       4),
            'train_macro_f1': round(train_f1,   4),
            'train_acc':      round(train_acc,  4),
            'val_macro_f1':   round(val_f1,     4),
            'val_acc':        round(val_acc,    4),
            'lr':             current_lr,
            'epoch_time_s':   epoch_time,
        })
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    elapsed = round(time.time() - t_train_start, 1)
    return best_f1, best_epoch, epoch_logs, best_state, elapsed


# ── main experiment ───────────────────────────────────────────────────────────
def run_experiment(experiment: str, freeze_embedding: bool = True):
    t_exp_start = time.time()
    print(f"\n{'='*60}")
    print(f"Experiment       : {experiment.upper()}")
    print(f"Seeds            : {SEEDS}")
    print(f"Models           : {MODELS}")
    print(f"Device           : {DEVICE}")
    print(f"Freeze embedding : {freeze_embedding}")
    print(f"{'='*60}")

    dfs    = load_data(experiment)
    tokens = {s: tokenize_df(dfs[s], experiment) for s in ['train', 'validation', 'test']}
    labels = {s: [LABEL2IDX[l] for l in dfs[s]['sentiment']]
              for s in ['train', 'validation', 'test']}

    print("Building vocabulary ...")
    vocab = build_vocab(tokens['train'] + tokens['validation'],
                        min_freq=1, special_tokens=['<PAD>', '<UNK>'])
    print(f"  vocab size: {len(vocab):,}")

    print("Loading fastText embedding ...")
    ft           = FastTextEmbedding(EMB_PATH)
    embed_weight = torch.tensor(ft.build_matrix(vocab), dtype=torch.float32)

    max_len  = HP['max_seq_len']
    datasets = {s: ABSADataset(tokens[s], labels[s], vocab, max_len)
                for s in ['train', 'validation', 'test']}

    emb_mode    = 'frozen' if freeze_embedding else 'unfrozen'
    exp_log_dir = os.path.join(LOG_DIR, emb_mode, experiment)
    exp_mdl_dir = os.path.join(MDL_DIR, emb_mode, experiment)
    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(exp_mdl_dir, exist_ok=True)

    # class weights (balanced) to handle class imbalance
    y_train  = np.array(labels['train'])
    classes  = np.arange(len(LABELS))
    weights  = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: { {l: round(float(w), 3) for l, w in zip(LABELS, weights)} }")

    # shared eval loaders (no shuffle, larger batch for speed)
    train_eval_loader = DataLoader(
        datasets['train'], batch_size=128, shuffle=False, num_workers=2
    )
    val_loader = DataLoader(
        datasets['validation'], batch_size=128, shuffle=False, num_workers=2
    )

    # accumulate results: {arch: [f1_seed1, f1_seed2, ...]}
    all_results = {arch: [] for arch in MODELS}
    all_epoch_logs = {arch: {} for arch in MODELS}

    for arch in MODELS:
        print(f"\n--- {arch.upper()} ---")
        seed_records = []

        for seed in SEEDS:
            train_loader = DataLoader(
                datasets['train'], batch_size=HP['batch_size'],
                shuffle=True, num_workers=2,
                worker_init_fn=lambda wid: np.random.seed(seed + wid)
            )

            f1, epoch, epoch_logs, state, elapsed = train_one_seed(
                arch, seed, train_loader, train_eval_loader, val_loader,
                len(vocab), embed_weight, class_weights, freeze_embedding
            )
            print(f"  seed={seed} → val Macro-F1={f1:.4f}  (stopped ep{epoch})  [{elapsed}s]")

            # save checkpoint
            ckpt_path = os.path.join(exp_mdl_dir, f'{arch}_seed{seed}.pt')
            torch.save(state, ckpt_path)

            seed_records.append({
                'arch': arch, 'seed': seed,
                'val_macro_f1': round(f1, 4), 'best_epoch': epoch,
                'training_time_s': elapsed,
            })
            all_results[arch].append(f1)
            all_epoch_logs[arch][seed] = epoch_logs

        # per-arch seed summary
        f1s  = all_results[arch]
        mean = float(np.mean(f1s))
        std  = float(np.std(f1s))
        print(f"  → mean val Macro-F1 = {mean:.4f} ± {std:.4f}")

        pd.DataFrame(seed_records).to_csv(
            os.path.join(exp_log_dir, f'{arch}_seed_results.csv'), index=False)

        # epoch log per seed
        for seed, logs in all_epoch_logs[arch].items():
            pd.DataFrame(logs).assign(arch=arch, seed=seed).to_csv(
                os.path.join(exp_log_dir, f'{arch}_seed{seed}_epoch_log.csv'), index=False)

    # overall summary table
    summary_rows = []
    for arch in MODELS:
        f1s = all_results[arch]
        times = [r['training_time_s'] for r in seed_records]
        summary_rows.append({
            'arch':               arch,
            'mean_val_macro_f1':  round(float(np.mean(f1s)),   4),
            'std_val_macro_f1':   round(float(np.std(f1s)),    4),
            'mean_time_s':        round(float(np.mean(times)),  1),
            'total_time_s':       round(float(np.sum(times)),   1),
            'freeze_embedding':   freeze_embedding,
            'seeds':              str(SEEDS),
            'timestamp':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(exp_log_dir, 'training_summary.csv'), index=False)

    print(f"\n{'─'*50}")
    print(f"Training Summary — {experiment.upper()}")
    print(f"{'─'*50}")
    print(summary_df[['arch', 'mean_val_macro_f1', 'std_val_macro_f1']].to_string(index=False))
    total_exp_time = round(time.time() - t_exp_start, 1)
    print(f"\nModels saved to : outputs/models/{emb_mode}/{experiment}/")
    print(f"Logs saved to   : outputs/logs/{emb_mode}/{experiment}/")
    print(f"Total time      : {total_exp_time}s  ({total_exp_time/60:.1f} min)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='both',
                        choices=['aspect', 'overall', 'both'])
    emb_group = parser.add_mutually_exclusive_group()
    emb_group.add_argument('--freeze', dest='freeze_embedding', action='store_true',
                           help='Freeze fastText embedding (default from config)')
    emb_group.add_argument('--unfreeze', dest='freeze_embedding', action='store_false',
                           help='Unfreeze fastText embedding for fine-tuning')
    parser.set_defaults(freeze_embedding=EMB_FREEZE)
    args = parser.parse_args()

    assert os.path.exists(EMB_PATH), (
        f"\nfastText model not found: {EMB_PATH}\n"
        f"Download:\n"
        f"  cd rnn_absa/embeddings/\n"
        f"  wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz\n"
        f"  gunzip cc.id.300.bin.gz\n"
    )

    if args.experiment in ('aspect', 'both'):
        run_experiment('aspect', freeze_embedding=args.freeze_embedding)
    if args.experiment in ('overall', 'both'):
        run_experiment('overall', freeze_embedding=args.freeze_embedding)
