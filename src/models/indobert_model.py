"""
IndoBERT-base-p2 fine-tuning untuk ABSA.
Checkpoint: indobenchmark/indobert-base-p2
Input format: [ASPEK] {aspect} [ULASAN] {review_text}
"""

import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score
from src.preprocessing.text_preprocessing import preprocess_for_indobert


LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_config(path: str = 'configs/indobert.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ABSADataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = [
            preprocess_for_indobert(t) for t in df['input_text'].tolist()
        ]
        self.labels = [LABEL2ID[s] for s in df['sentiment'].tolist()]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length  = self.max_length,
            padding     = 'max_length',
            truncation  = True,
            return_tensors = 'pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


class IndoBERTABSA:
    def __init__(
        self,
        checkpoint:  str = 'indobenchmark/indobert-base-p2',
        config_path: str = 'configs/indobert.yaml',
        seed:        int = 42,
    ):
        self.config     = load_config(config_path)
        self.checkpoint = checkpoint
        self.seed       = seed
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer  = None
        self.model      = None
        self.best_params = None
        set_seed(seed)

    def _init_model(self, checkpoint: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels  = len(LABEL2ID),
            id2label    = ID2LABEL,
            label2id    = LABEL2ID,
        ).to(self.device)

    def _train_epoch(self, loader, optimizer, scheduler, scaler=None):
        self.model.train()
        total_loss = 0
        for batch in loader:
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels         = batch['labels'].to(self.device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    out  = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate(self, loader) -> tuple[float, float]:
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['labels'].to(self.device)
                out            = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred           = out.logits.argmax(-1).cpu().numpy()
                preds.extend(pred)
                trues.extend(labels.cpu().numpy())
        f1  = f1_score(trues, preds, average='macro', zero_division=0)
        acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
        return f1, acc

    def tune(
        self,
        df_train: pd.DataFrame,
        df_val:   pd.DataFrame,
        log_dir:  str = 'outputs/logs',
    ) -> None:
        """
        Grid search hyperparameter berdasarkan Macro-F1 pada validasi.
        Menyimpan checkpoint terbaik ke outputs/models/indobert_best/.
        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        cfg  = self.config['training']
        hw   = self.config.get('hardware', {})
        use_fp16 = hw.get('fp16', False) and torch.cuda.is_available()
        patience = cfg.get('early_stopping_patience', 2)

        from itertools import product as iproduct
        combos = list(iproduct(
            self.config['training']['learning_rate'],
            self.config['training']['batch_size'],
            self.config['training']['epochs'],
            self.config['training']['warmup_ratio'],
            self.config['tokenizer']['max_length'],
        ))

        best_f1      = -1
        best_params  = None
        all_logs     = []

        print(f'IndoBERT tuning: {len(combos)} kombinasi pada device={self.device}')
        for lr, bs, epochs, warmup, maxlen in combos:
            print(f'\nLR={lr} | BS={bs} | epochs={epochs} | warmup={warmup} | maxlen={maxlen}')
            self._init_model(self.checkpoint)
            train_ds = ABSADataset(df_train, self.tokenizer, maxlen)
            val_ds   = ABSADataset(df_val,   self.tokenizer, maxlen)
            train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2)
            val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2)

            total_steps   = len(train_dl) * epochs
            warmup_steps  = int(total_steps * warmup)
            optimizer     = AdamW(self.model.parameters(), lr=lr,
                                  weight_decay=cfg['weight_decay'])
            scheduler     = get_linear_schedule_with_warmup(
                optimizer, warmup_steps, total_steps)
            scaler        = torch.amp.GradScaler('cuda') if use_fp16 else None

            best_epoch_f1 = -1
            patience_cnt  = 0
            epoch_log     = []
            for ep in range(1, epochs + 1):
                train_loss = self._train_epoch(train_dl, optimizer, scheduler, scaler)
                val_f1, val_acc = self._evaluate(val_dl)
                print(f'  Epoch {ep}: loss={train_loss:.4f} | val_f1={val_f1:.4f} | val_acc={val_acc:.4f}')
                epoch_log.append({'epoch': ep, 'train_loss': float(train_loss),
                                  'val_macro_f1': float(val_f1), 'val_acc': float(val_acc)})
                if val_f1 > best_epoch_f1:
                    best_epoch_f1 = val_f1
                    patience_cnt  = 0
                else:
                    patience_cnt += 1
                if patience_cnt >= patience:
                    print(f'  Early stop pada epoch {ep}')
                    break

            params = {
                'learning_rate': lr, 'batch_size': bs, 'epochs': epochs,
                'warmup_ratio': warmup, 'max_length': maxlen,
                'val_macro_f1': round(best_epoch_f1, 4),
            }
            all_logs.append({**params, 'epoch_log': epoch_log})

            if best_epoch_f1 > best_f1:
                best_f1     = best_epoch_f1
                best_params = params
                save_path   = 'outputs/models/indobert_best'
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f'  Checkpoint baru disimpan (val_f1={best_f1:.4f})')

        self.best_params = best_params
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 1. full nested JSON (epoch-level detail)
        with open(f'{log_dir}/indobert_tuning_log.json', 'w') as f:
            json.dump(all_logs, f, indent=2)

        # 2. best_params.json (mudah dibaca tanpa parse nested)
        best_log = {
            'timestamp':         ts,
            'n_combos':          len(combos),
            'best_val_macro_f1': round(best_f1, 4),
            'best_params':       best_params,
        }
        with open(f'{log_dir}/indobert_best_params.json', 'w') as f:
            json.dump(best_log, f, indent=2)

        # 3. flat CSV per kombinasi (mudah dibuka di Excel)
        summary_rows = []
        for entry in all_logs:
            best_ep_f1 = max(e['val_macro_f1'] for e in entry['epoch_log'])
            best_ep    = max(entry['epoch_log'], key=lambda e: e['val_macro_f1'])
            summary_rows.append({
                'learning_rate':     entry['learning_rate'],
                'batch_size':        entry['batch_size'],
                'epochs':            entry['epochs'],
                'warmup_ratio':      entry['warmup_ratio'],
                'max_length':        entry['max_length'],
                'best_val_macro_f1': round(best_ep_f1, 4),
                'best_epoch':        best_ep['epoch'],
                'best_val_acc':      round(best_ep['val_acc'], 4),
                'final_train_loss':  round(entry['epoch_log'][-1]['train_loss'], 4),
            })
        pd.DataFrame(summary_rows).sort_values(
            'best_val_macro_f1', ascending=False
        ).to_csv(f'{log_dir}/indobert_tuning_summary.csv', index=False)

        # 4. flat CSV per epoch (satu baris per epoch per kombinasi)
        epoch_rows = []
        for combo_id, entry in enumerate(all_logs, 1):
            for ep in entry['epoch_log']:
                epoch_rows.append({
                    'combo_id':     combo_id,
                    'learning_rate': entry['learning_rate'],
                    'batch_size':    entry['batch_size'],
                    'max_length':    entry['max_length'],
                    'epoch':         ep['epoch'],
                    'train_loss':    round(ep['train_loss'], 4),
                    'val_macro_f1':  round(ep['val_macro_f1'], 4),
                    'val_acc':       round(ep['val_acc'], 4),
                })
        pd.DataFrame(epoch_rows).to_csv(
            f'{log_dir}/indobert_epoch_log.csv', index=False)

        print(f'\nTuning selesai. Best val_macro_f1={best_f1:.4f}')
        print(best_params)
        # reload checkpoint terbaik agar self.model siap untuk predict()
        self._init_model('outputs/models/indobert_best')

    def predict(self, df: pd.DataFrame, model_path: str = 'outputs/models/indobert_best',
                max_length: int = 128, batch_size: int = 32) -> list[str]:
        if self.model is None or model_path != getattr(self, '_loaded_path', None):
            self._init_model(model_path)
            self._loaded_path = model_path
        ds = ABSADataset(df, self.tokenizer, max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dl:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out  = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = out.logits.argmax(-1).cpu().numpy()
                preds.extend([ID2LABEL[p] for p in pred])
        return preds
