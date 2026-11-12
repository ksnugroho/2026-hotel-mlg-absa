import os
import json
from datetime import datetime

import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)

LABEL_ORDER = ['negative', 'neutral', 'positive']


def full_report(
    df: pd.DataFrame,
    model_name: str,
    pred_col: str = 'predicted_sentiment',
    true_col: str = 'sentiment',
    out_dir: str = 'outputs/metrics',
    split: str = 'test',
    experiment: str = 'aspect',   # 'aspect' or 'overall'
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    prefix    = f"{out_dir}/{model_name}_{experiment}_{split}"
    y_true    = df[true_col]
    y_pred    = df[pred_col]
    labels    = [l for l in LABEL_ORDER if l in y_true.unique()]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    acc    = float(accuracy_score(y_true, y_pred))
    mac_p  = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    mac_r  = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    mac_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    wt_f1  = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    overall = {
        'model': model_name, 'experiment': experiment, 'split': split,
        'timestamp': timestamp, 'n_samples': len(df),
        'accuracy': acc, 'macro_precision': mac_p,
        'macro_recall': mac_r, 'macro_f1': mac_f1, 'weighted_f1': wt_f1,
    }

    pd.DataFrame([overall]).to_csv(f"{prefix}_overall.csv", index=False)
    with open(f"{prefix}_overall.json", 'w') as f:
        json.dump(overall, f, indent=2)

    # per-class
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    rows = []
    for lbl in labels:
        r = report.get(lbl, {})
        rows.append({'label': lbl, 'precision': r.get('precision', 0),
                     'recall': r.get('recall', 0), 'f1': r.get('f1-score', 0),
                     'support': r.get('support', 0)})
    pd.DataFrame(rows).to_csv(f"{prefix}_per_class.csv", index=False)

    # text report
    with open(f"{prefix}_classification_report.txt", 'w') as f:
        f.write(f"Model     : {model_name}\n")
        f.write(f"Experiment: {experiment}\n")
        f.write(f"Split     : {split}\n")
        f.write(f"Timestamp : {timestamp}\n\n")
        f.write(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # per-aspect (only for aspect experiment)
    if experiment == 'aspect' and 'aspect' in df.columns:
        asp_rows = []
        for asp, grp in df.groupby('aspect'):
            yt, yp = grp[true_col], grp[pred_col]
            asp_rows.append({
                'aspect': asp, 'n_samples': len(grp),
                'accuracy': float(accuracy_score(yt, yp)),
                'macro_precision': float(precision_score(yt, yp, average='macro', zero_division=0)),
                'macro_recall': float(recall_score(yt, yp, average='macro', zero_division=0)),
                'macro_f1': float(f1_score(yt, yp, average='macro', zero_division=0)),
                'weighted_f1': float(f1_score(yt, yp, average='weighted', zero_division=0)),
            })
        pd.DataFrame(asp_rows).to_csv(f"{prefix}_per_aspect.csv", index=False)

    return overall
