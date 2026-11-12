"""
Plot confusion matrix per model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

LABELS = ['negative', 'neutral', 'positive']


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    out_dir:    str = 'outputs/figures',
    labels:     list = LABELS,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cm       = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm     = np.divide(cm.astype(float), row_sums,
                         where=row_sums != 0,
                         out=np.zeros_like(cm, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title, fmt in zip(
        axes,
        [cm, norm],
        [f'{model_name} — Count', f'{model_name} — Normalized'],
        ['.0f', '.2f'],
    ):
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title, fontweight='bold')
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                ax.text(j, i, format(val, fmt), ha='center', va='center',
                        color='white' if data[i, j] > data.max() * 0.6 else 'black',
                        fontsize=10)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path = f'{out_dir}/confusion_matrix_{model_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_all_models(df_test, models: dict, out_dir: str = 'outputs/figures') -> None:
    """
    models: {'lexicon': y_pred_array, 'svm': ..., 'indobert': ...}
    """
    y_true = df_test['sentiment'].values
    for name, y_pred in models.items():
        plot_confusion_matrix(y_true, y_pred, model_name=name, out_dir=out_dir)
