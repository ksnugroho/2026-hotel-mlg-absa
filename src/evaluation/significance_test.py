"""
McNemar's test (exact) + koreksi Holm untuk perbandingan tiga model.
Pasangan: Lexicon vs SVM, Lexicon vs IndoBERT, SVM vs IndoBERT.
"""

import numpy as np
import pandas as pd
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests


def mcnemar_exact(y_true, y_pred_a, y_pred_b) -> tuple[float, int, int]:
    """
    McNemar exact (binomial).
    b = A benar, B salah
    c = A salah, B benar
    p-value dua sisi dari distribusi binomial.
    """
    correct_a = (np.array(y_pred_a) == np.array(y_true))
    correct_b = (np.array(y_pred_b) == np.array(y_true))

    b = int(np.sum(correct_a & ~correct_b))   # A benar, B salah
    c = int(np.sum(~correct_a & correct_b))   # A salah, B benar
    n = b + c

    if n == 0:
        return 1.0, b, c

    # exact binomial two-sided
    k    = min(b, c)
    pval = 2 * binom.cdf(k, n, 0.5)
    pval = min(pval, 1.0)
    return pval, b, c


def run_significance_tests(
    y_true,
    y_pred_lexicon,
    y_pred_svm,
    y_pred_indobert,
    alpha:   float = 0.05,
    out_dir: str   = 'outputs/metrics',
) -> pd.DataFrame:
    import os; os.makedirs(out_dir, exist_ok=True)

    pairs = [
        ('Lexicon', 'SVM',       y_pred_lexicon, y_pred_svm),
        ('Lexicon', 'IndoBERT',  y_pred_lexicon, y_pred_indobert),
        ('SVM',     'IndoBERT',  y_pred_svm,     y_pred_indobert),
    ]

    rows = []
    pvals = []
    for name_a, name_b, pred_a, pred_b in pairs:
        pval, b, c = mcnemar_exact(y_true, pred_a, pred_b)
        rows.append({
            'model_a': name_a, 'model_b': name_b,
            'b': b, 'c': c, 'discordant': b + c,
            'raw_pvalue': round(pval, 6),
        })
        pvals.append(pval)

    # koreksi Holm
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='holm')
    for i, row in enumerate(rows):
        row['pvalue_holm'] = round(pvals_corrected[i], 6)
        row['significant'] = bool(reject[i])
        row['alpha']       = alpha

    df = pd.DataFrame(rows)
    path = f'{out_dir}/significance_mcnemar_holm.csv'
    df.to_csv(path, index=False)
    print(f'\nSaved: {path}')
    print('\nMcNemar + Holm correction:')
    print(df.to_string(index=False))
    return df
