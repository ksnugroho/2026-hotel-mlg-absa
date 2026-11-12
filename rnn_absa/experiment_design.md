# Experiment Design: Bidirectional RNN Architectures for ABSA

## Title
A Comparative Study of Bidirectional Recurrent Architectures for Aspect-Based Sentiment Analysis on Indonesian Hotel Reviews

## Research Questions
- **RQ1**: How do unidirectional vs. bidirectional recurrent architectures compare on aspect-level ABSA?
- **RQ2**: How do the same architectures perform on overall sentiment classification?
- **RQ3**: Does bidirectionality consistently improve performance across both tasks and all architecture types?

## Models (6 total)
| # | Model | Bidirectional | Gating |
|---|-------|--------------|--------|
| 1 | RNN   | No  | None |
| 2 | LSTM  | No  | Full (forget, input, output) |
| 3 | GRU   | No  | Simplified (reset, update) |
| 4 | BiRNN | Yes | None |
| 5 | BiLSTM| Yes | Full |
| 6 | BiGRU | Yes | Simplified |

## Embedding
- **fastText Indonesian** (cc.id.300.bin) — 300-dim, pre-trained on Common Crawl + Wikipedia ID
- Frozen during training (not fine-tuned)

## Experiments
### Exp A — Aspect-level ABSA
- Input: `[ASPEK] {aspect} [ULASAN] {review_text}`
- Aspects: kebersihan, pelayanan, fasilitas, lokasi, harga, kenyamanan_kamar (6 aspects)
- Labels: positive, negative, neutral (3 classes)
- Evaluation: Macro-F1 (primary), Accuracy, Weighted-F1

### Exp B — Overall Sentiment
- Input: `{review_text}`
- Label: overall_sentiment (positive, negative, neutral)
- Evaluation: Macro-F1 (primary), Accuracy, Weighted-F1

## Data
- Reuse splits from `../data/processed/` (train/validation/test, 70:15:15, seed=42)
- Exp A: filter rows where aspect != overall_sentiment
- Exp B: filter rows where aspect == overall_sentiment

## Hyperparameter Grid (per model)
| Parameter | Values |
|-----------|--------|
| hidden_size | [128, 256] |
| num_layers | [1, 2] |
| dropout | [0.3] |
| learning_rate | [1e-3, 1e-4] |
| batch_size | [32] |
| epochs | up to 50, early stopping patience=5 |

Total: 2×2×2 = 8 combos × 6 models × 2 experiments = 96 runs

## Selection Metric
Macro-F1 on validation set

## Significance Test
McNemar's exact test + Holm correction (pairwise, 15 pairs per experiment)

## Primary Metric
Macro-F1 (consistent with main experiment)

## Outputs
```
rnn_absa/outputs/
├── logs/           # best_params per model per experiment
├── metrics/        # overall, per_class, per_aspect CSVs
├── models/         # saved model weights (.pt)
├── predictions/    # pred_{model}_{exp}_{split}.csv
└── figures/        # publication figures
```
