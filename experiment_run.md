# Experiment Run — Hotel Malang ABSA

Semua perintah dijalankan dari direktori root proyek:
```
cd /home/adminsvr/Documents/ksnugroho/2026-hotel-mlg-absa
conda activate ksn_nlp_task
```

---

## 0. Setup Data dari Label Studio

```bash
python src/export_dataset_label_studio.py
```

Output:
- `data/raw/dataset.csv` — 3574 ulasan + majority vote per aspek
- `data/raw/annotations.csv` — anotasi mentah semua annotator

---

## 1. EDA (Exploratory Data Analysis)

```bash
python src/eda.py
```

Output di `outputs/`:
- `summary.txt`, `label_distribution.csv`
- `annotator_agreement_detail.csv`, `annotator_agreement_summary.csv`
- `figures/` — 6 figure PNG

---

## 2. Inter-Annotator Agreement

```bash
python src/kappa_score.py
```

Output di `outputs/`:
- `kappa_cohens_pairwise.csv`
- `kappa_fleiss.csv`
- `figures/kappa_cohens_heatmap.png`, `figures/kappa_fleiss_bar.png`

---

## 3. Persiapan Data Eksperimen

```bash
python src/01_prepare_data.py
```

Output di `data/processed/`:
- `train.csv`, `validation.csv`, `test.csv`
- `../manifests/split_manifest.json`

---

## 4. Pendekatan 1: Lexicon (InSet)

```bash
python src/02_run_lexicon.py
```

Output di `outputs/predictions/`:
- `pred_lexicon_validation.csv`
- `pred_lexicon_test.csv`

---

## 5. Pendekatan 2: TF-IDF + SVM

```bash
python src/03_train_svm.py
```

Output:
- `outputs/models/svm_pipeline.pkl`
- `outputs/metrics/svm_grid_search_results.csv`
- `outputs/predictions/pred_svm_{validation,test}.csv`

---

## 6. Pendekatan 3: IndoBERT

```bash
python src/04_train_indobert.py
```

> Estimasi waktu: ~2–4 jam (tergantung GPU).  
> Checkpoint disimpan per epoch ke `outputs/models/indobert_best/`.

Output:
- `outputs/models/indobert_best/` — model + tokenizer HuggingFace
- `outputs/logs/` — training logs
- `outputs/predictions/pred_indobert_{validation,test}.csv`

---

## 7. Evaluasi & Uji Signifikansi

```bash
python src/05_evaluate_test.py
```

Output di `outputs/metrics/` dan `outputs/figures/`:
- `comparison_overall.csv` — Accuracy, Macro-F1, Precision, Recall ketiga model
- `metrics_{lexicon,svm,indobert}_test.csv` — metrik detail per model
- `per_aspect_{lexicon,svm,indobert}_test.csv` — metrik per aspek
- `confusion_matrix_{lexicon,svm,indobert}.png`
- `significance_mcnemar_holm.csv` — McNemar's exact + koreksi Holm

---

## 8. Inferensi Full Dataset (Model Terbaik)

Edit `MODEL_BEST` di `src/06_infer_full_dataset.py` sesuai hasil step 7:

```python
MODEL_BEST = 'indobert'   # ganti: 'lexicon' | 'svm' | 'indobert'
```

```bash
python src/06_infer_full_dataset.py
```

Output:
- `outputs/predictions/full_inference.csv` — prediksi seluruh dataset

---

## Struktur Output Akhir

```
outputs/
├── models/
│   ├── svm_pipeline.pkl
│   └── indobert_best/
├── predictions/
│   ├── pred_lexicon_{validation,test}.csv
│   ├── pred_svm_{validation,test}.csv
│   ├── pred_indobert_{validation,test}.csv
│   ├── predictions_combined_test.csv
│   └── full_inference.csv
├── metrics/
│   ├── comparison_overall.csv
│   ├── svm_grid_search_results.csv
│   └── significance_mcnemar_holm.csv
├── figures/
│   ├── confusion_matrix_*.png
│   ├── kappa_*.png
│   └── *.png (EDA figures)
└── logs/
```
