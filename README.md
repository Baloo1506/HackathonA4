# 🛡️ TalentGuard AI — Attrition Risk Predictor

This repository contains an end-to-end pipeline that trains an attrition prediction model, runs fairness audits, computes SHAP explanations, and serves an interactive Streamlit dashboard for exploration and reporting.

Prerequisites
- Python 3.9+
- pip
- Recommended: virtual environment (venv or conda)

Quick install
```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

Project highlights
- `train_pipeline.py` — trains the model and writes outputs in `artifacts/` and app-friendly `models/`
- `app.py` — Streamlit dashboard (reads from `models/`)
- `artifacts/` — raw training outputs (pickles, shap arrays, CSVs)
- `models/` — saved artefacts the dashboard expects (`saved_model.joblib`, `X_full.parquet`, `y_full.parquet`, `feature_cols.joblib`, `employee_ids.joblib`, `shap_values.joblib`, `metrics.json`)

Run locally (short)
1) Place dataset in `data/` (e.g., `data/HRDataset_v14.csv`).
2) Train the model (this writes `artifacts/` and `models/`):
```bash
python train_pipeline.py --data data/HRDataset_v14.csv
```
3) Start the dashboard from the repo root:
```bash
streamlit run app.py
```

Notes / troubleshooting
- If the dashboard shows "No model found":
  - Confirm `models/saved_model.joblib` exists. If not, re-run `train_pipeline.py` or run `python scripts/sync_artifacts_to_models.py` to copy from `artifacts/`.
  - If Streamlit was running before files were created, clear cache and restart:
    ```bash
    streamlit cache clear
    streamlit run app.py
    ```
- Optional warnings about `tensorflow` / `fairlearn` are informational — install those packages only if you need the related fairness algorithms.

If you want, I can run the dashboard here, capture the console logs, and help debug any errors — say "Run the dashboard" and I'll start it and paste the output.
# 🛡️ TalentGuard AI — Attrition Risk Predictor

Responsible HR attrition intelligence: this repository contains an end-to-end pipeline that trains an attrition prediction model, runs fairness audits, computes SHAP explanations, and serves an interactive Streamlit dashboard for exploration and reporting.

This README contains setup and run instructions for local development.

## Quick facts

- Model types: Random Forest (default), Logistic Regression, Gradient Boosting
- Explainability: SHAP (global + per-employee)
- Fairness: IBM AIF360 audits and optional reweighing
- Dashboard: Streamlit (interactive visualizations and model card)

## Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- pip
- Recommended virtual environment (venv/conda)

Install required Python packages:

```bash
python -m pip install -r requirements.txt
```

Some optional features (AdversarialDebiasing, certain AIF360 reductions) require extra packages such as `tensorflow` or `fairlearn`. Those are optional and will only enable extra functionality.

## Run locally (recommended workflow)

1) Create and activate a venv

```bash
# 🛡️ TalentGuard AI — Attrition Risk Predictor

This repository contains an end-to-end pipeline that trains an attrition prediction model, runs fairness audits, computes SHAP explanations, and serves an interactive Streamlit dashboard for exploration and reporting.

## Quick summary

- Model types: Random Forest (default), Logistic Regression, Gradient Boosting
- Explainability: SHAP (global + per-employee)
- Fairness: IBM AIF360 audits and optional reweighing
- Dashboard: Streamlit (interactive visualizations and model card)

## Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- pip
- Recommended: virtual environment (venv/conda)

## Install

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

Some optional features (AdversarialDebiasing, certain AIF360 reductions) require extra packages such as `tensorflow` or `fairlearn`. Those are optional and will only enable extra functionality.

## Project layout (important files)

- `train_pipeline.py` — trains the model and writes outputs in `artifacts/` and `models/`
- `app.py` — Streamlit dashboard (reads from `models/`)
- `artifacts/` — raw training outputs (pickles, SHAP arrays, CSVs)
- `models/` — saved artefacts the dashboard expects (`saved_model.joblib`, `X_full.parquet`, `y_full.parquet`, `feature_cols.joblib`, `employee_ids.joblib`, `shap_values.joblib`, `metrics.json`)
- `utils/` — preprocessing, NLP, explainability, and audit helpers

## Run locally — step by step

1) Place dataset in `data/` (for example `data/HRDataset_v14.csv`).

2) Train the model (this writes `artifacts/` and `models/`):

```bash
python train_pipeline.py --data data/HRDataset_v14.csv
```

You can also compare models or select a different algorithm:

```bash
python train_pipeline.py --data data/HRDataset_v14.csv --compare-all
python train_pipeline.py --data data/HRDataset_v14.csv --model logistic_regression
```

3) Start the Streamlit dashboard from the repository root:

```bash
streamlit run app.py
```

Important: start Streamlit from the repo root so `Path('models')` resolves correctly.

If Streamlit was running before `models/` files existed, clear Streamlit's cache and restart:

```bash
streamlit cache clear
streamlit run app.py
```

## Troubleshooting

- App shows "No model found":
  - Confirm `models/saved_model.joblib` exists. If not, re-run `train_pipeline.py` (it now writes model files) or run `python scripts/sync_artifacts_to_models.py` to copy from `artifacts/`.
  - Stop Streamlit, clear cache, and restart.

- SHAP or plotting errors:
  - Confirm `shap` and `plotly` are installed. Use `pip install shap plotly`.

- Optional fairness features warn about missing packages (tensorflow/fairlearn):
  - These are optional. Install them only if you need the corresponding debiasing algorithms.

## Development notes

- The main model logic is in `models/attrition_model.py`. The Streamlit app expects the `models/` directory to contain the saved model and a set of artefacts with specific names. If you change filenames, update `app.py`'s `load_artefacts()` accordingly.

## Quick commands

```bash
# create venv + install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train (writes artifacts + models/)
python train_pipeline.py --data data/HRDataset_v14.csv

# run dashboard
streamlit run app.py
```

## Need help?

If the dashboard still shows no data after following these steps, paste the Streamlit terminal output or the web UI traceback here and I'll help debug further.

|------|--------------|
| **📊 Overview** | KPIs, risk distribution donut, histogram, top-10 highest-risk employees |
| **🔍 Employee Explorer** | Filterable/sortable employee table, individual deep-dive with retention suggestions |
| **🧠 Explainability** | Global SHAP feature importance, per-employee waterfall explanation |
| **⚖️ Ethics Audit** | AIF360 fairness metrics (pre- and post-modelling), debiasing status, AI Act note |
| **📋 Model Card** | Full model transparency doc, downloadable JSON |

---

## Responsible AI Highlights

### Cybersecurity
- All PII columns (`Employee_Name`, `DOB`, `Email`, etc.) are **dropped before any processing**
- Employee IDs are replaced with a **salted SHA-256 hash** — irreversible without the salt
- System is classified as **High-Risk** under the EU AI Act (employment domain)

### Ethics AI
```python
# Fairness check runs automatically
pre_audit = audit_dataset(X, y, df_sensitive)
# If bias detected (DI < 0.8), reweighing is applied:
X, y, weights = apply_reweighing(X, y, df_sensitive, attr="Sex")
model.fit(X, y, sample_weight=weights)
```

# 🛡️ TalentGuard AI — Attrition Risk Predictor

Responsible HR attrition intelligence: this repository contains an end-to-end pipeline that trains an attrition prediction model, runs fairness audits, computes SHAP explanations, and serves an interactive Streamlit dashboard for exploration and reporting.

This README describes how to set up and run the project locally, what artifacts are generated, and common troubleshooting steps.

## Quick facts

- Model types: Random Forest (default), Logistic Regression, Gradient Boosting
- Explainability: SHAP (global + per-employee)
- Fairness: IBM AIF360 audits and optional reweighing
- Dashboard: Streamlit (interactive visualizations and model card)

## Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- pip
- Recommended virtual environment (venv/conda)

Install required Python packages:

```bash
python -m pip install -r requirements.txt
```

Note: some optional features (AdversarialDebiasing, certain AIF360 reductions) require extra packages such as `tensorflow` or `fairlearn`. Those are optional and produce warnings when missing.

## Project layout

Key files and folders:

- `app.py` — Streamlit dashboard (5 pages)
- `train_pipeline.py` — end-to-end training + artifact writer
- `requirements.txt` — project dependencies
- `data/` — place input CSV dataset here
- `artifacts/` — outputs created by the training pipeline
- `models/` — app-friendly artefacts (saved model, X/y parquet, feature cols)
- `utils/` — preprocessing, NLP, explainability, and audit helpers

## Run locally — step by step

1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux (zsh/bash)
# .venv\Scripts\activate    # Windows (PowerShell)
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Place the dataset

Download the Kaggle HR dataset and place it under `data/`.

```bash
mkdir -p data
cp /path/to/HRDataset_v14.csv data/HRDataset_v14.csv
