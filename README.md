# 🛡️ TalentGuard AI — Attrition Risk Predictor

Responsible HR attrition intelligence: this repository contains an end-to-end pipeline that trains an attrition prediction model, runs fairness audits, computes SHAP explanations, and serves an interactive Streamlit dashboard for exploration and reporting.

## Quick summary

- **Model types**: Random Forest (default), Logistic Regression, Gradient Boosting
- **Explainability**: SHAP (global + per-employee)
- **Fairness**: IBM AIF360 audits and optional reweighing
- **Dashboard**: Streamlit (interactive visualizations and model card)

## Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- pip
- Recommended: virtual environment (venv/conda)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate    # Windows (PowerShell)
pip install -r requirements.txt
```

**Note**: Some optional features (AdversarialDebiasing, certain AIF360 reductions) require extra packages such as `tensorflow` or `fairlearn`. These are optional and will produce warnings if missing—only install them if you need those specific debiasing algorithms.

## Project layout

Key files and folders:

- **`app.py`** — Streamlit dashboard (5 pages: Overview, Employee Explorer, Explainability, Ethics Audit, Model Card)
- **`train_pipeline.py`** — end-to-end training + artifact writer
- **`requirements.txt`** — project dependencies
- **`data/`** — place input CSV dataset here (e.g., `HRDataset_v14.csv`)
- **`artifacts/`** — raw training outputs (pickles, SHAP arrays, CSVs, model card)
- **`models/`** — app-friendly artefacts the dashboard reads:
  - `saved_model.joblib` (trained model)
  - `X_full.parquet` and `y_full.parquet` (training data)
  - `feature_cols.joblib` (feature names)
  - `employee_ids.joblib` (employee ID list)
  - `shap_values.joblib` (SHAP matrix)
  - `metrics.json` (model metrics)
- **`utils/`** — preprocessing, NLP, explainability, and audit helpers

## Run locally — step by step

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Download the Kaggle HR dataset and place it under `data/`:

```bash
mkdir -p data
cp /path/to/HRDataset_v14.csv data/HRDataset_v14.csv
```

### 4. Train the model

This writes outputs to both `artifacts/` (canonical) and `models/` (app-friendly):

```bash
# Basic training (uses auto-detected default if --data omitted)
python train_pipeline.py --data data/HRDataset_v14.csv

# Compare multiple model types
python train_pipeline.py --data data/HRDataset_v14.csv --compare-all

# Use logistic regression
python train_pipeline.py --data data/HRDataset_v14.csv --model logistic_regression
```

### 5. Start the Streamlit dashboard

From the repository root, run:

```bash
streamlit run app.py
```

**Important**: Start Streamlit from the repo root so `Path('models')` resolves correctly.

If Streamlit was running before `models/` files existed, clear Streamlit's cache and restart:

```bash
streamlit cache clear
streamlit run app.py
```

## Dashboard features

| Page | Description |
|------|-------------|
| 📊 **Overview** | KPIs, risk distribution donut chart, histogram, top 10 highest-risk employees |
| 🔍 **Employee Explorer** | Filterable/sortable employee table, individual deep-dive with retention suggestions |
| 🧠 **Explainability** | Global SHAP feature importance, per-employee waterfall explanation |
| ⚖️ **Ethics Audit** | AIF360 fairness metrics (pre- and post-modelling), debiasing status, EU AI Act classification |
| 📋 **Model Card** | Full model transparency document, downloadable JSON |

## Responsible AI highlights

### Cybersecurity
- All PII columns (Employee_Name, DOB, Email, etc.) are dropped before any processing
- Employee IDs are replaced with a salted SHA-256 hash — irreversible without the salt
- System is classified as **High-Risk** under the EU AI Act (employment domain)

### Fairness & Ethics
Fairness checks run automatically during training:

```python
# Pre-audit for bias
pre_audit = audit_dataset(X, y, df_sensitive)

# If bias detected (DI < 0.8), reweighing is applied:
X, y, weights = apply_reweighing(X, y, df_sensitive, attr="Sex")
model.fit(X, y, sample_weight=weights)
```

## Troubleshooting

### App shows "No model found"

1. Confirm `models/saved_model.joblib` exists. If not:
   - Re-run `python train_pipeline.py --data data/HRDataset_v14.csv`
   - Or sync from legacy artifacts: `python scripts/sync_artifacts_to_models.py`

2. Stop Streamlit, clear cache, and restart:
   ```bash
   streamlit cache clear
   streamlit run app.py
   ```

### SHAP or plotting errors

Confirm `shap` and `plotly` are installed:

```bash
pip install shap plotly
```

### Optional fairness warnings (tensorflow/fairlearn)

These warnings are informational. Install these packages only if you need the corresponding debiasing algorithms:

```bash
pip install tensorflow fairlearn
```

## Development notes

- The main model logic is in `models/attrition_model.py`
- The Streamlit app expects `models/` to contain the saved model and artefacts with specific filenames
- If you change filenames, update `app.py`'s `load_artefacts()` function accordingly

## Quick commands (summary)

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
