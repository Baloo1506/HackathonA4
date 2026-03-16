# 🛡️ TalentGuard AI

> Responsible HR Attrition Intelligence — Hackathon Project

<<<<<<< HEAD
<<<<<<< HEAD
A complete AI solution that predicts employee attrition risk, explains *why*, audits for bias, and presents everything in a clean HR dashboard. Built across four responsible AI pillars:

| Pillar | Implementation |
|--------|---------------|
| 🔐 **Cybersecurity** | GDPR anonymization, salted ID hashing, AI Act risk classification |
| ⚖️ **Ethics AI** | AIF360 fairness audit (Sex, RaceDesc), Reweighing debiasing |
| 🌿 **Frugal AI** | Random Forest (not deep learning), CodeCarbon emissions tracking |
| 🧠 **Explainable AI** | SHAP per-employee "why" explanations, global feature importance |

---

## Project Structure

```
talentguard/
├── app.py                      # Streamlit dashboard (5 pages)
├── train_pipeline.py           # End-to-end training script
├── requirements.txt
├── data/
│   └── HRDataset_v14.1.csv     # ← place your Kaggle CSV here
├── artifacts/                  # ← generated after training
│   ├── random_forest.pkl
│   ├── shap_values.npy
│   ├── risk_scores.csv
│   ├── model_card.json
│   └── feature_importance.csv
├── models/
│   └── attrition_model.py      # ML models, training, inference
└── utils/
    ├── preprocessing.py        # Anonymization, feature engineering
    ├── nlp_pipeline.py         # Sentiment analysis (DistilBERT / TextBlob)
    ├── explainability.py       # SHAP plots and explanations
    └── ethics_audit.py         # AIF360 fairness metrics + debiasing
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data
```bash
mkdir data
cp /path/to/HRDataset_v14.1.csv data/
```

### 3. Train the model
```bash
# Basic run
python train_pipeline.py --data data/HRDataset_v14.1.csv

# Compare all models (Frugal AI demo)
python train_pipeline.py --data data/HRDataset_v14.1.csv --compare-all

# Use logistic regression instead (even lighter)
python train_pipeline.py --data data/HRDataset_v14.1.csv --model logistic_regression
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

---

## Dashboard Pages

| Page | What it shows |
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

### Frugal AI
- **No neural networks** — Random Forest or Logistic Regression only
- **CodeCarbon** measures CO₂ emissions during training (typically < 0.001 g for this dataset)
- Model comparison included to demonstrate performance/cost tradeoffs

### Explainable AI
Every prediction includes:
1. A risk score (0–100%)
2. Top 3–5 contributing factors with SHAP values
3. Plain-English explanation: *"Employee X is at risk because salary is low and engagement score is below average"*
4. Suggested HR intervention

---

## Hackathon Deliverables

| Deliverable | Location |
|-------------|----------|
| **Model Card** | `artifacts/model_card.json` + dashboard "📋 Model Card" page |
| **Data Card** | Described in Model Card `training_data` section |
| **Demo** | `streamlit run app.py` |
| **Pitch** | Use the architecture diagram from the solution proposal |

---

## Dataset

Kaggle: [Human Resources Data Set](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)  
Designed by Dr. Rich Huebner & Dr. Carla Patalano.  
~400 synthetic employee records with performance, salary, engagement, and termination data.
=======
# Objective #
## Product : ##
- AI solution that helps HR to manage talents
- To help people discover four major themes of AI through a problem 
=======
A complete AI solution that predicts employee attrition risk, explains *why*, audits for bias, and presents everything in a clean HR dashboard. Built across four responsible AI pillars:
>>>>>>> 144a312 (Feat: Initial commit with project structure and initial files)

| Pillar | Implementation |
|--------|---------------|
| 🔐 **Cybersecurity** | GDPR anonymization, salted ID hashing, AI Act risk classification |
| ⚖️ **Ethics AI** | AIF360 fairness audit (Sex, RaceDesc), Reweighing debiasing |
| 🌿 **Frugal AI** | Random Forest (not deep learning), CodeCarbon emissions tracking |
| 🧠 **Explainable AI** | SHAP per-employee "why" explanations, global feature importance |

<<<<<<< HEAD
***Trouver les gens qui risquent de partir, et proposer des solutions pour qu'ils ne partent pas***
>>>>>>> 8e147f7 (Add retention strategy in French to README)
=======
---

## Project Structure

```
talentguard/
├── app.py                      # Streamlit dashboard (5 pages)
├── train_pipeline.py           # End-to-end training script
├── requirements.txt
├── data/
│   └── HRDataset_v14.1.csv     # ← place your Kaggle CSV here
├── artifacts/                  # ← generated after training
│   ├── random_forest.pkl
│   ├── shap_values.npy
│   ├── risk_scores.csv
│   ├── model_card.json
│   └── feature_importance.csv
├── models/
│   └── attrition_model.py      # ML models, training, inference
└── utils/
    ├── preprocessing.py        # Anonymization, feature engineering
    ├── nlp_pipeline.py         # Sentiment analysis (DistilBERT / TextBlob)
    ├── explainability.py       # SHAP plots and explanations
    └── ethics_audit.py         # AIF360 fairness metrics + debiasing
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data
```bash
mkdir data
cp /path/to/HRDataset_v14.1.csv data/
```

### 3. Train the model
```bash
# Basic run
python train_pipeline.py --data data/HRDataset_v14.1.csv

# Compare all models (Frugal AI demo)
python train_pipeline.py --data data/HRDataset_v14.1.csv --compare-all

# Use logistic regression instead (even lighter)
python train_pipeline.py --data data/HRDataset_v14.1.csv --model logistic_regression
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

---

## Dashboard Pages

| Page | What it shows |
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

### Frugal AI
- **No neural networks** — Random Forest or Logistic Regression only
- **CodeCarbon** measures CO₂ emissions during training (typically < 0.001 g for this dataset)
- Model comparison included to demonstrate performance/cost tradeoffs

### Explainable AI
Every prediction includes:
1. A risk score (0–100%)
2. Top 3–5 contributing factors with SHAP values
3. Plain-English explanation: *"Employee X is at risk because salary is low and engagement score is below average"*
4. Suggested HR intervention

---

## Hackathon Deliverables

| Deliverable | Location |
|-------------|----------|
| **Model Card** | `artifacts/model_card.json` + dashboard "📋 Model Card" page |
| **Data Card** | Described in Model Card `training_data` section |
| **Demo** | `streamlit run app.py` |
| **Pitch** | Use the architecture diagram from the solution proposal |

---

## Dataset

Kaggle: [Human Resources Data Set](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)  
Designed by Dr. Rich Huebner & Dr. Carla Patalano.  
~400 synthetic employee records with performance, salary, engagement, and termination data.
>>>>>>> 144a312 (Feat: Initial commit with project structure and initial files)
