# Architecture Documentation: Trusted HR-AI System

**Version**: 1.0  
**Date**: March 2026  
**Team**: GROUP N°17 (Hackathon A4)

---

## 1. System Overview

### 1.1 Purpose
Trusted HR-AI is an explainable machine learning system designed to predict voluntary employee attrition and empower HR teams with transparent, actionable retention insights. The system combines predictive analytics, explainability (SHAP), and fairness auditing into a unified platform.

### 1.2 Scope
- **Input**: HR dataset (employees, demographics, engagement, salary, behavior)
- **Processing**: Feature engineering, NLP, model training, explainability, fairness audit
- **Output**: Risk scores + explanations for active employees; fairness metrics; model performance statistics

---

## 2. Technical Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRUSTED HR-AI PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 1: DATA INGESTION & PRIVACY ───────────────────────────────┐
│                                                                     │
│  Input:  HRDataset_v14_enriched.csv (400 rows, 40 columns)          │
│                                                                     │
│  Step 1a: Load & Filter                                             │
│    └─ Read CSV                                                      │
│    └─ Filter: EmploymentStatus ∈ {Active, Voluntarily Terminated}   │
│    └─ Exclude: Terminated for Cause                                 │
│    └─ Create binary target: 1 = Voluntarily Terminated, 0 = Active  │
│                                                                     │
│  Step 1b: Privacy Cleaning                                          │
│    └─ Drop direct identifiers (names, IDs)                          │
│    └─ Drop outcome leakage (TermReason, dates, status)              │
│    └─ Separate protected attributes (Sex, Race, Ethnicity)          │
│    └─ Drop quasi-identifiers (Zip, State, DOB, raw DateofHire)      │
│    └─ Validate no data leakage remains                              │
│                                                                     │
│  Output: Clean dataset, protected attributes in separate df         │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 2: FEATURE ENGINEERING ────────────────────────────────────┐
│                                                                     │
│  Structured Features:                                               │
│    ├─ Temporal: age (from DOB), tenure_years (hire → term/snapshot) │
│    ├─ Financial: Salary, salary_vs_dept_mean (relative pay)         │
│    ├─ Engagement: EngagementSurvey, EmpSatisfaction, composite      │
│    ├─ Behavioral: Absences, DaysLateLast30, SpecialProjectsCount    │
│    └─ Categorical: Department, Position, MaritalDesc, etc. (encoded)│
│                                                                     │
│  NLP Features (extracted but NOT modeled):                          │
│    ├─ VADER Sentiment: transfer_request_sentiment, feedback_sentiment
│    ├─ Keyword Flags: has_transfer_request, feedback_has_departure   │
│    └─ EDA Flags: compensation, growth, workload, management themes  │
│                                                                     │
│  Output: 14 model features + advisory NLP signals                   │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 3: DATA PREPARATION ───────────────────────────────────────┐
│                                                                     │
│  Stratified Train/Test Split (80/20):                               │
│    ├─ Training: 320 employees                                       │
│    ├─ Test: 80 employees (held-out, never touched)                  │
│    └─ Preserve attrition rate in both sets (15%)                    │
│                                                                     │
│  Class Balancing (SMOTE on training only):                          │
│    ├─ Synthesize minority-class samples                             │
│    ├─ Training becomes 360 Active, 280 Attrition (balanced)         │
│    ├─ Test set remains untouched (true 15% attrition rate)          │
│    └─ Rationale: Avoid data loss; synthesize realistic patterns     │
│                                                                     │
│  Preprocessing:                                                     │
│    ├─ Impute missing values (median for numeric)                    │
│    ├─ Scale features (StandardScaler for LR only; tree models native)
│    └─ Encode categoricals (LabelEncoder)                            │
│                                                                     │
│  Output: X_train, X_test, y_train, y_test (balanced for training)   │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 4: MODEL TRAINING & SELECTION ─────────────────────────────┐
│                                                                     │
│  Candidate Models:                                                  │
│                                                                     │
│    Model 1: Logistic Regression                                     │
│      ├─ Algorithm: Linear classification + sigmoid                  │
│      ├─ Performance: AUC=0.82, F1=0.61 (baseline)                   │
│      ├─ Advantage: Fully interpretable coefficients                 │
│      └─ Use: SHAP LinearExplainer                                   │
│                                                                     │
│    Model 2: Random Forest                                           │
│      ├─ Algorithm: Ensemble of decision trees (200 trees)           │
│      ├─ Performance: AUC=0.85, F1=0.66                              │
│      ├─ Advantage: Captures non-linear relationships                │
│      └─ Use: SHAP TreeExplainer                                     │
│                                                                     │
│    Model 3: XGBoost ⭐ SELECTED                                     │
│      ├─ Algorithm: Gradient boosting (200 boosting rounds)          │
│      ├─ Hyperparameters: max_depth=5, learning_rate=0.1             │
│      ├─ Performance: AUC=0.87 ✅, F1=0.68 ✅ (best)                 │
│      ├─ Advantages: High performance + handles feature interactions │
│      └─ Use: SHAP TreeExplainer                                     │
│                                                                     │
│  Selection Criterion: Best AUC-ROC on test set                      │
│  Output: Best model (XGBoost), test performance metrics             │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 5: EXPLAINABILITY (SHAP) ──────────────────────────────────┐
│                                                                     │
│  SHAP Framework:                                                    │
│    ├─ Algorithm: TreeExplainer (XGBoost model)                      │
│    ├─ Computation: Tree path traversal (fast, accurate)             │
│    └─ Output: Shapley values per feature per prediction             │
│                                                                     │
│  Global Explanations (all test set):                                │
│    ├─ Feature Importance: Mean |SHAP| per feature                   │
│    ├─ Summary Plot: Visualization of top drivers                    │
│    └─ Interpretation: "Engagement Survey contributes 0.18 to risk"  │
│                                                                     │
│  Local Explanations (per employee):                                 │
│    ├─ Waterfall Plot: Contribution of each feature to prediction    │
│    ├─ Base Value: Population average risk (15%)                     │
│    ├─ Feature Contributions: How each feature pushes risk up/down   │
│    └─ Interpretation: "Employee X is 75% risk due to low engagement"│
│                                                                     │
│  Output: SHAP values, summary plots, waterfall plots per employee   │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 6: FAIRNESS AUDIT ─────────────────────────────────────────┐
│                                                                     │
│  Post-Hoc Analysis (on test set):                                   │
│    ├─ Retrieve protected attributes: Sex, Race, Ethnicity           │
│    ├─ Stratify predictions by demographic                           │
│    ├─ Compute metrics:                                              │
│    │  ├─ False Positive Rate (FPR): % of active flagged as risk     │
│    │  └─ False Negative Rate (FNR): % of risk missed by model       │
│    ├─ Compare: max(FPR_group1 - FPR_group2)                         │
│    └─ Flag: If disparity >10%, investigate further                  │
│                                                                     │
│  Results (test set):                                                │
│    ├─ Sex: <1% disparity ✅ (Fair)                                  │
│    ├─ Race: <1% disparity ✅ (Fair)                                 │
│    └─ Ethnicity: <1% disparity ✅ (Fair)                            │
│                                                                     │
│  Output: Fairness metrics, disparity analysis, recommendations      │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 7: RISK REPORTING & DATA EXPORT ───────────────────────────┐
│                                                                     │
│  Active Employee Scoring & Profiling:                               │
│    ├─ Load ALL active employees (not just test set)                 │
│    ├─ Generate risk scores (0–1, calibrated probabilities)          │
│    ├─ Extract top 3 SHAP drivers per employee                       │
│    └─ Anonymize: Remove names, use anonymized IDs                   │
│                                                                     │
│  Export to Web App:                                                 │
│    ├─ CSV Format: id, risk_score, driver_1-3, action_1-3, ...       │
│    └─ SHAP Matrix: Local explanations for drill-down views          │
│                                                                     │
│  Output: risk_report_active.csv & shap_active_df.csv                │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 8: WEB DASHBOARD & GEN AI (STREAMLIT + GEMMA 3) ───────────┐
│                                                                     │
│  Frontend UI (Streamlit):                                           │
│    ├─ Loads risk_report_active.csv & shap_active_df.csv             │
│    ├─ Displays High-Risk Employee Table (Filters & Search)          │
│    └─ Interactive "What-If" Simulator for HR scenario testing       │
│                                                                     │
│  Local Gen AI Integration (Gemma 3:4B):                             │
│    ├─ Privacy: Runs 100% locally (Zero API leakage)                 │
│    ├─ Input Prompt: Anonymized SHAP drivers & NLP sentiments        │
│    └─ Output: Natural language summaries & targeted HR action plans │
│                                                                     │
│  Output: Interactive, secure, and explainable HR decision platform  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Raw HR Data (HRDataset_v14_enriched.csv)
        │
        ↓
┌──────────────────────────────┐
│  Data Cleaning & Validation  │  Remove identifiers, outcome leakage
├──────────────────────────────┤
│  Privacy-Compliant Dataset   │  Separate protected attributes
└──────────────────────────────┘
        │
        ↓
┌──────────────────────────────┐
│  Feature Engineering         │  Temporal, financial, engagement
├──────────────────────────────┤
│  Structured Features (14)    │  + NLP advisory signals
│  + Protected Attributes (df) │
└──────────────────────────────┘
        │
        ├─────────────────────────────────────────┐
        │                                         │
        ↓                                         ↓
┌──────────────────┐                  ┌──────────────────┐
│   Train Set      │                  │   Test Set       │
│   (320 rows)     │                  │   (80 rows)      │
│   ↓ SMOTE ↓      │                  │  (Hold-out)      │
│   (640 rows)     │                  └──────────────────┘
└──────────────────┘                            │
        │                                       ↓
        ↓                                ┌──────────────────┐
┌──────────────────┐                     │  Model Eval      │
│  Model Training  │                     │  AUC=0.87 ✅     │
│  LR/RF/XGBoost   │ ─────────────→      │  F1=0.68 ✅      │
└──────────────────┘                     └──────────────────┘
        │                                       │
        └───────────────────────┬───────────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Select Best Model     │
                    │ (XGBoost)             │
                    └───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ↓                       ↓
        ┌────────────────────┐  ┌──────────────────┐
        │   SHAP Analysis    │  │  Fairness Audit  │
        │  Global + Local    │  │  Disparate Impact│
        └────────────────────┘  └──────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────────┐
                    │ Active Employees Scoring  │
                    │ Calibrated Model          │
                    └───────────────────────────┘
                                │
                                ↓
                    ┌───────────────────────────┐
                    │ Risk Report Generation    │
                    │ SHAP Drivers + Actions    │
                    │ CSV Export                │
                    └───────────────────────────┘
                                │
                                ↓
             ┌────────────────────────────────────────┐
             │         STREAMLIT WEB APP UI           │
             │   Dashboard / Tables / SHAP Graphs     │
             └────────────────────────────────────────┘
                    │                         ↑
      Anonymized SHAP Drivers                 │
        & Sentiment Scores                    │
                    │                  Generated Natural
                    ↓                 Language Explanations
             ┌────────────────────────────────────────┐
             │          LOCAL LLM GENERATION          │
             │    Gemma 3:4B (Running Locally)        │
             │ Translates SHAP into HR Action Plans   │
             └────────────────────────────────────────┘
```

---



### Documentation Standards
- [Model Card Best Practices](https://huggingface.co/docs/hub/model-cards)
- [Data Card Best Practices](https://arxiv.org/abs/1803.09010)

---

**Architecture Document Certification**: This document describes the complete technical architecture of Trusted HR-AI as of March 2026. It has been prepared by GROUP N°17 for hackathon evaluation and future reference.
