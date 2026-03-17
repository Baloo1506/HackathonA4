# Trusted HR-AI: Employee Retention Decision-Support System# HackathonA4



**Team**: GROUP N°17 — GOETGHEBEUR Alexandre, ALI Muhammad, LEBRETON Gaspard, POINTEAU Adrien, TEPLOV MichaelGROUP N°17 : GOETGHEBEUR Alexandre, ALI Muhammad, LEBRETON Gaspard, POINTEAU Adrien, TEPLOV Michael  



---

# Trusted HR-AI: Employee Retention Dashboard

## 📋 Executive Summary

## Project Overview

**Trusted HR-AI** is an explainable machine learning platform that predicts voluntary employee attrition and empowers HR teams to intervene proactively. Built on rigorous fairness principles, the system combines predictive analytics with transparent explanations to help organizations retain top talent while respecting employee privacy and ensuring ethical decision-making.This project is the backend intelligence for an interactive HR web application designed to proactively manage employee turnover. The platform features a dashboard that filters and displays active employees whose predicted probability of resigning exceeds a critical threshold. 



This repository contains the complete analytics pipeline: from raw HR data through feature engineering, model training, fairness auditing, and explainability analysis—culminating in actionable risk profiles for each high-risk employee.By clicking on a specific high-risk employee, HR managers can access a detailed "Risk Profile" explaining exactly *why* that person is predicted to leave, empowering them to take targeted, fair, and secure preventive actions.



------



## 🎯 Project Objectives##  Key Features & User Experience

* **High-Risk Employee Dashboard**: A centralized view displaying all active employees at risk of leaving.

### Primary Goals* **Individual Drill-Down (Explainability)**: Click on any employee to view their personalized risk drivers (e.g., Low Salary, Poor Sentiment, High Absences) powered by SHAP values.

1. **Predict Attrition Risk**: Accurately identify active employees at high risk of voluntary resignation.* **Secure & Anonymous**: Built with Privacy-by-Design. All Personally Identifiable Information (PII) like names are removed before modeling to ensure GDPR compliance.

2. **Explainability**: Provide transparent, interpretable explanations for each prediction using SHAP values.* **Ethical & Fair**: The model is rigorously audited for demographic parity to ensure risk scores are not biased against specific genders or ethnicities.

3. **Privacy-by-Design**: Remove all direct identifiers and outcome-leakage fields before modeling.

4. **Ethical AI**: Conduct rigorous fairness audits to ensure predictions are unbiased across demographic groups.---

5. **Actionability**: Generate clear, prioritized risk profiles that enable targeted HR interventions.

## Technical Pipeline & Scope

### Success CriteriaOur pipeline processes a real-world HR dataset and transforms it into actionable, ethical insights for the frontend dashboard:

- **Model Performance**: AUC-ROC > 0.85, F1 > 0.65 on unseen test data

- **Fairness**: Demographic parity analysis (no >10% disparity in false positive/negative rates across sex/race)1. **Cybersecurity & Data Privacy**: 

- **Explainability**: SHAP values explain >80% of prediction variance for top employees   * Direct identifiers (`Employee_Name`) are stripped out.

- **Reproducibility**: End-to-end pipeline runs in <5 minutes on standard hardware   * Only anonymized `EmpID`s are passed to the frontend to protect employee privacy.

2. **Feature Engineering & NLP (VADER)**: 

---   * We calculate employee tenure and extract qualitative insights from unstructured text (Internal Transfer Requests and HR Feedback).

   * `vaderSentiment` is used to calculate sentiment polarity scores, translating textual feedback into numerical risk signals.

## 📊 Scope & Deliverables3. **Data Augmentation (SMOTE)**: 

   * Because the dataset is small and imbalanced, we use **SMOTE** (Synthetic Minority Over-sampling Technique) via `imblearn` on the training set. This synthesizes minority-class samples (resignations) to train a robust model without exposing real data.

### In Scope4. **Predictive Modeling**: 

- **Dataset**: ~400 employees from HR records (HRDataset_v14_enriched.csv)   * We train robust classifiers (Logistic Regression, Random Forest, XGBoost) and apply probability calibration to generate an accurate risk percentage (0-100%) for the web interface.

- **Modeling**: Multi-class binary classification (Active vs. Voluntarily Terminated)5. **Explainable AI (SHAP)**: 

- **Features**: Structured HR metrics + NLP-derived signals from feedback text   * We integrate `SHAP` (SHapley Additive exPlanations). For every prediction, SHAP calculates the exact contribution of each feature, allowing the web app to display local waterfall plots and top risk factors for any specific user.

- **Explainability**: SHAP waterfall plots and feature importance analysis6. **Ethics & Fairness Audit**: 

- **Fairness**: Disparate impact analysis on protected attributes (Sex, Race, Ethnicity)   * A disparate impact analysis is conducted on sensitive attributes (`Sex`, `RaceDesc`, `HispanicLatino`) to ensure the model's predictions remain equitable and unbiased across all demographics.



### Out of Scope---

- **Forecasting**: Not a time-series model; predictions are based on snapshot data

- **Causal Inference**: Associations identified, not causal mechanisms## Personae

- **Terminated-for-Cause**: Excluded from analysis (focus on voluntary resignation)* **The Client (HR Manager)**: "I need a secure, user-friendly site where I can see *who* is at risk of leaving, click on their profile to understand *why*, and intervene before it's too late."

- **Real-time Deployment**: Proof-of-concept; production requires security hardening* **The Solution Provider (HR-AI Company)**: "We deliver a web-based AI dashboard that empowers HR teams with transparent, ethical, and highly accurate retention insights."



------



## 👥 User Personas## Instructions



### 1. **HR Manager / Operations Lead**### Prerequisites

**Goals**: Ensure you have Python installed along with the required backend libraries:

- Identify which employees are most at risk of leaving```bash

- Understand key risk factors for each employeepip install pandas numpy scikit-learn imbalanced-learn shap vaderSentiment xgboost
- Take targeted, fair preventive actions

**Pain Points**:
- Manual, reactive approach to turnover detection
- No transparency into why employees leave
- Limited visibility into early warning signs
- Concern about algorithmic bias in decisions

**How This Solution Helps**:
- Dashboard displays all at-risk employees in one view
- Click-through drill-down shows personalized risk factors with confidence scores
- SHAP explanations justify each prediction (no "black box" decisions)
- Fairness audit report confirms model treats all demographics equally

### 2. **Data Analyst / HR Insights Team**
**Goals**:
- Monitor model performance and data quality
- Investigate patterns in feedback and sentiment
- Generate periodic reports on attrition trends
- Debug edge cases and improve model

**Pain Points**:
- Static, ad-hoc reporting
- Hard to extract insights from unstructured HR data
- No systematic fairness monitoring
- Difficult to reproduce analyses

**How This Solution Helps**:
- Fully reproducible Jupyter notebook with documented pipeline
- Automated NLP sentiment and keyword extraction
- Fairness audit with demographic breakdowns
- Feature importance rankings to guide focus areas

### 3. **Compliance / Ethics Officer**
**Goals**:
- Ensure AI system respects privacy and fairness regulations (GDPR, fair lending, etc.)
- Audit decision-making process for bias
- Maintain transparency and explainability documentation
- Manage liability and risk

**Pain Points**:
- "Black box" ML models difficult to audit
- Risk of demographic bias in hiring/retention decisions
- PII exposure in data pipelines
- Regulatory uncertainty

**How This Solution Helps**:
- Formal data card and model card documenting choices
- No direct identifiers used in modeling
- SHAP explainability enables auditing individual decisions
- Disparate impact analysis with clear fairness metrics

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HR ATTRITION PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

1. DATA INGESTION & PRIVACY
   ├─ Load: HRDataset_v14_enriched.csv (~400 rows, 40 columns)
   ├─ Filter: Keep Active + Voluntarily Terminated employees
   ├─ Remove: Direct identifiers (Name, EmpID, Manager names)
   ├─ Exclude: Outcome leakage fields (TermReason, Termd date)
   └─ Separate: Protected attributes for fairness audit only

2. FEATURE ENGINEERING
   ├─ Structured Features
   │  ├─ Temporal: age (from DOB), tenure_years (hire → term/snapshot)
   │  ├─ Financial: Salary, salary_vs_dept_mean (relative pay signal)
   │  ├─ Engagement: EngagementSurvey, EmpSatisfaction
   │  ├─ Behavioral: Absences, DaysLateLast30, SpecialProjectsCount
   │  └─ Composite: engagement_x_satisfaction (interaction term)
   │
   ├─ NLP Features (VADER Sentiment Analysis)
   │  ├─ Transfer Request: has_transfer_request, transfer_sentiment
   │  ├─ Feedback Signals: feedback_sentiment, feedback_has_departure_intent
   │  └─ ⚠️ Note: Raw text excluded from model (potential data leakage)
   │
   └─ Categorical Encoding
      └─ Label encode: Department, Position, MaritalDesc, etc.

3. TRAIN/TEST SPLIT & BALANCING
   ├─ Stratified 80/20 split (preserve class distribution)
   ├─ Apply SMOTE to training set only (synthesize minority samples)
   ├─ Scale features (StandardScaler for LR; native for RF/XGBoost)
   └─ ⚠️ Test set never touched (no data leakage)

4. MODEL TRAINING
   ├─ Logistic Regression (baseline, interpretable)
   ├─ Random Forest (ensemble robustness)
   └─ XGBoost (gradient boosting, high performance)
   
   Selection: Best model by AUC-ROC on validation set

5. EXPLAINABILITY
   ├─ SHAP TreeExplainer (for ensemble models)
   ├─ SHAP LinearExplainer (for logistic regression)
   ├─ Summary plots (feature importance)
   └─ Waterfall plots (per-employee explanation)

6. FAIRNESS AUDIT
   ├─ Stratify predictions by Sex, Race, Ethnicity
   ├─ Compare false positive/negative rates (disparate impact)
   ├─ Flag >10% disparity threshold
   └─ Recommend mitigation if necessary

7. RISK REPORT GENERATION
   └─ Active employees only
      ├─ Filter: prediction probability > threshold (e.g., 15%)
      ├─ Sort: descending risk score
      ├─ Annotate: top 3 SHAP drivers for each employee
      └─ Export: CSV for dashboard ingestion
```

---

## 📚 User Story & Motivation

### **User Story**

> **As an** HR Manager at a growing technology company,  
> **I want** a transparent system that tells me which employees are likely to resign,  
> **So that** I can proactively intervene with targeted retention strategies before they leave.

#### **Acceptance Criteria**
1. ✅ **Dashboard displays all active employees with attrition risk > 15%** (clear visual priority)
2. ✅ **Clicking on an employee shows their top 3 risk factors** (e.g., "Low salary relative to department", "Recent negative feedback trend", "High absence rate")
3. ✅ **Model explains its reasoning transparently** (SHAP values show exact contribution of each factor)
4. ✅ **System is fair across demographics** (equal false positive rates for men/women, all ethnicities)
5. ✅ **System respects privacy** (no employee names in model, anonymized identifiers only)
6. ✅ **Reports are actionable** (interventions are suggested: salary review, career coaching, workload adjustment)

---

### **Motivation for Choosing This Approach**

#### **1. Business Context**
- **Employee turnover is expensive**: Replacing a mid-level employee costs 50–200% of annual salary (recruiting, onboarding, lost productivity)
- **Early warning matters**: Detecting risk 3–6 months early enables retention interventions
- **One-size-fits-all doesn't work**: Generic retention programs fail; targeted interventions succeed

#### **2. AI/ML Rationale**
- **Predictive models outperform human intuition**: Even experienced HR managers can't identify flight-risk employees consistently
- **Explainability is non-negotiable**: Decisions about employees must be transparent and defensible, not black-box
- **Fairness is legal & ethical**: GDPR, equal opportunity laws, and public trust require auditing for bias

#### **3. Technical Choices**

| Choice | Alternative | Why We Chose |
|--------|-------------|------------|
| **SHAP for Explainability** | LIME, feature importance, attention | SHAP is theoretically sound (Shapley values), handles feature dependencies, works with any model |
| **SMOTE for Balancing** | Class weights, undersampling, threshold tuning | SMOTE avoids throwing away data; class weights alone insufficient for 20% attrition rate |
| **Logistic Regression as Baseline** | Deep learning, simple heuristics | Interpretable coefficients, regulatory-friendly, strong baseline |
| **Fairness Audit Separate from Model** | Including protected attributes, mitigation in loss | Privacy-compliant; prevents proxy discrimination via indirect attributes |
| **Structured + NLP Features** | NLP only, structured only | Holistic signal; structured features for reliability, NLP for context (feedback themes) |
| **Privacy-by-Design** | Strip PII at end | Removes privacy risk at source; simpler to audit; aligns with GDPR principle of data minimization |

#### **4. Why This Matters**
- **Reduces turnover**: Targeted interventions (e.g., salary review) are more cost-effective than generic programs
- **Improves morale**: Employees see fair, transparent decision-making (not arbitrary "favoritism")
- **Protects the company**: Fairness audit & explainability shield against discrimination claims
- **Builds trust in AI**: Transparent, auditable decisions increase adoption of ML in HR

---

## 🛠️ Technical Documentation

### Prerequisites
Ensure you have Python 3.8+ and pip installed. All required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=1.3.0              # Data manipulation
numpy>=1.21.0              # Numerical computing
scikit-learn>=1.0.0        # Machine learning (LR, RF, preprocessing)
imbalanced-learn>=0.8.0    # SMOTE for class balancing
xgboost>=1.5.0             # Gradient boosting
shap>=0.41.0               # SHAP explainability
vaderSentiment>=3.3.2      # NLP sentiment analysis
matplotlib>=3.4.0          # Plotting
seaborn>=0.11.0            # Statistical visualization
jupyter>=1.0.0             # Notebook environment
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Baloo1506/HackathonA4.git
cd HackathonA4

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

#### **1. Main Pipeline Notebook**
```bash
jupyter notebook hr_attribution_7.ipynb
```

This notebook contains the complete end-to-end pipeline with 9 sections:
- **Section 0**: Imports & environment setup
- **Section 1**: Data loading & employment status filtering
- **Section 2**: Privacy-compliant data cleaning
- **Section 3**: Structured feature engineering (temporal, financial, behavioral)
- **Section 4**: NLP feature engineering (VADER sentiment, keyword extraction)
- **Section 5**: Exploratory data analysis (EDA)
- **Section 6**: Model training & evaluation (LR, RF, XGBoost)
- **Section 7**: SHAP explainability analysis
- **Section 8**: Fairness audit & disparate impact analysis
- **Section 9**: Risk report generation for active employees

**Execution Time**: ~5–10 minutes on standard hardware (depending on SHAP computation)

#### **2. Outputs**
The notebook generates:
- `risk_report_active_[timestamp].csv`: High-risk employees with top 3 SHAP drivers
- SHAP waterfall plots (per-employee explanations)
- Fairness audit tables (disparate impact by demographic)
- Model performance metrics (AUC, F1, confusion matrices)

---

## 📁 File Structure

```
HackathonA4/
├── README.md                              # This file (executive summary + architecture)
├── DATA_CARD.md                           # Dataset documentation
├── MODEL_CARD.md                          # Model documentation
├── ARCHITECTURE.md                        # System design & pipeline detail
├── INDEX.md                               # Documentation index & reading guide
├── hr_attribution_7.ipynb                 # Main analysis notebook (complete pipeline)
├── HRDataset_v14_enriched.csv            # Input HR dataset
├── HRDataset_v14.csv                      # Original unenriched dataset (reference)
├── risk_report_active_[timestamp].csv    # Output: high-risk employee report
├── shap_active_df.csv                     # SHAP values for active employees (reference)
├── requirements.txt                       # Python dependencies
└── claude.md                              # Internal notes (implementation plan)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
jupyter notebook hr_attribution_7.ipynb
# Execute cells sequentially from top to bottom
```

### 3. Review Outputs
- **Model Performance**: Section 6 (AUC, F1, confusion matrices)
- **Feature Importance**: Section 7 (SHAP plots)
- **Fairness Audit**: Section 8 (disparate impact analysis)
- **Risk Report**: Section 9 (exported CSV for dashboard)

### 4. Interpret Results
- **High-Risk Employees**: Probability > 15% (top 20% flagged)
- **Intervention Targets**: Prioritize by prediction confidence
- **Risk Drivers**: Check top 3 SHAP values for each employee

---

## 🔒 Privacy & Security

### Built-in Safeguards
✅ **Direct identifiers removed**: Employee_Name, EmpID, Manager info → never reach model  
✅ **Outcome leakage prevented**: TermReason, DateofTermination → excluded from features  
✅ **Protected attributes separated**: Sex, Race, Ethnicity → fairness audit only, no modeling  
✅ **Data minimization**: Only features with business + statistical value retained  
✅ **GDPR-compliant**: Right to explanation (SHAP) + right to erasure (no name storage)  

### Recommendations for Deployment
- Use only anonymized employee IDs in risk report
- Encrypt scores in transit (HTTPS + TLS)
- Audit access logs (who viewed which risk profiles)
- Retrain quarterly to detect fairness drift
- Disclose model use to employees (transparency)

---

## 📞 Support & Questions

**Repository**: [github.com/Baloo1506/HackathonA4](https://github.com/Baloo1506/HackathonA4)  
**Default Branch**: `main`  
**Current Branch**: `final-muhammadali` (development)  

For issues, questions, or contributions:
1. Check existing issues on GitHub
2. Open a new issue with detailed description
3. Submit pull requests with improvements

---

## 📜 License & Attribution

**Team**: GROUP N°17  
- GOETGHEBEUR Alexandre
- ALI Muhammad
- LEBRETON Gaspard
- POINTEAU Adrien
- TEPLOV Michael

**Hackathon**: A4 (Spring 2026)  

---

## 🙏 Acknowledgments

- Dataset provided by HR team (Jan 2019 snapshot)
- SHAP library: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- SMOTE technique: [Chawla et al., 2002](https://arxiv.org/abs/1106.1813)
- VADER sentiment: [Hutto & Gilbert, 2014](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
