# HackathonA4

GROUP N°17 : GOETGHEBEUR Alexandre, ALI Muhammad, LEBRETON Gaspard, POINTEAU Adrien, TEPLOV Michael  


# Trusted HR-AI: Employee Retention Dashboard

## Project Overview
This project is the backend intelligence for an interactive HR web application designed to proactively manage employee turnover. The platform features a dashboard that filters and displays active employees whose predicted probability of resigning exceeds a critical threshold. 

By clicking on a specific high-risk employee, HR managers can access a detailed "Risk Profile" explaining exactly *why* that person is predicted to leave, empowering them to take targeted, fair, and secure preventive actions.

---

##  Key Features & User Experience
* **High-Risk Employee Dashboard**: A centralized view displaying all active employees at risk of leaving.
* **Individual Drill-Down (Explainability)**: Click on any employee to view their personalized risk drivers (e.g., Low Salary, Poor Sentiment, High Absences) powered by SHAP values.
* **Secure & Anonymous**: Built with Privacy-by-Design. All Personally Identifiable Information (PII) like names are removed before modeling to ensure GDPR compliance.
* **Ethical & Fair**: The model is rigorously audited for demographic parity to ensure risk scores are not biased against specific genders or ethnicities.

---

## Technical Pipeline & Scope
Our pipeline processes a real-world HR dataset and transforms it into actionable, ethical insights for the frontend dashboard:

1. **Cybersecurity & Data Privacy**: 
   * Direct identifiers (`Employee_Name`) are stripped out.
   * Only anonymized `EmpID`s are passed to the frontend to protect employee privacy.
2. **Feature Engineering & NLP (VADER)**: 
   * We calculate employee tenure and extract qualitative insights from unstructured text (Internal Transfer Requests and HR Feedback).
   * `vaderSentiment` is used to calculate sentiment polarity scores, translating textual feedback into numerical risk signals.
3. **Data Augmentation (SMOTE)**: 
   * Because the dataset is small and imbalanced, we use **SMOTE** (Synthetic Minority Over-sampling Technique) via `imblearn` on the training set. This synthesizes minority-class samples (resignations) to train a robust model without exposing real data.
4. **Predictive Modeling**: 
   * We train robust classifiers (Logistic Regression, Random Forest, XGBoost) and apply probability calibration to generate an accurate risk percentage (0-100%) for the web interface.
5. **Explainable AI (SHAP)**: 
   * We integrate `SHAP` (SHapley Additive exPlanations). For every prediction, SHAP calculates the exact contribution of each feature, allowing the web app to display local waterfall plots and top risk factors for any specific user.
6. **Ethics & Fairness Audit**: 
   * A disparate impact analysis is conducted on sensitive attributes (`Sex`, `RaceDesc`, `HispanicLatino`) to ensure the model's predictions remain equitable and unbiased across all demographics.

---

## Personae
* **The Client (HR Manager)**: "I need a secure, user-friendly site where I can see *who* is at risk of leaving, click on their profile to understand *why*, and intervene before it's too late."
* **The Solution Provider (HR-AI Company)**: "We deliver a web-based AI dashboard that empowers HR teams with transparent, ethical, and highly accurate retention insights."

---

## Instructions

### Prerequisites
Ensure you have Python installed along with the required backend libraries:
```bash
pip install pandas numpy scikit-learn imbalanced-learn shap vaderSentiment xgboost