# HackathonA4

GROUP N°17 : GOETGHEBEUR Alexandre, ALI Muhammad, LEBRETON Gaspard, POINTEAU Adrien, TEPLOV Michael  

## Product : ##
- AI solution that helps HR to manage talents
- To help people discover four major themes of AI through a problem 


# Trusted HR-AI: Employee Retention & Turnover Analysis

## Project Overview
This project addresses the challenge of an imaginary company facing a high rate of resignations. We utilize an AI-driven approach to understand turnover causes and help preserve talent. Our solution is built on the pillars of **Trusted AI**, specifically focusing on **Cybersecurity** and **Ethics**.

---

## Objectives
The main goal is to design an AI solution to help HR management identify resignation factors and suggest preventive actions.
* **Predictive Risk Assessment**: Predict which employees are at risk of leaving the company.
* **Root Cause Analysis**: Explain why employees are likely to quit based on structured data and text analysis.
* **Cybersecurity Compliance**: Ensure data protection and GDPR compliance through anonymization of sensitive staff data.
* **Ethical Auditing**: Avoid algorithmic bias and discrimination to ensure a fair model.

---

## Scope & Challenge
We process an open-source HR dataset containing information on approximately 400 employees, including age, position, salary, and performance metrics.
* **Sensitive Attributes**: The data includes gender (`Sex`) and ethnicity (`RaceDesc`), requiring careful ethical handling.
* **Data Enrichment**: We integrate unstructured textual data, such as internal transfer requests and anonymized exit interview feedback, to capture qualitative context.
* **Cyber-Hygiene**: As a primary security measure, the dataset is anonymized by removing or masking sensitive identifiers to remain legally compliant.
* **Fairness Audit**: We audit predictions to check if the model treats all employees fairly or shows signs of discrimination.

---

## Personae
* **The Client (HR Manager)**: "I need a solution to retain my employees and understand the real reasons behind their departures."
* **The Solution Provider (HR-AI Company)**: "We bring responsible AI solutions that help HR identify risks while protecting data and ensuring non-discrimination."

---

##  Instructions

### 1. Data Preparation (Cybersecurity Focus)
* **Anonymization**: Before training, identifiers like names must be removed or codified.
* **Synthetic Expansion**: To address the modest size of the dataset, use **SDV (Synthetic Data Vault)** to generate synthetic records mirroring real statistics, which enhances privacy.

### 2. Feature Engineering (NLP Focus)
* Analyze textual feedback using NLP (sentiment analysis, theme extraction) to provide qualitative insights into resignations.
* Use these insights to complement structured numerical data like satisfaction scores and absences.

### 3. Model Training & Validation
* Train a predictive model (e.g., Random Forest or XGBoost) to predict the `Termd` status.
* Evaluate the model using the **AI Act** risk pyramid, ensuring transparency and measuring the risk level.

### 4. Fairness & Transparency (Ethics Focus)
* **Bias Check**: Use tools like `AIF360` to audit the model for unfair biases towards specific gender or ethnic groups.
* **Explainability**: Integrate XAI tools (like SHAP or LIME) so HR managers can understand the *why* behind a high-risk prediction.

---

##  Deliverables
* **Model Card**: Explanation of the chosen architecture and logic.
* **Data Card**: Documentation of the processing and anonymization steps.
* **Live Demo**: Presentation of the risk-scoring dashboard.
