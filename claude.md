# HR Attrition AI — Implementation Plan

## Context

The project goal is to predict which active employees are at risk of voluntary resignation, explain why, and suggest preventive HR actions. The dataset is `HRDataset_v14_enriched.csv` (~400 employees, 40 columns including two text fields). The solution must respect ethical constraints (no Sex/RaceDesc/HispanicLatino as features), privacy rules (remove identifiers and leakage fields), and be fully explainable.

The final deliverable is a single reproducible Jupyter notebook + a risk report CSV for active employees.

---

## Critical Files

- `c:\Users\mtepl\Documents\A4-S2\Explainability AI\HRDataset_v14_enriched.csv` — main dataset
- `c:\Users\mtepl\Documents\A4-S2\Explainability AI\ai.md` — project scope and rules

---

## Notebook Structure

### Section 0 — Setup & Imports
- Libraries: pandas, numpy, sklearn, xgboost/lightgbm, shap, vaderSentiment, matplotlib, seaborn

---

### Section 1 — Data Loading & Filtering

1. Load `HRDataset_v14_enriched.csv`
2. **Filter rows**: keep only `EmploymentStatus` in `["Active", "Voluntarily Terminated"]` — drop "Terminated for Cause"
3. Create binary target: `target = 1` if `Voluntarily Terminated`, else `0`

---

### Section 2 — Privacy & Data Cleaning

Remove direct identifiers:
- Drop: `Employee_Name`, `EmpID`, `ManagerName`, `ManagerID`

Remove outcome leakage:
- Drop: `Termd`, `DateofTermination`, `TermReason`, `EmploymentStatus`, `EmpStatusID`

Keep for fairness audit only (separate df, not in model):
- `Sex`, `RaceDesc`, `HispanicLatino`

Drop/generalize quasi-identifiers:
- `DOB` → compute `age = current_year - birth_year`; drop `DOB`
- `DateofHire` → compute `tenure_years`; drop `DateofHire`
- Drop `Zip`, `State` (too granular, low business value)
- Drop `LastPerformanceReview_Date` → compute `days_since_review`; drop original date

---

### Section 3 — Feature Engineering (Structured)

New engineered features to add to the dataset:

| Feature | Formula |
|---|---|
| `age` | year(today) − year(DOB) |
| `tenure_years` | (today − DateofHire) / 365 |
| `days_since_review` | (today − LastPerformanceReview_Date).days |
| `salary_vs_dept_mean` | Salary / dept_mean_salary — relative pay signal |
| `engagement_x_satisfaction` | EngagementSurvey × EmpSatisfaction — composite engagement score |

Drop: `MarriedID`, `MaritalStatusID`, `GenderID`, `DeptID`, `PositionID`, `PerfScoreID`, `FromDiversityJobFairID` — replaced by their text equivalents or irrelevant IDs.

Keep: `Department`, `Position`, `MaritalDesc`, `CitizenDesc`, `RecruitmentSource`, `PerformanceScore`, `EngagementSurvey`, `EmpSatisfaction`, `SpecialProjectsCount`, `DaysLateLast30`, `Absences`, `Salary`, `age`, `tenure_years`, `days_since_review`, `salary_vs_dept_mean`, `engagement_x_satisfaction`

Encode categoricals: one-hot or label encoding for `Department`, `Position`, `MaritalDesc`, `CitizenDesc`, `RecruitmentSource`, `PerformanceScore`

---

### Section 4 — NLP Feature Engineering (Text)

Apply to both `Internal_Transfer_Request` and `Feedback_RH`.

#### 4a. Transfer Request Signal
- `has_transfer_request`: binary flag — 1 if `Internal_Transfer_Request` is non-null/non-empty
- `transfer_request_sentiment`: VADER compound score on request text

#### 4b. Feedback Sentiment
- `feedback_sentiment`: VADER compound score on `Feedback_RH` (-1 to +1)
- `feedback_has_compensation`: keyword flag (salary, pay, compensation, underpaid)
- `feedback_has_growth`: keyword flag (career, promotion, growth, development, opportunity)
- `feedback_has_workload`: keyword flag (burnout, overload, hours, stress, schedule)
- `feedback_has_management`: keyword flag (manager, leadership, unfair, micromanage)
- `feedback_has_departure_intent`: keyword flag (leaving, quit, resign, move on, other opportunity)

These NLP features are added to the modeling dataset as numeric columns.

> **No raw text is used directly in the model** — only derived numeric signals. Raw text is masked/aggregated in any output.

---

### Section 5 — EDA

- Class distribution (target balance)
- Correlation heatmap of numeric features
- Attrition rate by Department, Position, PerformanceScore
- Distribution of EngagementSurvey and EmpSatisfaction by target
- Sentiment distribution for departed vs. active employees
- Top transfer request themes (word frequency, active vs. terminated)

---

### Section 6 — Modeling

#### 6a. Train/Test Split
- 80/20 stratified split on `target`

#### 6b. Models to Compare
1. **Logistic Regression** — baseline, fully interpretable coefficients
2. **Random Forest** — handles non-linearity, provides feature importance
3. **XGBoost** — best performance candidate

All models use `class_weight='balanced'` (or `scale_pos_weight` for XGBoost) to handle class imbalance.

#### 6c. Evaluation Metrics
- AUC-ROC (primary)
- F1-score, Precision, Recall (on test set)
- Confusion matrix
- Precision-Recall curve

#### 6d. Hyperparameter Tuning
- `GridSearchCV` or `RandomizedSearchCV` on best model only (to stay frugal)

---

### Section 7 — Explainability (SHAP)

Using the best performing model:

- **Global**: SHAP summary plot (top 15 features driving attrition risk)
- **Global**: SHAP bar plot (mean absolute SHAP values)
- **Local**: SHAP waterfall/force plots for 2-3 example employees (one high-risk, one low-risk)
- **Department-level**: SHAP values aggregated per department

This section answers: *why does the model predict high risk for a given employee?*

---

### Section 8 — Fairness Audit

Using the separate sensitive-attribute dataframe (not in model):

- Compute attrition rate by `Sex`, `RaceDesc`, `HispanicLatino` group
- Compute model AUC and false positive rate by group
- Check for **disparate impact**: flag if any group's predicted positive rate < 80% of highest group
- Report: fairness table + brief interpretation

---

### Section 9 — Risk Report for Active Employees

1. Filter active employees (before any split — re-predict on all actives)
2. Add risk score (model probability) and risk level: Low (<0.3), Medium (0.3–0.6), High (>0.6)
3. For each active employee, compute top 3 SHAP contributors
4. Map top SHAP features to **retention actions**:

| SHAP signal | Recommended action |
|---|---|
| Low EmpSatisfaction | HR check-in / satisfaction interview |
| Low EngagementSurvey | Manager discussion / project reassignment |
| has_transfer_request = 1 | Internal mobility conversation |
| salary_vs_dept_mean < 1 | Compensation review |
| feedback_has_growth = 1 | Development plan / promotion track |
| feedback_has_workload = 1 | Workload audit / staffing review |
| feedback_has_management = 1 | Manager coaching or team reassignment |
| High Absences or DaysLateLast30 | Wellbeing or flexibility discussion |

5. Output: `risk_report_active.csv` with columns: `[anonymized_id, Department, Position, risk_score, risk_level, top_reason_1, top_reason_2, top_reason_3, recommended_action]`

> Raw text and personal identifiers are **not** included in the output CSV.

---

### Section 10 — Ethics & Cyber Summary

Short markdown cells summarizing:
- Which fields were removed and why
- Which fields are excluded from model features (protected attributes)
- Fairness audit results summary
- Frugality note (no external API calls, no LLM, lightweight local NLP)

---

## Extra Data Added to Dataset

The plan enriches the dataset with the following **computed** columns (all derived from existing fields, no external data):
- `age`, `tenure_years`, `days_since_review`, `salary_vs_dept_mean`, `engagement_x_satisfaction`
- NLP features from text: `has_transfer_request`, `transfer_request_sentiment`, `feedback_sentiment`, 4× keyword flags

These add behavioral and contextual signals without introducing new privacy risk.

---

## Verification

1. Run notebook top-to-bottom: no errors, all cells produce output
2. Check `risk_report_active.csv` is generated: contains only active employees, no raw PII, valid risk scores
3. Fairness audit section: at least one table comparing metrics across groups
4. SHAP plots render correctly (at least summary plot)
5. Model comparison section shows ≥2 models evaluated with AUC scores
