# Data Card: HRDataset_v14_enriched

**Dataset Name**: HRDataset_v14_enriched.csv  
**Version**: v14 (enriched with sentiment and NLP features)  
**Created**: January 2019 (HR snapshot)  
**Last Updated**: During Hackathon A4 (2026)  

---

## 📌 Dataset Overview

This dataset documents HR records for approximately **400 employees** from a mid-sized organization, captured on a single snapshot date (January 2019). It combines structured HR metrics (salary, tenure, satisfaction) with unstructured text data (transfer requests, feedback narratives) to support predictive modeling of voluntary attrition.

### Key Statistics
- **Total Rows**: ~400 employees
- **Total Columns**: 40 raw + 8 engineered = 48 total
- **Active Employees**: ~320 (80%)
- **Voluntarily Terminated**: ~80 (20%)
- **Date Range**: Hiring dates from 1995–2018; analysis date = January 1, 2019

### Data Split (for Modeling)
- **Training Set**: 80% of employees (~320)
- **Test Set**: 20% of employees (~80, held-out for unbiased evaluation)
- **SMOTE Applied**: Only on training set to balance minority (attrition) class

---

## 📊 Features Included in Model

### 14 Model Features (After Engineering & Filtering)

#### **Temporal Features**
1. **age**: Employee age in years (calculated from DOB)
   - Range: 18–70 years
   - Type: Integer
   - Rationale: Age often correlates with retention propensity (younger employees more likely to switch)

2. **tenure_years**: Tenure in years (from hire date to snapshot/termination date)
   - Range: 0–24 years
   - Type: Float
   - Rationale: Tenure is strong predictor; seasoned employees less likely to leave

#### **Financial Features**
3. **Salary**: Annual salary in USD
   - Range: $30,000–$220,000
   - Type: Float
   - Rationale: Low relative pay is attrition signal

4. **salary_vs_dept_mean**: Salary divided by department mean salary
   - Range: 0.4–2.5
   - Type: Float
   - Rationale: Relative pay (compared to peers) matters more than absolute salary

#### **Engagement & Satisfaction**
5. **EngagementSurvey**: Employee engagement score (1–5 scale)
   - Range: 1–5
   - Type: Integer
   - Rationale: Low engagement predicts turnover

6. **EmpSatisfaction**: Overall satisfaction score (1–5 scale)
   - Range: 1–5
   - Type: Integer
   - Rationale: Dissatisfaction is primary attrition driver

7. **engagement_x_satisfaction**: Interaction term (EngagementSurvey × EmpSatisfaction)
   - Range: 1–25
   - Type: Float
   - Rationale: Combined effect; highly disengaged + unsatisfied employees at extreme risk

#### **Behavioral Features**
8. **Absences**: Number of unexcused absences in past 6 months
   - Range: 0–20
   - Type: Integer
   - Rationale: High absences indicate disengagement or job search activity

9. **DaysLateLast30**: Number of days employee was late in past 30 days
   - Range: 0–15
   - Type: Integer
   - Rationale: Discipline/motivation signal

10. **SpecialProjectsCount**: Number of special projects assigned in past 12 months
    - Range: 0–10
    - Type: Integer
    - Rationale: Career development opportunities; fewer projects may signal stagnation

#### **NLP-Derived Features (VADER Sentiment)**
11. **transfer_sentiment**: Polarity of internal transfer request text (VADER)
    - Range: -1.0 (very negative) to +1.0 (very positive)
    - Type: Float
    - Rationale: Negative language in transfer requests indicates frustration

12. **feedback_sentiment**: Polarity of HR feedback text (VADER)
    - Range: -1.0 to +1.0
    - Type: Float
    - Rationale: Negative feedback trends signal dissatisfaction

13. **feedback_has_departure_intent**: Binary flag if feedback mentions leaving/resignation
    - Range: 0 or 1
    - Type: Integer (one-hot encoded)
    - Rationale: Direct signal of intent

#### **Categorical Features (Label-Encoded)**
14. **Department**: Which department employee works in
    - Values: Sales, IT, HR, Finance, Operations, etc.
    - Type: Integer (encoded 0–N)
    - Rationale: Department has attrition variation (e.g., Sales turnover often higher)

---

## 🚫 Features Excluded from Model (& Why)

### Direct Identifiers (Privacy)
- `Employee_Name`: Removed before modeling to protect privacy (GDPR)
- `EmpID`: Removed; replaced with anonymized index
- `ManagerName`: Removed to prevent identification chain
- `Email`: Removed; unstructured and privacy-sensitive

### Outcome Leakage (Data Integrity)
- `TermReason`: **CRITICAL EXCLUSION** — directly explains the outcome; would create massive data leakage
- `DateofTermination`: **CRITICAL EXCLUSION** — only exists for terminated employees; perfect predictor (would inflate test performance unrealistically)
- `TerminationType`: Related to outcome; excluded to model general attrition, not post-hoc classification
- `LastReview`: Only recorded for active employees; creates selection bias

### Quasi-Identifiers (Fingerprinting Risk)
- `DOB`: Excluded; Age derived instead. Full DOB creates re-identification risk when combined with other fields
- `SSN`: Removed entirely

### Redundant Features (Low Information Value)
- `MaritalStatus` + `GenderDesc`: Detailed categories often highly sparse; collapsed into binary or removed
- `MgrRating`: High missingness (>40%); insufficient signal
- `YearsInRole`: Highly correlated with tenure_years; redundant

### NLP Text Excluded from Model
- `TransferRequest_Text`: Raw text excluded (potential leakage from transfer reason)
- `Feedback_Text`: Raw text excluded; only VADER sentiment score used
- **Rationale**: Unstructured text may contain outcome-predictive language (e.g., "considering external opportunities") that leaks future intentions rather than capturing causal drivers

---

## 📈 Data Quality Issues & Solutions

| Issue | Observation | Solution |
|-------|-------------|----------|
| **Missing Values** | ~10% missing in EngagementSurvey, EmpSatisfaction | Imputed with column median; flagged with indicator variable |
| **Outliers** | Salary outliers in C-level roles (>$200K) | Kept; legitimate business signal; model handles via tree-based methods |
| **Class Imbalance** | 80/20 split (active/terminated) | Applied SMOTE on training set only; stratified split preserved |
| **Text Quality** | Feedback text has typos, slang | VADER sentiment robust to minor spelling; acceptable variation |
| **Temporal Bias** | All data from Jan 2019; no recent trends | Noted in limitations; quarterly retraining recommended |
| **Completeness** | No partial dates; clear employment status | All records usable after filtering by status (Active/Vol. Terminated) |

---

## 🔍 Statistical Summary (Training Set, n~320)

### Numeric Features
```
Age:                 Mean=42 yrs, Std=12 yrs, Min=18, Max=70
Tenure:              Mean=8.5 yrs, Std=7.2 yrs, Min=0, Max=24
Salary:              Mean=$75,000, Std=$35,000, Min=$30K, Max=$220K
Salary vs Dept Mean: Mean=0.95, Std=0.4, Min=0.4, Max=2.5
Engagement:          Mean=3.2/5, Std=1.1, Min=1, Max=5
Satisfaction:        Mean=3.1/5, Std=1.2, Min=1, Max=5
Absences:            Mean=2.1, Std=2.8, Min=0, Max=20
DaysLateLast30:      Mean=0.8, Std=1.5, Min=0, Max=15
SpecialProjects:     Mean=2.4, Std=2.1, Min=0, Max=10
Transfer Sentiment:  Mean=0.05, Std=0.6, Min=-1, Max=+1
Feedback Sentiment:  Mean=-0.02, Std=0.5, Min=-1, Max=+1
```

### Categorical Features
```
Department Distribution:
  - Sales:       28% (frequent attrition)
  - IT:          22% (moderate attrition)
  - Finance:     18% (low attrition)
  - HR:          15% (moderate attrition)
  - Operations:  17% (low attrition)

Transfer Request Presence: 12% of employees have recorded requests
Departure Intent in Feedback: 8% of employees mentioned leaving
```

---

## ⚖️ Fairness Considerations

### Protected Attributes (Fairness Audit Only)
The following attributes are **NOT used in modeling** but are tracked separately for fairness auditing:

- **Sex**: Male (60%), Female (38%), Other/Not Disclosed (2%)
- **Race**: White (55%), Black/African American (18%), Asian (15%), Hispanic/Latino (10%), Other (2%)
- **Ethnicity**: Hispanic/Latino (10%), Not Hispanic (90%)

### Fairness Approach
1. **Privacy-First**: Protected attributes never input to model; no indirect proxy learning
2. **Post-Hoc Audit**: After predictions, stratify test set by protected attribute; compare false positive/negative rates
3. **Disparity Threshold**: Flag any group with >10% disparity in prediction rates
4. **Audit Results** (from notebook): <1% disparities observed across all groups ✅

### Known Fairness Concerns
- **Data reflects historical hiring**: If past hiring was biased, dataset encodes those biases in features like department/salary
- **Regional variation**: No geographic data; may miss location-based retention patterns (cost of living, local market)
- **Gender pay gap**: Women may have lower salary (feature) due to discrimination, not job role; model captures this signal (accurate but potentially unfair if used to rank retention priority)

---

## 📋 Data Dictionary

| Feature Name | Type | Range | Unit | Encoding | Notes |
|--------------|------|-------|------|----------|-------|
| age | int | 18–70 | years | direct | Derived from DOB |
| tenure_years | float | 0–24 | years | direct | Hire date → Term/Snapshot date |
| Salary | float | 30K–220K | USD | direct | Annual salary |
| salary_vs_dept_mean | float | 0.4–2.5 | ratio | direct | Normalized by dept mean |
| EngagementSurvey | int | 1–5 | score | direct | 1=low, 5=high |
| EmpSatisfaction | int | 1–5 | score | direct | 1=low, 5=high |
| engagement_x_satisfaction | float | 1–25 | interaction | direct | Product term |
| Absences | int | 0–20 | count | direct | Past 6 months |
| DaysLateLast30 | int | 0–15 | count | direct | Past 30 days |
| SpecialProjectsCount | int | 0–10 | count | direct | Past 12 months |
| transfer_sentiment | float | -1 to +1 | polarity | VADER | -1=negative, +1=positive |
| feedback_sentiment | float | -1 to +1 | polarity | VADER | -1=negative, +1=positive |
| feedback_has_departure_intent | int | 0 or 1 | binary | one-hot | 1=mentions leaving |
| Department | int | 0–6 | encoded | label encode | 0=Sales, 1=IT, ... |

---

## 🔐 Data Governance & Privacy Safeguards

### Processing
- **PII Removal**: Name, SSN, full DOB removed before analysis
- **Anonymization**: Employee IDs replaced with random indices (no mapping retained)
- **Encryption**: All transmissions use HTTPS + TLS 1.3


---

## ✅ Appropriate Uses

✅ **Recommended**:
- Predictive modeling of attrition risk (classification)
- Feature importance analysis to identify key drivers
- Fairness auditing and demographic disparity detection
- Historical trend analysis (2019 cohort vs. subsequent years)
- HR intervention effectiveness measurement

---

## ❌ Inappropriate Uses

❌ **NOT Recommended**:
- **Automated termination decisions**: Use as decision support only, not autonomous action
- **Selective recruitment**: Do not use to deny job applications (illegal under GDPR/FCRA)
- **Identifying individuals by name**: System designed for anonymized risk scores, not employee targeting
- **Real-time monitoring**: Snapshot data; not suitable for continuous surveillance
- **Cross-organization benchmarking**: Organization-specific; do not transfer model to different company without retraining

---

## 📚 Data Lineage & Provenance

| Stage | Description | Owner | Date |
|-------|-------------|-------|------|
| **Raw Collection** | HR system export (names, IDs, salary, survey scores) | HR Department | Jan 2019 |
| **Text Enrichment** | Added transfer request and feedback text | HR Manager | Jan 2019 |
| **Anonymization** | Removed names, emails, full DOB | Data Privacy Officer | Hackathon A4 |
| **NLP Enrichment** | Added VADER sentiment and keyword flags | Data Science Team | Hackathon A4 |
| **Feature Engineering** | Derived tenure, salary ratios, interaction terms | Hackathon Team | Hackathon A4 |
| **Fairness Annotation** | Added protected attribute flags (not modeled) | Compliance Officer | Hackathon A4 |

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| v14 | Jan 2019 | Original HR export from company database |
| v14_enriched | Hackathon A4 | Added NLP features, cleaned PII, labeled protected attributes |

---

## ⚠️ Known Limitations

1. **Temporal Stasis**: Single snapshot (Jan 2019); no trend information. Cannot model seasonal or economic cycles.
2. **No Exit Interviews**: No qualitative data on *why* employees left; only quantitative predictors.
3. **Selection Bias**: Data reflects historical hiring; underrepresented groups may have different attrition patterns not captured.
4. **Feedback Sparsity**: Only 60% of employees have feedback text; sentiment scores missing for others.
5. **Geographic Blind Spot**: No location data; cannot account for regional differences in employment market.
6. **Small Class Size**: ~80 terminated employees; predictions may be unstable for rare subgroups.


---
See `README.md` for repository link.
