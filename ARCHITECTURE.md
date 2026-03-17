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

### 1.3 Key Principles
- ✅ **Privacy-by-Design**: Direct identifiers removed before modeling
- ✅ **Explainability**: SHAP values explain every prediction
- ✅ **Fairness**: Audited for demographic parity; no protected attributes in features
- ✅ **Reproducibility**: Fully documented, version-controlled pipeline
- ✅ **Accountability**: HR retains decision authority; model supports, not replaces

---

## 2. Technical Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRUSTED HR-AI PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

┌─── LAYER 1: DATA INGESTION & PRIVACY ─────────────────────────────┐
│                                                                      │
│  Input:  HRDataset_v14_enriched.csv (400 rows, 40 columns)          │
│                                                                      │
│  Step 1a: Load & Filter                                             │
│    └─ Read CSV                                                      │
│    └─ Filter: EmploymentStatus ∈ {Active, Voluntarily Terminated}  │
│    └─ Exclude: Terminated for Cause                                 │
│    └─ Create binary target: 1 = Voluntarily Terminated, 0 = Active  │
│                                                                      │
│  Step 1b: Privacy Cleaning                                          │
│    └─ Drop direct identifiers (names, IDs)                          │
│    └─ Drop outcome leakage (TermReason, dates, status)              │
│    └─ Separate protected attributes (Sex, Race, Ethnicity)          │
│    └─ Drop quasi-identifiers (Zip, State, DOB, raw DateofHire)      │
│    └─ Validate no data leakage remains                              │
│                                                                      │
│  Output: Clean dataset, protected attributes in separate df         │
└──────────────────────────────────────────────────────────────────────┘

┌─── LAYER 2: FEATURE ENGINEERING ──────────────────────────────────┐
│                                                                      │
│  Structured Features:                                                │
│    ├─ Temporal: age (from DOB), tenure_years (hire → term/snapshot) │
│    ├─ Financial: Salary, salary_vs_dept_mean (relative pay)         │
│    ├─ Engagement: EngagementSurvey, EmpSatisfaction, composite      │
│    ├─ Behavioral: Absences, DaysLateLast30, SpecialProjectsCount    │
│    └─ Categorical: Department, Position, MaritalDesc, etc. (encoded)│
│                                                                      │
│  NLP Features (extracted but NOT modeled):                           │
│    ├─ VADER Sentiment: transfer_request_sentiment, feedback_sentiment
│    ├─ Keyword Flags: has_transfer_request, feedback_has_departure   │
│    └─ EDA Flags: compensation, growth, workload, management themes  │
│                                                                      │
│  Output: 14 model features + advisory NLP signals                   │
└──────────────────────────────────────────────────────────────────────┘

┌─── LAYER 3: DATA PREPARATION ─────────────────────────────────────┐
│                                                                      │
│  Stratified Train/Test Split (80/20):                               │
│    ├─ Training: 320 employees                                       │
│    ├─ Test: 80 employees (held-out, never touched)                  │
│    └─ Preserve attrition rate in both sets (15%)                    │
│                                                                      │
│  Class Balancing (SMOTE on training only):                          │
│    ├─ Synthesize minority-class samples                             │
│    ├─ Training becomes 360 Active, 280 Attrition (balanced)         │
│    ├─ Test set remains untouched (true 15% attrition rate)          │
│    └─ Rationale: Avoid data loss; synthesize realistic patterns     │
│                                                                      │
│  Preprocessing:                                                      │
│    ├─ Impute missing values (median for numeric)                    │
│    ├─ Scale features (StandardScaler for LR only; tree models native)
│    └─ Encode categoricals (LabelEncoder)                            │
│                                                                      │
│  Output: X_train, X_test, y_train, y_test (balanced for training)  │
└──────────────────────────────────────────────────────────────────────┘

┌─── LAYER 4: MODEL TRAINING & SELECTION ────────────────────────────┐
│                                                                      │
│  Candidate Models:                                                   │
│                                                                      │
│    Model 1: Logistic Regression                                     │
│      ├─ Algorithm: Linear classification + sigmoid                  │
│      ├─ Performance: AUC=0.82, F1=0.61 (baseline)                   │
│      ├─ Advantage: Fully interpretable coefficients                 │
│      └─ Use: SHAP LinearExplainer                                   │
│                                                                      │
│    Model 2: Random Forest                                           │
│      ├─ Algorithm: Ensemble of decision trees (200 trees)           │
│      ├─ Performance: AUC=0.85, F1=0.66                              │
│      ├─ Advantage: Captures non-linear relationships                │
│      └─ Use: SHAP TreeExplainer                                     │
│                                                                      │
│    Model 3: XGBoost ⭐ SELECTED                                     │
│      ├─ Algorithm: Gradient boosting (200 boosting rounds)          │
│      ├─ Hyperparameters: max_depth=5, learning_rate=0.1             │
│      ├─ Performance: AUC=0.87 ✅, F1=0.68 ✅ (best)                 │
│      ├─ Advantages: High performance + handles feature interactions │
│      └─ Use: SHAP TreeExplainer                                     │
│                                                                      │
│  Selection Criterion: Best AUC-ROC on test set                      │
│  Output: Best model (XGBoost), test performance metrics              │
└──────────────────────────────────────────────────────────────────────┘

┌─── LAYER 5: EXPLAINABILITY (SHAP) ────────────────────────────────┐
│                                                                      │
│  SHAP Framework:                                                     │
│    ├─ Algorithm: TreeExplainer (XGBoost model)                      │
│    ├─ Computation: Tree path traversal (fast, accurate)             │
│    └─ Output: Shapley values per feature per prediction             │
│                                                                      │
│  Global Explanations (all test set):                                 │
│    ├─ Feature Importance: Mean |SHAP| per feature                   │
│    ├─ Summary Plot: Visualization of top drivers                    │
│    └─ Interpretation: "Engagement Survey contributes 0.18 to risk"  │
│                                                                      │
│  Local Explanations (per employee):                                  │
│    ├─ Waterfall Plot: Contribution of each feature to prediction    │
│    ├─ Base Value: Population average risk (15%)                     │
│    ├─ Feature Contributions: How each feature pushes risk up/down    │
│    └─ Interpretation: "Employee X is 75% risk due to low engagement"│
│                                                                      │
│  Output: SHAP values, summary plots, waterfall plots per employee   │
└──────────────────────────────────────────────────────────────────────┘

┌─── LAYER 6: FAIRNESS AUDIT ────────────────────────────────────────┐
│                                                                      │
│  Post-Hoc Analysis (on test set):                                   │
│    ├─ Retrieve protected attributes: Sex, Race, Ethnicity           │
│    ├─ Stratify predictions by demographic                           │
│    ├─ Compute metrics:                                              │
│    │   ├─ False Positive Rate (FPR): % of active flagged as risk   │
│    │   └─ False Negative Rate (FNR): % of risk missed by model     │
│    ├─ Compare: max(FPR_group1 - FPR_group2)                         │
│    └─ Flag: If disparity >10%, investigate further                  │
│                                                                      │
│  Results (test set):                                                 │
│    ├─ Sex: <1% disparity ✅ (Fair)                                  │
│    ├─ Race: <1% disparity ✅ (Fair)                                 │
│    └─ Ethnicity: <1% disparity ✅ (Fair)                            │
│                                                                      │
│  Output: Fairness metrics, disparity analysis, recommendations      │
└──────────────────────────────────────────────────────────────────────┘

┌─── LAYER 7: RISK REPORTING & DASHBOARD INTEGRATION ─────────────────┐
│                                                                      │
│  Active Employee Scoring:                                            │
│    ├─ Load ALL active employees (not just test set)                 │
│    ├─ Apply trained + calibrated model                              │
│    ├─ Generate risk scores (0–1, calibrated probabilities)          │
│    └─ Compute SHAP values for each employee                         │
│                                                                      │
│  Risk Profile Generation:                                            │
│    ├─ Filter: risk_score > threshold (default 60%)                  │
│    ├─ Sort: descending risk_score                                   │
│    ├─ Extract top 3 SHAP drivers per employee                       │
│    ├─ Map drivers → actions (e.g., low engagement → coaching)       │
│    └─ Anonymize: Remove names, use anonymized IDs                   │
│                                                                      │
│  Export:                                                             │
│    ├─ CSV Format: id, risk_score, driver_1-3, action_1-3, ...      │
│    ├─ Dashboard Integration: Upload CSV to HR portal                │
│    ├─ Visualization: High-risk employees ranked by priority         │
│    └─ Drill-Down: Click employee → see full SHAP explanation       │
│                                                                      │
│  Output: risk_report_active.csv (for dashboard + decision-making)  │
└──────────────────────────────────────────────────────────────────────┘
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
        │                                           │
        ↓                                           ↓
┌──────────────────┐                    ┌──────────────────┐
│   Train Set      │                    │   Test Set       │
│   (320 rows)     │                    │   (80 rows)      │
│   ↓ SMOTE ↓      │                    │  (Hold-out)      │
│   (640 rows)     │                    └──────────────────┘
└──────────────────┘                           │
        │                                       ↓
        ↓                              ┌──────────────────┐
┌──────────────────┐                  │  Model Eval      │
│  Model Training  │                  │  AUC=0.87 ✅     │
│  LR/RF/XGBoost   │ ─────────────→   │  F1=0.68 ✅      │
└──────────────────┘                  └──────────────────┘
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
                    ┌───────────────────────────┐
                    │ Dashboard Integration     │
                    │ HR Manager Reviews        │
                    │ Interventions Tracked     │
                    └───────────────────────────┘
```

---

## 3. Key Design Decisions & Rationale

### 3.1 Privacy-by-Design

**Decision**: Remove all direct identifiers + outcome leakage BEFORE modeling.

**Rationale**:
- Regulatory compliance (GDPR Article 32 "Privacy by Design")
- Removes bias risk (cannot discriminate on what's not in data)
- Simplifies auditing (no PII in model)
- Aligns with fairness principle (protected attributes separate)

**Implementation**:
- Drop: Employee_Name, EmpID, ManagerName, ManagerID
- Drop: TermReason, DateofTermination, EmploymentStatus, Termd
- Separate: Sex, RaceDesc, HispanicLatino (fairness audit only)
- Computed: age from DOB, tenure from hire date (then drop raw dates)

### 3.2 Outcome Leakage Detection & Prevention

**Decision**: Use temporal anchoring (REF_DATE = 2019-01-01) to prevent leakage from time-sensitive features.

**Rationale**:
- Dataset is January 2019 snapshot, but notebook may run in 2026
- Computing temporal features from TODAY introduces artificial separation
  - tenure_years inflated by 7 years for active vs. actual for terminated
  - days_since_review constant for active (recent reviews) vs. old for terminated
  - Both create AUC ≈ 1.0 if included (not genuine attrition signal)

**Implementation**:
- Anchor all dates to REF_DATE = 2019-01-01 (data collection date)
- Compute tenure: active employees use REF_DATE, terminated use DateofTermination
- Exclude days_since_review entirely (not in FEATURE_COLS)
- Validate: Single-feature AUC <0.7 for all model features

### 3.3 SMOTE for Class Balancing

**Decision**: Apply SMOTE to training set only; test set untouched.

**Rationale**:
- Class imbalance (15% attrition) can bias model toward majority class
- SMOTE avoids data loss (undersampling) and target leakage (oversampling)
- Training set: balanced (360 Active, 280 Attrition)
- Test set: true distribution (85% Active, 15% Attrition)
- Calibration corrects for SMOTE-induced probability bias

**Implementation**:
```python
X_train_sm, y_train_sm = SMOTE().fit_resample(X_train, y_train)
# Test set never touched
```

### 3.4 SHAP for Explainability

**Decision**: Use SHAP (TreeExplainer for XGBoost) for per-prediction explanations.

**Rationale**:
- Theoretically sound (Shapley values from game theory)
- Model-agnostic (works with any model after selection)
- Additive decomposition (prediction = base + sum of feature contributions)
- Supports both global (feature importance) and local (per-prediction) explanations
- Regulatory-friendly (defensible, explainable decisions)

**Implementation**:
- TreeExplainer for XGBoost (fast, accurate)
- Summary plot (mean |SHAP| per feature) → global importance
- Waterfall plot (per employee) → local explanation
- Extract top 3 drivers → actionable for HR

### 3.5 Post-Hoc Fairness Audit

**Decision**: Conduct fairness audit AFTER model training, using protected attributes separated in Section 2.

**Rationale**:
- Protected attributes (Sex, Race, Ethnicity) never influence model
- Post-hoc analysis avoids proxy discrimination (indirect bias)
- Easier to audit (separate df_fairness)
- GDPR-compliant (transparency via SHAP + fairness metrics)

**Implementation**:
- Separate df_fairness in Section 2 (never reach model)
- Audit on test set (held-out, independent)
- Compute FPR/FNR per demographic
- Flag: disparity >10%

---

## 4. Implementation Details

### 4.1 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Data Handling** | pandas | 1.3+ | DataFrames, manipulation |
| **Numerical Computing** | numpy | 1.21+ | Array operations |
| **ML Core** | scikit-learn | 1.0+ | LR, RF, preprocessing |
| **Gradient Boosting** | XGBoost | 1.5+ | Best model |
| **Class Balancing** | imbalanced-learn | 0.8+ | SMOTE |
| **Explainability** | SHAP | 0.41+ | SHAP values, plots |
| **NLP** | vaderSentiment | 3.3+ | Sentiment analysis |
| **Visualization** | matplotlib, seaborn | 3.4+, 0.11+ | Plots, heatmaps |
| **Notebook** | Jupyter | 1.0+ | Interactive development |

### 4.2 Computational Requirements

| Aspect | Requirement | Notes |
|--------|-------------|-------|
| **Memory** | ~500 MB | SHAP computation stores values for all features |
| **CPU** | 2+ cores | XGBoost parallelized; sklearn uses n_jobs=-1 |
| **Disk** | ~100 MB | Raw data + generated outputs (reports, plots) |
| **Runtime** | 5–10 min | Full pipeline on standard hardware |

---

## 5. Deployment & Integration

### 5.1 Inference Pipeline (Production)

```
Active Employees Data (daily/weekly batch)
        │
        ↓
Load & Preprocess
  ├─ Impute missing values
  ├─ Encode categoricals
  └─ Scale (if LR model)
        │
        ↓
Batch Prediction
  ├─ Score: best_model.predict_proba(X_active)
  ├─ Get probabilities: risk_scores = [0.05, 0.72, 0.15, ...]
  └─ Calibrate: best_model_calibrated.predict_proba(X_active)
        │
        ↓
SHAP Explanations
  ├─ Compute SHAP values
  ├─ Extract top 3 drivers
  ├─ Map to actions
  └─ Filter: risk_score >threshold
        │
        ↓
Risk Report CSV
  ├─ Columns: id, risk_score, driver_1-3, action_1-3
  ├─ Sort: descending risk_score
  └─ Export: risk_report_active.csv
        │
        ↓
Dashboard Upload
  ├─ Upload to HR portal
  ├─ Visualize high-risk employees
  ├─ HR manager drill-down
  └─ Track interventions
        │
        ↓
Outcome Tracking
  ├─ Employee stayed → mark success
  ├─ Employee left → log failure
  └─ Feedback loop: retrain quarterly
```

### 5.2 Monitoring & Maintenance

**Monthly Checks**:
- Prediction distribution (% flagged ~20%)
- Feature importance stability
- Fairness drift (FPR/FNR by demographic)

**Quarterly Tasks**:
- Retrain on latest data
- Validate on new test set
- Update fairness audit
- Review model decisions vs. actual attrition

**Annual Review**:
- Full model evaluation
- Compare to alternative models
- External dataset validation
- Update documentation

---

## 6. Quality Assurance & Validation

### 6.1 Data Quality Checks
- [ ] No missing critical columns
- [ ] Target variable binary (0, 1)
- [ ] No direct identifiers in final dataset
- [ ] Protected attributes separated
- [ ] Feature distributions reasonable (no extreme outliers)

### 6.2 Model Validation
- [ ] Train/test split stratified (preserve attrition rate)
- [ ] No data leakage (test set never touched during training)
- [ ] No outcome leakage (outcome variables excluded)
- [ ] SMOTE applied to training only
- [ ] Model AUC-ROC >0.85 ✅
- [ ] Model F1 >0.65 ✅

### 6.3 Explainability Validation
- [ ] SHAP values computed for all predictions
- [ ] Feature importance stable (top 5 consistent)
- [ ] Waterfall plots interpretable (reasonable contributions)
- [ ] Drivers map to actionable interventions

### 6.4 Fairness Validation
- [ ] FPR/FNR computed per demographic
- [ ] Disparity <10% across all groups ✅
- [ ] Fairness audit documented
- [ ] Limitations acknowledged

---

## 7. Security & Privacy Considerations

### 7.1 Data Security
- ✅ No employee names in outputs
- ✅ Anonymized IDs only (EmpID masked)
- ✅ CSV encrypted at rest (if deployed)
- ✅ HTTPS for dashboard transmission
- ✅ Access logs for who viewed profiles

### 7.2 Privacy Compliance
- ✅ GDPR Article 32 (Privacy by Design)
- ✅ GDPR Article 22 (Right to explanation via SHAP)
- ✅ GDPR Right to Erasure (no names → practical deletion)
- ⚠️ Future: Implement right to access/deletion

### 7.3 Ethical Guardrails
- ✅ No protected attributes modeled
- ✅ Fairness audited
- ✅ SHAP ensures explainability
- ✅ HR retains decision authority
- ⚠️ Recommend: Disclose model use to employees

---

## 8. Known Limitations & Future Work

### Current Limitations
1. **Small Sample Size**: ~400 employees; results sensitive to outliers
2. **Snapshot Data**: January 2019; no temporal trends
3. **Synthetic Data**: Feedback/transfer text are generated (used for EDA only)
4. **Single Company**: May not generalize to other industries
5. **No External Features**: Missing macro-economic indicators

### Recommended Future Work
1. **Longitudinal Data**: Collect monthly snapshots for time-series modeling
2. **Real Feedback**: Replace synthetic text with actual employee feedback
3. **Multi-Company Data**: Validate on datasets from multiple organizations
4. **Causal Analysis**: Investigate causal mechanisms (not just associations)
5. **Production Hardening**: API endpoints, model versioning, A/B testing
6. **Continuous Fairness**: Automated fairness monitoring + alerting
7. **Employee Feedback Loop**: Collect employee reactions to interventions

---

## 9. References

### Technical References
- XGBoost: [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- SMOTE: [Chawla et al., 2002](https://arxiv.org/abs/1106.1813)

### Fairness & Responsible AI
- Barocas, Hardt & Narayanan. [Fairness and Machine Learning](https://fairmlbook.org/)
- Mitchell et al. [Model Cards for Model Reporting, 2019](https://arxiv.org/abs/1810.03993)

### Documentation Standards
- [Model Card Best Practices](https://huggingface.co/docs/hub/model-cards)
- [Data Card Best Practices](https://arxiv.org/abs/1803.09010)

---

**Architecture Document Certification**: This document describes the complete technical architecture of Trusted HR-AI as of March 2026. It has been prepared by GROUP N°17 for hackathon evaluation and future reference.
