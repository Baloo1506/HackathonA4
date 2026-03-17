#!/usr/bin/env python3
"""
Quick Guide: How to Save Your Model and Start the API

This script shows you the exact steps needed.
"""

import os
import sys

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║         HOW TO START THE FLASK API WITH YOUR MODEL                    ║
╚═══════════════════════════════════════════════════════════════════════╝

STEP 1: Save Model from Jupyter Notebook
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Open hr_attribution_7.ipynb and add this cell at END OF SECTION 6:

---BEGIN CODE---

import json
import joblib
from datetime import datetime

# Define feature labels (human-readable)
FEATURE_LABELS = {
    'EmpSatisfaction': 'Employee satisfaction score (1–5)',
    'EngagementSurvey': 'Engagement survey score (0–5)',
    'engagement_x_satisfaction': 'Engagement × satisfaction composite',
    'has_transfer_request': 'Has submitted a transfer request',
    'transfer_request_sentiment': 'Sentiment of transfer request',
    'salary_vs_dept_mean': 'Salary vs. department average (ratio)',
    'Absences': 'Number of absences',
    'DaysLateLast30': 'Days late in last 30 days',
    'tenure_years': 'Tenure (years at data snapshot)',
    'SpecialProjectsCount': 'Number of special projects',
    'Salary': 'Annual salary',
    'age': 'Age (at data snapshot)',
    'Department_enc': 'Department (encoded)',
    'Position_enc': 'Position (encoded)',
    'MaritalDesc_enc': 'Marital status (encoded)',
    'CitizenDesc_enc': 'Citizenship (encoded)',
    'RecruitmentSource_enc': 'Recruitment source (encoded)',
    'PerformanceScore_enc': 'Performance score (encoded)',
}

# Define action mappings (HR actions for each feature)
ACTION_MAP = {
    'EmpSatisfaction': 'HR check-in / satisfaction interview',
    'EngagementSurvey': 'Manager discussion / project reassignment',
    'engagement_x_satisfaction': 'HR check-in / engagement improvement plan',
    'has_transfer_request': 'Internal mobility conversation',
    'transfer_request_sentiment': 'Review transfer request and respond formally',
    'salary_vs_dept_mean': 'Compensation review',
    'Salary': 'Compensation review',
    'Absences': 'Wellbeing or flexibility discussion',
    'DaysLateLast30': 'Wellbeing or flexibility discussion',
    'tenure_years': 'Career development milestone review',
    'SpecialProjectsCount': 'Assign meaningful stretch projects',
    'age': 'Age-appropriate retention strategy',
}

# Save model
joblib.dump(best_model_calibrated, 'model.pkl', compress=3)
print('✓ Saved model to model.pkl')

# Save SHAP explainer
joblib.dump(explainer, 'explainer.pkl', compress=3)
print('✓ Saved explainer to explainer.pkl')

# Save scaler if using Logistic Regression
if best_name == 'Logistic Regression':
    joblib.dump(scaler, 'scaler.pkl', compress=3)
    print('✓ Saved scaler to scaler.pkl')

# Save metadata
metadata = {
    'created': datetime.utcnow().isoformat(),
    'model_type': best_name,
    'feature_cols': FEATURE_COLS,
    'feature_labels': FEATURE_LABELS,
    'action_map': ACTION_MAP,
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print('✓ Saved metadata to metadata.json')
print('✓ All files saved successfully!')

---END CODE---


STEP 2: Verify Files Were Created
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In terminal, check that these files exist:
  ls -lh model.pkl explainer.pkl metadata.json

You should see:
  -rw-r--r--  2.5M  model.pkl
  -rw-r--r-- 15.3M  explainer.pkl
  -rw-r--r--  3.1K  metadata.json


STEP 3: Start Flask API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run this command:
  python app.py

You should see:
  INFO: Starting Flask API...
  INFO: ✓ Loaded model from model.pkl
  INFO: ✓ Loaded explainer from explainer.pkl
  INFO: ✓ Loaded metadata with 16 features
  Running on http://0.0.0.0:5000


STEP 4: Test API (In Another Terminal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run this command:
  python test_api.py

You should see:
  ✓ Passed: 9/9


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
             NEED HELP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full documentation:
  • ADD_TO_NOTEBOOK.md — Copy-paste code for notebook
  • API_QUICKSTART.md — Complete setup guide
  • FLASK_API_START_HERE.md — 5-minute quick start

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Check what files exist
print("\nCURRENT STATUS:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

files_needed = ['model.pkl', 'explainer.pkl', 'metadata.json']
for fname in files_needed:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024*1024 else f"{size / 1024:.1f}KB"
        print(f"  ✓ {fname} ({size_str}) — FOUND")
    else:
        print(f"  ✗ {fname} — MISSING (needed)")

print("\nNEXT ACTION:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

missing_files = [f for f in files_needed if not os.path.exists(f)]
if missing_files:
    print(f"\n⚠️  Missing {len(missing_files)} file(s). Follow STEP 1 above:")
    print("\n1. Open hr_attribution_7.ipynb")
    print("2. Go to end of Section 6 (after training)")
    print("3. Copy the code from 'BEGIN CODE' to 'END CODE' above")
    print("4. Paste it into a new cell")
    print("5. Run the cell")
    print("6. Then run: python app.py")
else:
    print("\n✓ All model files found!")
    print("\nYou can now start the API:")
    print("  python app.py")
