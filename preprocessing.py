"""
HR Dataset Preprocessing Pipeline
===================================
Goal: anonymize the HR dataset and prepare it for attrition prediction.

Steps:
  1. Remove directly identifying confidential columns (PII)
     (employee name, employee ID, zip code, manager name/ID)
  2. Convert date columns into useful numeric metrics
     - DOB -> Age (in years)
     - DateofHire -> TenureYears (years of service)
     - LastPerformanceReview_Date -> DaysSinceLastReview
  3. Drop redundant columns
     (numeric ID versions of categorical columns already present)
  4. Drop columns not available at inference time or that cause data leakage
     (DateofTermination, TermReason, EmploymentStatus)
  5. One-Hot encode categorical variables
  6. Min-Max scale continuous numeric variables

Usage:
    python preprocessing.py
    -> produces: HRDataset_anonymized.csv    (anonymized, human-readable)
    -> produces: HRDataset_preprocessed.csv  (encoded + scaled, ML-ready)
"""

import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_FILE = "HRDataset_v14_enriched.csv"
OUTPUT_ANONYMIZED = "HRDataset_anonymized.csv"
OUTPUT_ML_READY = "HRDataset_preprocessed.csv"

# Columns removed because they directly identify an individual (PII)
PII_COLUMNS = [
    "Employee_Name",   # Full employee name
    "EmpID",           # Unique employee identifier
    "Zip",             # Zip code (precise location)
    "ManagerName",     # Manager's full name
    "ManagerID",       # Manager's unique identifier
]

# Redundant columns: numeric ID versions of categorical columns already present
REDUNDANT_ID_COLUMNS = [
    "MarriedID",       # duplicate of MaritalDesc
    "MaritalStatusID", # duplicate of MaritalDesc
    "GenderID",        # duplicate of Sex
    "EmpStatusID",     # duplicate of EmploymentStatus
    "DeptID",          # duplicate of Department
    "PerfScoreID",     # duplicate of PerformanceScore
    "PositionID",      # duplicate of Position
]

# Date columns to transform into numeric metrics
DATE_COLUMNS_TO_TRANSFORM = {
    "DOB": "Age",
    "DateofHire": "TenureYears",
    "LastPerformanceReview_Date": "DaysSinceLastReview",
}

# Columns to drop after transformation or because they are unavailable at inference time
COLUMNS_TO_DROP_AFTER_TRANSFORM = [
    "DateofTermination",  # not available at prediction time (future event)
    "TermReason",         # not available at prediction time (future event)
    "EmploymentStatus",   # directly derived from Termd (target), causes data leakage
]

# Continuous numeric variables to scale
NUMERIC_COLS_TO_SCALE = [
    "Salary",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "Age",
    "TenureYears",
    "DaysSinceLastReview",
]

# Categorical variables to one-hot encode
CATEGORICAL_COLS = [
    "Position",
    "State",
    "Sex",
    "MaritalDesc",
    "CitizenDesc",
    "HispanicLatino",
    "RaceDesc",
    "RecruitmentSource",
    "PerformanceScore",
    "Department",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_date(series: pd.Series) -> pd.Series:
    """Parse a date column into datetime, handling multiple date formats."""
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    # Fix 2-digit years misinterpreted as future dates (e.g. 05/05/75 -> 1975, not 2075)
    future_cutoff = pd.Timestamp(date.today()) + pd.DateOffset(years=1)
    future_mask = parsed > future_cutoff
    if future_mask.any():
        parsed = parsed.where(~future_mask,
                              parsed[future_mask] - pd.DateOffset(years=100))
    return parsed


def compute_age(dob_series: pd.Series, reference_date: date = None) -> pd.Series:
    """Compute age in whole years from a date-of-birth column."""
    if reference_date is None:
        reference_date = date.today()
    dob = parse_date(dob_series)
    ref = pd.Timestamp(reference_date)
    age = (ref - dob).dt.days // 365
    return age.astype("Int64")


def compute_tenure_years(hire_series: pd.Series, reference_date: date = None) -> pd.Series:
    """Compute years of service from a hire-date column."""
    if reference_date is None:
        reference_date = date.today()
    hire = parse_date(hire_series)
    ref = pd.Timestamp(reference_date)
    tenure = (ref - hire).dt.days / 365.25
    return tenure.round(2)


def compute_days_since_review(review_series: pd.Series, reference_date: date = None) -> pd.Series:
    """Compute number of days elapsed since the last performance review."""
    if reference_date is None:
        reference_date = date.today()
    review = parse_date(review_series)
    ref = pd.Timestamp(reference_date)
    days = (ref - review).dt.days
    return days.astype("Int64")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(input_file: str = INPUT_FILE,
                      output_anonymized: str = OUTPUT_ANONYMIZED,
                      output_ml_ready: str = OUTPUT_ML_READY,
                      reference_date: date = None) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    input_file : path to the source CSV file
    output_anonymized : path for the anonymized human-readable CSV output
    output_ml_ready : path for the ML-ready encoded and scaled CSV output
    reference_date : reference date for age/tenure calculations (useful for tests)

    Returns
    -------
    df : DataFrame ready for ML model training
    """
    print(f"[1/7] Loading '{input_file}'...")
    df = pd.read_csv(input_file)
    print(f"      {len(df)} rows x {len(df.columns)} columns loaded.")

    # ------------------------------------------------------------------
    # Step 1: Remove PII columns
    # ------------------------------------------------------------------
    print("[2/7] Removing confidential PII columns...")
    cols_to_remove = [c for c in PII_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_remove)
    print(f"      Dropped: {cols_to_remove}")

    # ------------------------------------------------------------------
    # Step 2: Convert date columns into numeric metrics
    # ------------------------------------------------------------------
    print("[3/7] Converting date columns to numeric metrics...")
    df["Age"] = compute_age(df["DOB"], reference_date)
    df["TenureYears"] = compute_tenure_years(df["DateofHire"], reference_date)
    df["DaysSinceLastReview"] = compute_days_since_review(
        df["LastPerformanceReview_Date"], reference_date
    )
    df = df.drop(columns=["DOB", "DateofHire", "LastPerformanceReview_Date"])
    print("      DOB -> Age, DateofHire -> TenureYears, LastPerformanceReview_Date -> DaysSinceLastReview")

    # ------------------------------------------------------------------
    # Step 3: Drop redundant numeric ID columns
    # ------------------------------------------------------------------
    print("[4/7] Dropping redundant ID columns...")
    cols_redundant = [c for c in REDUNDANT_ID_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_redundant)
    print(f"      Dropped: {cols_redundant}")

    # ------------------------------------------------------------------
    # Step 4: Drop leakage columns (not available at inference time)
    # ------------------------------------------------------------------
    print("[5/7] Dropping data leakage columns...")
    cols_leakage = [c for c in COLUMNS_TO_DROP_AFTER_TRANSFORM if c in df.columns]
    df = df.drop(columns=cols_leakage)
    print(f"      Dropped: {cols_leakage}")

    # Target column: Termd (1 = left the company, 0 = still employed)
    target = df["Termd"].copy()

    # ------------------------------------------------------------------
    # Save anonymized human-readable CSV (before encoding)
    # ------------------------------------------------------------------
    print(f"      -> Saving '{output_anonymized}' ({len(df)} rows x {len(df.columns)} columns)")
    df.to_csv(output_anonymized, index=False)

    # ------------------------------------------------------------------
    # Step 5: Handle missing values
    # ------------------------------------------------------------------
    print("[6/7] Handling missing values...")

    # Drop free-text columns: unstructured for standard ML and potentially re-identifying
    for free_text_col in ("Feedback_RH", "Internal_Transfer_Request"):
        if free_text_col in df.columns:
            df = df.drop(columns=[free_text_col])
            print(f"      '{free_text_col}' dropped (unstructured free text).")

    # Numeric columns: impute with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"      '{col}': NaN replaced with median ({median_val:.2f})")

    # Categorical columns: impute with mode
    cat_cols = df.select_dtypes(include=["str", "object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"      '{col}': NaN replaced with mode ('{mode_val}')")

    # ------------------------------------------------------------------
    # Step 6: One-Hot encode categorical variables
    # ------------------------------------------------------------------
    cat_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True, dtype=int)

    # ------------------------------------------------------------------
    # Step 7: Min-Max scale continuous numeric variables
    # ------------------------------------------------------------------
    cols_to_scale = [c for c in NUMERIC_COLS_TO_SCALE if c in df.columns]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # ------------------------------------------------------------------
    # Save ML-ready CSV
    # ------------------------------------------------------------------
    print(f"[7/7] Saving '{output_ml_ready}' ({len(df)} rows x {len(df.columns)} columns)...")
    df.to_csv(output_ml_ready, index=False)

    print("\n✅ Preprocessing complete.")
    print(f"   * {output_anonymized}  -> anonymized, human-readable (for HR)")
    print(f"   * {output_ml_ready}    -> encoded and scaled, ML-ready")
    print(f"   * Target: 'Termd'  --  0 = active ({(target==0).sum()}), 1 = left ({(target==1).sum()})")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
