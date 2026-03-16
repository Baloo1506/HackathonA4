"""
utils/preprocessing.py
-----------------------
GDPR-compliant data loading, anonymization, cleaning, and feature
engineering for the TalentGuard AI attrition prediction pipeline.

Public API (imported by train_pipeline.py):
    load_and_anonymize(csv_path) -> DataFrame
    clean(df)                    -> DataFrame
    build_feature_matrix(df)     -> (X, y, df_sensitive)
"""

import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Column groups ─────────────────────────────────────────────────────────────
PII_DROP_COLS = [
    "Employee_Name", "EmpID", "MarriedID", "MaritalStatusID",
    "GenderID", "EmpStatusID", "DeptID", "PerfScoreID",
    "FromDiversityJobFairID", "Zip", "DOB", "DateofHire",
    "DateofTermination", "LastPerformanceReview_Date",
    "ManagerID", "ManagerName",
]
HASH_COLS       = ["EmployeeNumber"]
SENSITIVE_ATTRS = ["Sex", "RaceDesc"]
TARGET_COL      = "Termd"

NUMERIC_FEATURES = [
    "Salary", "EngagementSurvey", "EmpSatisfaction",
    "SpecialProjectsCount", "DaysLateLast30", "Absences",
]
CATEGORICAL_FEATURES = [
    "Department", "Position", "MaritalDesc", "CitizenDesc",
    "HispanicLatino", "RaceDesc", "RecruitmentSource",
    "PerformanceScore", "EmploymentStatus", "Sex", "State",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_id(value: str, salt: str = "talentguard_2024") -> str:
    """One-way salted hash to pseudonymize employee IDs."""
    return "EMP_" + hashlib.sha256(f"{salt}{value}".encode()).hexdigest()[:8].upper()


# ── Public functions ──────────────────────────────────────────────────────────

def load_and_anonymize(csv_path: str) -> pd.DataFrame:
    """
    Load the raw HR CSV, drop direct PII, pseudonymize IDs,
    and derive privacy-safe features from dates.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Derive tenure BEFORE dropping hire date
    ref = pd.Timestamp("2024-01-01")
    if "DateofHire" in df.columns:
        df["TenureYears"] = (
            (ref - pd.to_datetime(df["DateofHire"], errors="coerce")).dt.days / 365.25
        ).round(1)

    # Derive age bucket BEFORE dropping DOB
    if "DOB" in df.columns:
        age = (ref - pd.to_datetime(df["DOB"], errors="coerce")).dt.days / 365.25
        df["AgeBucket"] = pd.cut(
            age, bins=[0, 25, 35, 45, 55, 120],
            labels=["<25", "25-34", "35-44", "45-54", "55+"]
        ).astype(str)

    # Drop all PII columns
    cols_to_drop = [c for c in PII_DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Pseudonymize employee number
    for col in HASH_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(_hash_id)

    print(f"[Anonymize] Dropped {len(cols_to_drop)} PII columns. "
          f"Remaining: {df.shape[1]} columns, {df.shape[0]} rows.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Strip whitespace from strings
      - Normalise Yes/No → 1/0
      - Drop columns with >60% missing values
      - Clip salary outliers
    """
    df = df.copy()

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Yes / No → 1 / 0
    yn_map = {"Yes": 1, "No": 0, "Y": 1, "N": 0}
    for col in df.columns:
        if df[col].dropna().isin(yn_map.keys()).all():
            df[col] = df[col].map(yn_map)

    # Drop near-empty columns
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > 0.6].index.tolist()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"[Clean] Dropped {len(drop_cols)} sparse columns: {drop_cols}")

    # Clip salary
    if "Salary" in df.columns:
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
        lo, hi = df["Salary"].quantile([0.01, 0.99])
        df["Salary"] = df["Salary"].clip(lo, hi)

    print(f"[Clean] Shape after cleaning: {df.shape}")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Encode categoricals, fill missing values, and split into:
      X          – numeric feature matrix
      y          – binary target (Termd)
      df_sensitive – Sex / RaceDesc for AIF360 fairness audit
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. "
                         f"Available columns: {list(df.columns)}")

    df = df.copy()
    y = df[TARGET_COL].astype(int)

    # Preserve sensitive attributes before encoding
    df_sensitive = df[[c for c in SENSITIVE_ATTRS if c in df.columns]].copy()

    # Drop target and free-text / ID columns
    drop_for_X = [TARGET_COL, "TermReason", "EmployeeNumber",
                  "AgeBucket"]          # AgeBucket is string – drop or encode separately
    drop_for_X = [c for c in drop_for_X if c in df.columns]
    X = df.drop(columns=drop_for_X)

    # Label-encode remaining categoricals
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("Unknown"))

    # Fill remaining NaNs with column median (numeric only)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    # Encode sensitive attrs for AIF360 (needs numeric)
    for col in df_sensitive.columns:
        le = LabelEncoder()
        df_sensitive[col] = le.fit_transform(df_sensitive[col].astype(str).fillna("Unknown"))

    print(f"[Features] X: {X.shape} | "
          f"y distribution: {y.value_counts().to_dict()} | "
          f"Sensitive attrs: {list(df_sensitive.columns)}")
    return X, y, df_sensitive
