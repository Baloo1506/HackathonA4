#!/usr/bin/env python3
"""Convert pipeline artifacts/ outputs into the models/ layout the Streamlit app expects.

This script attempts to write parquet files for X/y when possible, and falls back to joblib
when parquet support (pyarrow/fastparquet) isn't installed.
"""
import os
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MOD = ROOT / "models"
MOD.mkdir(exist_ok=True)

def safe_to_parquet(df, path):
    try:
        df.to_parquet(path)
        return True
    except Exception:
        joblib.dump(df, str(path) + ".joblib")
        return False

def main():
    # 1) model
    pkl_model = ART / "random_forest.pkl"
    if pkl_model.exists():
        with open(pkl_model, "rb") as f:
            model = pickle.load(f)
        joblib.dump(model, MOD / "saved_model.joblib")
        print("Wrote models/saved_model.joblib")
    else:
        print("No random_forest.pkl found in artifacts/")

    # 2) feature cols
    feat_json = ART / "feature_names.json"
    if feat_json.exists():
        feat_cols = json.load(open(feat_json))
        joblib.dump(feat_cols, MOD / "feature_cols.joblib")
        print("Wrote models/feature_cols.joblib")

    # 3) employee ids from risk_scores.csv
    risk_csv = ART / "risk_scores.csv"
    if risk_csv.exists():
        df_risk = pd.read_csv(risk_csv)
        emp_ids = df_risk["employee_id"].tolist()
        joblib.dump(emp_ids, MOD / "employee_ids.joblib")
        print("Wrote models/employee_ids.joblib")

    # 4) X_full and y_full (CSV -> parquet or joblib)
    x_csv = ART / "X_full.csv"
    y_csv = ART / "y_full.csv"
    if x_csv.exists():
        df_x = pd.read_csv(x_csv)
        wrote = safe_to_parquet(df_x, MOD / "X_full.parquet")
        if wrote:
            print("Wrote models/X_full.parquet")
        else:
            print("Wrote models/X_full.parquet.joblib (parquet unavailable)")
    if y_csv.exists():
        df_y = pd.read_csv(y_csv)
        wrote = safe_to_parquet(df_y, MOD / "y_full.parquet")
        if wrote:
            print("Wrote models/y_full.parquet")
        else:
            print("Wrote models/y_full.parquet.joblib (parquet unavailable)")

    # 5) shap values
    shap_npy = ART / "shap_values.npy"
    if shap_npy.exists():
        shap = np.load(shap_npy, allow_pickle=True)
        joblib.dump(shap, MOD / "shap_values.joblib")
        print("Wrote models/shap_values.joblib")

    # 6) metrics.json generated from meta JSON
    meta_json = ART / "random_forest_meta.json"
    if meta_json.exists():
        meta = json.load(open(meta_json))
        metrics = {
            "auc": meta.get("roc_auc"),
            "cv_auc_mean": meta.get("cv_roc_auc_mean"),
            "accuracy": None
        }
        try:
            metrics["accuracy"] = meta.get("classification_report", {}).get("accuracy")
        except Exception:
            pass
        json.dump(metrics, open(MOD / "metrics.json", "w"), indent=2)
        print("Wrote models/metrics.json")

    print("Sync complete. You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()
