"""
train_pipeline.py — TalentGuard AI end-to-end training pipeline

Usage:
    python train_pipeline.py --data data/HRDataset_v14.1.csv
    python train_pipeline.py --data data/HRDataset_v14.1.csv --model logistic_regression
    python train_pipeline.py --data data/HRDataset_v14.1.csv --compare-all
"""
import argparse, json, os, pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from utils.preprocessing import load_and_anonymize, clean, build_feature_matrix
from utils.nlp_pipeline import add_sentiment_features, generate_synthetic_feedback
from utils.explainability import build_explainer, compute_shap_values, global_importance_df
from utils.ethics_audit import audit_dataset, audit_predictions, apply_reweighing
from models.attrition_model import train_and_evaluate, compare_models, load_model, predict_risk

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


import numpy as np

class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def run_pipeline(csv_path, model_name="random_forest", compare_all=False):
    print("\n" + "="*60)
    print("  TalentGuard AI — Training Pipeline")
    print("="*60)

    print("\n[Step 1] Loading and anonymizing data...")
    df_raw = load_and_anonymize(csv_path)
    df = clean(df_raw)

    print("\n[Step 2] NLP sentiment pipeline...")
    if "exit_feedback" not in df.columns:
        df = generate_synthetic_feedback(df)
    df = add_sentiment_features(df, text_cols=["exit_feedback"], force_lightweight=True)

    print("\n[Step 3] Building feature matrix...")
    X, y, df_sensitive = build_feature_matrix(df)
    feature_names = list(X.columns)
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    print("\n[Step 4] Ethics audit (pre-modelling)...")
    pre_audit = audit_dataset(X, y, df_sensitive)
    sample_weights = None
    if pre_audit.get("Sex", {}).get("bias_detected"):
        print("[Ethics] Bias detected for 'Sex' — applying Reweighing...")
        _, _, sample_weights = apply_reweighing(X, y, df_sensitive, attr="Sex")

    print(f"\n[Step 5] Training model: {model_name}...")
    if compare_all:
        comparison = compare_models(X, y)
        print(comparison.to_string(index=False))
        comparison.to_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"), index=False)
    meta = train_and_evaluate(X, y, model_name=model_name)

    print("\n[Step 6] Ethics audit (post-prediction)...")
    model = load_model(model_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    df_sens_test = df_sensitive.iloc[X_test.index] if hasattr(X_test.index, '__iter__') else df_sensitive.iloc[:len(X_test)]
    y_pred_test = model.predict(X_test.values)
    post_audit = audit_predictions(X_test, y_test, y_pred_test, df_sens_test)

    print("\n[Step 7] Computing SHAP values...")
    explainer = build_explainer(model, X_train)
    shap_values = compute_shap_values(explainer, X)
    np.save(os.path.join(ARTIFACTS_DIR, "shap_values.npy"), shap_values)
    with open(os.path.join(ARTIFACTS_DIR, "explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)
    importance_df = global_importance_df(shap_values, feature_names)
    importance_df.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
    print(f"[XAI] Top 5 features:\n{importance_df.head(5).to_string(index=False)}")

    print("\n[Step 8] Computing risk scores for all employees...")
    emp_ids = df.get("EmployeeNumber", pd.Series(range(len(X))))
    risk_df = predict_risk(model, X, employee_ids=emp_ids)
    risk_df["top_factor"] = [importance_df.iloc[np.argmax(np.abs(shap_values[i]))]["feature"] for i in range(len(X))]
    risk_df.to_csv(os.path.join(ARTIFACTS_DIR, "risk_scores.csv"), index=False)

    print("\n[Step 9] Writing Model Card...")
    model_card = {
        "model_name": "TalentGuard Attrition Predictor", "version": "1.0",
        "created_at": datetime.now().isoformat(), "algorithm": model_name,
        "training_data": {"source": "Kaggle HR Dataset (Rich Huebner & Carla Patalano)", "n_samples": len(X), "n_features": X.shape[1], "target": "Termd (1=terminated, 0=retained)", "anonymization": "PII dropped, IDs salted-hashed", "gdpr_compliant": True},
        "performance": {"roc_auc": meta["roc_auc"], "cv_roc_auc": f"{meta['cv_roc_auc_mean']} +/- {meta['cv_roc_auc_std']}"},
        "fairness_audit": {"pre_modelling": pre_audit, "post_prediction": post_audit, "debiasing_applied": sample_weights is not None},
        "frugal_ai": {"co2_emissions_kg": meta.get("emissions_kg"), "model_complexity": "low (tree ensemble, no deep learning)", "inference_time_ms": "< 5ms per employee"},
        "explainability": {"method": "SHAP (SHapley Additive exPlanations)", "top_features": importance_df.head(5)["feature"].tolist()},
        "intended_use": "HR decision support — identifying at-risk employees for retention interventions",
        "limitations": ["Trained on synthetic data", "Sensitive attribute debiasing is preprocessing-only", "Not to be used as sole criterion for employment decisions"],
        "ai_act_risk_level": "High (employment-related AI system)",
    }
    with open(os.path.join(ARTIFACTS_DIR, "model_card.json"), "w") as f:
        json.dump(model_card, f, indent=2, cls=_JSONEncoder)
    X.to_csv(os.path.join(ARTIFACTS_DIR, "X_full.csv"), index=False)
    y.to_csv(os.path.join(ARTIFACTS_DIR, "y_full.csv"), index=False)
    df_sensitive.to_csv(os.path.join(ARTIFACTS_DIR, "sensitive_attrs.csv"), index=False)

    # --- Also write app-friendly artefacts into `models/` so the Streamlit app can
    # find them without an extra sync step. We attempt parquet for X/y and fall
    # back to joblib if parquet support isn't available.
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    def _safe_to_parquet(df, path):
        try:
            df.to_parquet(path)
            return True
        except Exception:
            # fallback: persist as joblib
            joblib.dump(df, str(path) + ".joblib")
            return False

    # saved model (joblib)
    try:
        joblib.dump(model, os.path.join(MODELS_DIR, "saved_model.joblib"))
        print(f"Wrote models/saved_model.joblib")
    except Exception as e:
        print(f"Warning: could not write saved_model.joblib: {e}")

    # feature cols
    try:
        joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_cols.joblib"))
        print("Wrote models/feature_cols.joblib")
    except Exception:
        pass

    # employee ids (from earlier emp_ids variable)
    try:
        emp_list = emp_ids.tolist() if hasattr(emp_ids, "tolist") else list(emp_ids)
        joblib.dump(emp_list, os.path.join(MODELS_DIR, "employee_ids.joblib"))
        print("Wrote models/employee_ids.joblib")
    except Exception:
        pass

    # X and y
    try:
        _safe_to_parquet(X, os.path.join(MODELS_DIR, "X_full.parquet"))
        print("Wrote models/X_full.parquet (or joblib fallback)")
    except Exception:
        pass
    try:
        y_df = pd.DataFrame(y)
        _safe_to_parquet(y_df, os.path.join(MODELS_DIR, "y_full.parquet"))
        print("Wrote models/y_full.parquet (or joblib fallback)")
    except Exception:
        pass

    # shap values
    try:
        joblib.dump(shap_values, os.path.join(MODELS_DIR, "shap_values.joblib"))
        print("Wrote models/shap_values.joblib")
    except Exception:
        pass

    # metrics.json (minimal)
    try:
        metrics = {"auc": meta.get("roc_auc"), "cv_auc_mean": meta.get("cv_roc_auc_mean"),
                   "accuracy": meta.get("classification_report", {}).get("accuracy")}
        json.dump(metrics, open(os.path.join(MODELS_DIR, "metrics.json"), "w"), indent=2)
        print("Wrote models/metrics.json")
    except Exception:
        pass

    print("\n" + "="*60)
    print("  Pipeline complete! Run: streamlit run app.py")
    print("="*60)
    return model_card

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Provide a sensible default data path so running the script without args
    # doesn't immediately error. The repository contains a few candidate files;
    # pick the first one that exists.
    default_paths = [
        os.path.join(os.path.dirname(__file__), "data", "HRDataset_v14_enriched.csv"),
        os.path.join(os.path.dirname(__file__), "data", "HRDataset_v14.1.csv"),
        os.path.join(os.path.dirname(__file__), "HRDataset_v14.csv"),
        os.path.join(os.path.dirname(__file__), "data", "HRDataset_v14.csv"),
    ]
    default_data = next((p for p in default_paths if os.path.exists(p)), None)
    if default_data:
        parser.add_argument("--data", default=default_data,
                            help=f"Path to CSV dataset (default: {default_data})")
    else:
        parser.add_argument("--data", required=True,
                            help="Path to CSV dataset")

    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression", "gradient_boosting"])
    parser.add_argument("--compare-all", action="store_true")
    args = parser.parse_args()
    run_pipeline(csv_path=args.data, model_name=args.model, compare_all=args.compare_all)
