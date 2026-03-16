import os, json, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE = True
except ImportError:
    _SMOTE = False

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "logistic_regression": LogisticRegression(max_iter=500,
        class_weight="balanced", random_state=42, C=0.5),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100,
        max_depth=4, learning_rate=0.1, random_state=42),
}

def train_and_evaluate(X, y, model_name="random_forest", test_size=0.2, use_smote=True):
    print(f"\n[Model] Training '{model_name}' ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)
    feature_names = list(X.columns)
    if use_smote and _SMOTE and y_train.value_counts().min() > 5:
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_train, y_train)
    else:
        X_tr, y_tr = X_train.values, y_train.values
    model = MODELS[model_name]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X.values, y.values, cv=cv, scoring="roc_auc")
    print(f"[Model] ROC-AUC: {roc_auc:.3f} | CV: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(classification_report(y_test, y_pred, target_names=["Retained","At Risk"]))
    with open(os.path.join(ARTIFACTS_DIR, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(model, f)
    meta = {
        "model_name": model_name, "roc_auc": round(roc_auc, 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "feature_names": feature_names, "emissions_kg": None,
        "n_train": len(X_train), "n_test": len(X_test),
        "classification_report": report, "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(ARTIFACTS_DIR, f"{model_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta

def load_model(model_name="random_forest"):
    path = os.path.join(ARTIFACTS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Run train_pipeline.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_risk(model, X, employee_ids=None):
    # Return a DataFrame shaped the way the Streamlit app expects:
    # - Employee_Name (anonymized id or hash),
    # - RiskScore (float probability between 0 and 1),
    # - RiskLabel (categorical: Low/Medium/High),
    # - Prediction (0/1)
    probs = model.predict_proba(X.values)[:, 1]
    risk_labels = pd.cut(probs, bins=[0.0, 0.35, 0.65, 1.0],
                        labels=["Low", "Medium", "High"], include_lowest=True)
    result = pd.DataFrame({
        "Employee_Name": employee_ids.values if (employee_ids is not None and hasattr(employee_ids, "values")) else (list(employee_ids) if employee_ids is not None else list(range(len(X)))),
        "RiskScore": probs,
        "RiskLabel": risk_labels.astype(str),
        "Prediction": (probs >= 0.5).astype(int),
    })
    return result.sort_values("RiskScore", ascending=False).reset_index(drop=True)

def compare_models(X, y):
    rows = []
    for name in MODELS:
        meta = train_and_evaluate(X, y, model_name=name)
        rows.append({"Model": name.replace("_"," ").title(),
                     "ROC-AUC": meta["roc_auc"],
                     "CV ROC-AUC": f"{meta['cv_roc_auc_mean']:.3f} +/- {meta['cv_roc_auc_std']:.3f}"})
    return pd.DataFrame(rows)


def top_shap_features(shap_values, feat_cols, n=10):
    """Return a DataFrame with top features by mean(|SHAP|)."""
    import numpy as _np
    mean_abs = _np.mean(_np.abs(shap_values), axis=0)
    df = pd.DataFrame({"Feature": list(feat_cols), "Importance": mean_abs})
    return df.sort_values("Importance", ascending=False).head(n)


def employee_shap_explanation(shap_values, feat_cols, idx, n=8):
    """Return per-employee SHAP contributions as a DataFrame with Feature and SHAP columns."""
    vals = shap_values[idx]
    df = pd.DataFrame({"Feature": list(feat_cols), "SHAP": vals})
    df = df.reindex(df.SHAP.abs().sort_values(ascending=False).index)
    return df.head(n).reset_index(drop=True)


def generate_retention_suggestion(exp_df):
    """Simple heuristic to generate a retention suggestion from employee explanation DataFrame."""
    # Look for top positive SHAPs (increase risk) and return a short suggestion
    pos = exp_df[exp_df["SHAP"] > 0].sort_values("SHAP", ascending=False)
    if pos.empty:
        return "No clear risk drivers detected — consider follow-up survey."
    top = pos.iloc[0]["Feature"]
    return f"Top risk driver: {top}. Consider targeted intervention (coaching, compensation review, or role adjustment)."
