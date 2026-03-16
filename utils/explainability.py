import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from typing import List, Tuple

def build_explainer(model, X_background):
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier"):
        return shap.TreeExplainer(model)
    if model_type == "LogisticRegression":
        return shap.LinearExplainer(model, X_background.values)
    background = shap.sample(X_background.values, min(50, len(X_background)))
    return shap.KernelExplainer(model.predict_proba, background)

def compute_shap_values(explainer, X):
    """Return a 2D array (n_samples x n_features) for the positive class."""
    values = explainer.shap_values(X.values)
    # TreeExplainer on a classifier returns a list [class0, class1]
    # each element may itself be 2D or 3D depending on SHAP version
    if isinstance(values, list):
        values = values[1]          # positive class
    values = np.array(values)
    # If 3D (n_samples, n_features, n_classes), take last slice
    if values.ndim == 3:
        values = values[:, :, 1]
    return values                   # guaranteed 2D

def global_importance_df(shap_values, feature_names):
    """Mean absolute SHAP per feature, sorted descending."""
    sv = np.array(shap_values)
    if sv.ndim != 2:
        raise ValueError(f"Expected 2D shap_values, got shape {sv.shape}")
    mean_abs = np.abs(sv).mean(axis=0)   # 1D array, length = n_features
    df = pd.DataFrame({"feature": list(feature_names), "importance": mean_abs})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)

def plot_global_importance(shap_values, feature_names, top_n=15):
    df = global_importance_df(shap_values, feature_names).head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.4)))
    colors = ["#E85D24" if v > df["importance"].median() else "#3B8BD4" for v in df["importance"]]
    ax.barh(df["feature"][::-1], df["importance"][::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def explain_employee(shap_values, X, employee_idx, top_n=5):
    sv = np.array(shap_values)[employee_idx]
    feat_names = list(X.columns)
    df = pd.DataFrame({
        "feature": feat_names,
        "shap_value": sv,
        "feature_value": X.iloc[employee_idx].values,
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    lines = []
    for _, row in df.iterrows():
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        lines.append(f"  {row['feature']} = {round(row['feature_value'], 2)} "
                     f"({direction} risk by {abs(row['shap_value']):.3f})")
    explanation = "Top factors driving this prediction:\n" + "\n".join(lines)
    return df, explanation

def plot_employee_waterfall(shap_values, X, employee_idx, top_n=8):
    sv = np.array(shap_values)[employee_idx]
    feat_names = list(X.columns)
    indices = np.argsort(np.abs(sv))[::-1][:top_n]
    selected_sv = sv[indices]
    selected_names = [feat_names[i] for i in indices]
    order = np.argsort(selected_sv)
    sorted_sv = selected_sv[order]
    sorted_names = [selected_names[i] for i in order]
    colors = ["#E85D24" if v > 0 else "#3B8BD4" for v in sorted_sv]
    fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.45)))
    ax.barh(sorted_names, sorted_sv, color=colors, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on attrition risk)", fontsize=10)
    ax.set_title("Individual Prediction Explanation", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
