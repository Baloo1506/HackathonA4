"""
HR Attrition Prediction Pipeline
==================================
Goal: predict which employees are likely to leave the company
      and identify the key factors driving attrition.

This script:
  1. Calls the preprocessing pipeline (preprocessing.py)
  2. Trains a Logistic Regression, a Random Forest, and a Gradient Boosting model
  3. Evaluates models with stratified cross-validation (AUC, F1)
  4. Selects the best model and evaluates it on the held-out test set
  5. Displays the most important predictive features
  6. Generates visualizations (ROC curve, confusion matrix, feature importances)

Usage:
    python pipeline.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import run_preprocessing

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TOP_N_FEATURES = 20

TARGET_COL = "Termd"


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------
print("=" * 60)
print("  HR ATTRITION PREDICTION PIPELINE")
print("=" * 60)

df = run_preprocessing()

# ---------------------------------------------------------------------------
# 2. Split features and target
# ---------------------------------------------------------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"\n[ML] Dataset: {X.shape[0]} rows x {X.shape[1]} features")
print(f"[ML] Target distribution: active={int((y == 0).sum())}, "
      f"left={int((y == 1).sum())} "
      f"({100 * y.mean():.1f}% attrition rate)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    ),
}

# ---------------------------------------------------------------------------
# 4. Cross-validation
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}

print(f"\n[ML] Cross-validation ({CV_FOLDS} folds):")
print("-" * 50)
for name, model in models.items():
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="f1", n_jobs=-1)
    results[name] = {"auc_mean": auc_scores.mean(), "auc_std": auc_scores.std(),
                     "f1_mean": f1_scores.mean()}
    print(f"  {name:30s} | AUC = {auc_scores.mean():.3f} +/- {auc_scores.std():.3f} "
          f"| F1 = {f1_scores.mean():.3f}")

# Select best model by AUC
best_name = max(results, key=lambda n: results[n]["auc_mean"])
best_model = models[best_name]
print(f"\n  -> Best model: {best_name} (AUC = {results[best_name]['auc_mean']:.3f})")

# ---------------------------------------------------------------------------
# 5. Final training and evaluation on the test set
# ---------------------------------------------------------------------------
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\n[ML] Test set results ({best_name}):")
print("-" * 50)
print(f"  ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
print()
print(classification_report(y_test, y_pred,
                             target_names=["Active (0)", "Left (1)"]))

# ---------------------------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------------------------

# -- 6a. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title(f"ROC Curve -- {best_name}")
axes[0].legend(loc="lower right")

# -- 6b. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Active", "Left"],
    ax=axes[1],
    colorbar=False,
    cmap="Blues",
)
axes[1].set_title(f"Confusion Matrix -- {best_name}")

plt.tight_layout()
plt.savefig("roc_confusion_matrix.png", dpi=120)
print("[ML] Chart saved: roc_confusion_matrix.png")

# -- 6c. Feature importances (Random Forest / Gradient Boosting)
clf = best_model if not hasattr(best_model, "named_steps") else best_model.named_steps.get("clf", None)
if hasattr(clf, "feature_importances_"):
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(TOP_N_FEATURES).sort_values()

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    top_features.plot(kind="barh", ax=ax2, color="steelblue")
    ax2.set_title(f"Top {TOP_N_FEATURES} most important features\n({best_name})")
    ax2.set_xlabel("Importance (Gini)")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=120)
    print("[ML] Chart saved: feature_importances.png")

    print(f"\n[ML] Top {TOP_N_FEATURES} attrition predictors:")
    print("-" * 50)
    for feat, imp in importances.nlargest(TOP_N_FEATURES).items():
        print(f"  {feat:45s} {imp:.4f}")

elif hasattr(clf, "coef_"):
    # Logistic Regression: use absolute coefficient values
    coefs = pd.Series(np.abs(clf.coef_[0]), index=X.columns)
    top_features = coefs.nlargest(TOP_N_FEATURES).sort_values()

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    top_features.plot(kind="barh", ax=ax2, color="steelblue")
    ax2.set_title(f"Top {TOP_N_FEATURES} features (|coefficient|)\n({best_name})")
    ax2.set_xlabel("|Coefficient|")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=120)
    print("[ML] Chart saved: feature_importances.png")

print("\n✅ Pipeline complete.")
