"""
Save Model Artifacts for Production Deployment

This script serializes the trained model, SHAP explainer, and metadata
for use by the Flask API (app.py) and other production systems.

Run this after training your model in hr_attribution_7.ipynb.
It creates three files:
  - model.pkl: Trained Random Forest (or other model)
  - explainer.pkl: SHAP TreeExplainer
  - metadata.json: Feature names, labels, action mappings
  - scaler.pkl: StandardScaler (if using Logistic Regression)
"""

import json
import joblib
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# These variables should be defined in your notebook (hr_attribution_7.ipynb)
# after training. Copy them here or modify the script to import from notebook.

# Model & Explainer (from Section 6 & 7 of notebook)
# best_model = trained RandomForestClassifier or XGBClassifier
# best_model_calibrated = CalibratedClassifierCV wrapper (for predict_proba)
# explainer = SHAP explainer (TreeExplainer or LinearExplainer)

# Features & Labels (from Section 6 & 3 of notebook)
# FEATURE_COLS = list of feature names used in model
# FEATURE_LABELS = dict mapping feature names to human-readable labels
# ACTION_MAP = dict mapping feature names to HR action strings

# ─────────────────────────────────────────────────────────────────────────────
# Save artifacts from notebook environment
# ─────────────────────────────────────────────────────────────────────────────

def save_model_artifacts(best_model, best_model_calibrated, explainer,
                        FEATURE_COLS, FEATURE_LABELS, ACTION_MAP,
                        scaler=None, output_dir='.'):
    """
    Save all artifacts needed for production API.
    
    Args:
      best_model: Trained model (RandomForest, XGBoost, etc.)
      best_model_calibrated: Calibrated version for predict_proba
      explainer: SHAP explainer
      FEATURE_COLS: List of feature names
      FEATURE_LABELS: Dict of feature name -> human-readable label
      ACTION_MAP: Dict of feature name -> HR action string
      scaler: StandardScaler if using Logistic Regression (optional)
      output_dir: Directory to save files (default '.')
    
    Returns:
      dict: Paths to saved files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    files_saved = {}
    
    # ── 1. Save calibrated model (for predict_proba) ──────────────────────────
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(best_model_calibrated, model_path, compress=3)
    print(f'✓ Saved model to {model_path}')
    files_saved['model'] = model_path
    
    # ── 2. Save SHAP explainer ────────────────────────────────────────────────
    explainer_path = os.path.join(output_dir, 'explainer.pkl')
    joblib.dump(explainer, explainer_path, compress=3)
    print(f'✓ Saved explainer to {explainer_path}')
    files_saved['explainer'] = explainer_path
    
    # ── 3. Save scaler if present ─────────────────────────────────────────────
    if scaler is not None:
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path, compress=3)
        print(f'✓ Saved scaler to {scaler_path}')
        files_saved['scaler'] = scaler_path
    
    # ── 4. Save metadata (features, labels, actions) ───────────────────────────
    metadata = {
        'created': datetime.utcnow().isoformat(),
        'model_type': type(best_model).__name__,
        'feature_cols': FEATURE_COLS,
        'feature_labels': FEATURE_LABELS,
        'action_map': ACTION_MAP,
        'num_features': len(FEATURE_COLS),
        'has_scaler': scaler is not None,
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'✓ Saved metadata to {metadata_path}')
    files_saved['metadata'] = metadata_path
    
    print(f'\n✓ All artifacts saved to {output_dir}')
    return files_saved


# ─────────────────────────────────────────────────────────────────────────────
# Run from Jupyter notebook
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("""
    This script should be run from within your Jupyter notebook AFTER training.
    
    In hr_attribution_7.ipynb, add a new cell after Section 6 with:
    
    ---
    # Save model artifacts for production
    from save_model import save_model_artifacts
    
    save_model_artifacts(
        best_model=best_model,
        best_model_calibrated=best_model_calibrated,
        explainer=explainer,
        FEATURE_COLS=FEATURE_COLS,
        FEATURE_LABELS=FEATURE_LABELS,
        ACTION_MAP=ACTION_MAP,
        scaler=scaler,  # Only if using Logistic Regression
        output_dir='.'
    )
    ---
    """)
