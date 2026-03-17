"""
Flask API for HR Attrition Prediction Model

Three endpoints:
  POST /predict — Returns risk score, risk level, and top drivers for an employee
  POST /explainer — Returns SHAP values (feature importance) for explaining a prediction
  GET  /health — Health check for deployment monitoring
"""

import os
import json
import traceback
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import shap

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend calls from different domains

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model and explainer paths (can be local files or S3)
MODEL_PATH = 'model.pkl'
EXPLAINER_PATH = 'explainer.pkl'
METADATA_PATH = 'metadata.json'

# Global state
model = None
explainer = None
metadata = None
feature_cols = None
feature_labels = None
action_map = None
scaler = None  # For Logistic Regression only


def load_model_artifacts():
    """Load model, explainer, and metadata from disk or S3."""
    global model, explainer, metadata, feature_cols, feature_labels, action_map, scaler
    
    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            logger.warning(f'{MODEL_PATH} not found.')
            logger.info('To use the API with your trained model:')
            logger.info('  1. Open hr_attribution_7.ipynb')
            logger.info('  2. Add code from ADD_TO_NOTEBOOK.md to end of Section 6')
            logger.info('  3. Run the cell to save: model.pkl, explainer.pkl, metadata.json')
            logger.info('  4. Restart this Flask app')
            return False
        
        model = joblib.load(MODEL_PATH)
        logger.info(f'✓ Loaded model from {MODEL_PATH}')
        
        # Load explainer
        if os.path.exists(EXPLAINER_PATH):
            explainer = joblib.load(EXPLAINER_PATH)
            logger.info(f'✓ Loaded explainer from {EXPLAINER_PATH}')
        else:
            logger.warning(f'{EXPLAINER_PATH} not found. SHAP explanations unavailable.')
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            feature_cols = metadata.get('feature_cols', [])
            feature_labels = metadata.get('feature_labels', {})
            action_map = metadata.get('action_map', {})
            logger.info(f'✓ Loaded metadata with {len(feature_cols)} features')
        else:
            logger.warning(f'{METADATA_PATH} not found. Using defaults.')
            feature_cols = []
            feature_labels = {}
            action_map = {}
        
        # Load scaler if present (for Logistic Regression models)
        scaler_path = 'scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f'✓ Loaded scaler from {scaler_path}')
        
        return True
    
    except Exception as e:
        logger.error(f'Error loading model artifacts: {e}')
        logger.error(traceback.format_exc())
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(data, required_fields):
    """Validate that input has all required fields."""
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}"
    return True, None


def prepare_features(data, feature_cols):
    """Convert input JSON to feature array."""
    try:
        # Create DataFrame with single row
        features_dict = {col: data.get(col, np.nan) for col in feature_cols}
        X = pd.DataFrame([features_dict])
        
        # Fill NaN with median (assumes metadata includes median values)
        # For now, use 0 as fallback
        X = X.fillna(0.0)
        
        # Scale if scaler is available (for Logistic Regression)
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=feature_cols)
        
        return True, X
    
    except Exception as e:
        return False, f"Error preparing features: {str(e)}"


def get_risk_level(risk_score, percentiles=None):
    """Classify risk score into High/Medium/Low based on percentiles.
    
    Default thresholds (can be overridden with percentiles from metadata):
      High: >= 0.35 (top 15% of population)
      Medium: >= 0.25 (next 25%)
      Low: < 0.25
    """
    if percentiles:
        high_th = percentiles.get('high', 0.35)
        med_th = percentiles.get('medium', 0.25)
    else:
        high_th = 0.35
        med_th = 0.25
    
    if risk_score >= high_th:
        return 'High'
    elif risk_score >= med_th:
        return 'Medium'
    else:
        return 'Low'


def get_top_drivers(shap_values, feature_cols, n=3):
    """Extract top n positive SHAP drivers (exclude encoded columns)."""
    if shap_values is None or len(shap_values) == 0:
        return []
    
    # Create series of SHAP values
    shap_series = pd.Series(shap_values, index=feature_cols)
    
    # Exclude encoded columns (ending with _enc)
    enc_cols = [c for c in feature_cols if c.endswith('_enc')]
    shap_series = shap_series.drop(labels=enc_cols, errors='ignore')
    
    # Get top positive drivers
    top = shap_series.nlargest(n)
    
    drivers = []
    for feat, shap_val in top.items():
        label = feature_labels.get(feat, feat)
        drivers.append({
            'feature': feat,
            'label': label,
            'shap_value': float(shap_val),
            'action': action_map.get(feat, 'Monitor and follow-up')
        })
    
    return drivers


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for deployment monitoring.
    
    Returns:
      200 OK if model is loaded and ready
      503 Service Unavailable if model not loaded
    """
    if model is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded',
            'timestamp': datetime.utcnow().isoformat()
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'Model ready for predictions',
        'model_type': type(model).__name__,
        'features': len(feature_cols),
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict resignation risk for an employee.
    
    Request JSON format:
    {
      "age": 35,
      "tenure_years": 5.2,
      "Salary": 85000,
      "EngagementSurvey": 4.2,
      "EmpSatisfaction": 4,
      "Absences": 3,
      "DaysLateLast30": 0,
      "SpecialProjectsCount": 2,
      "Department_enc": 2,
      "Position_enc": 5,
      ...
    }
    
    Returns:
    {
      "risk_score": 0.32,
      "risk_level": "Medium",
      "risk_emoji": "🟡",
      "explanation": "Employee shows medium resignation risk...",
      "top_drivers": [
        {
          "feature": "EngagementSurvey",
          "label": "Engagement survey score",
          "shap_value": 0.15,
          "action": "Manager discussion / project reassignment"
        },
        ...
      ],
      "recommended_actions": [
        "Manager discussion / project reassignment",
        ...
      ]
    }
    """
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400
        
        # Validate input
        if not feature_cols:
            return jsonify({'error': 'Model not properly initialized'}), 503
        
        # Prepare features
        success, X = prepare_features(data, feature_cols)
        if not success:
            return jsonify({'error': X}), 400
        
        # Make prediction
        risk_score = float(model.predict_proba(X)[0, 1])  # Probability of resignation (class 1)
        risk_level = get_risk_level(risk_score)
        
        # Get SHAP values for explanation
        top_drivers = []
        if explainer is not None:
            try:
                shap_values = explainer.shap_values(X)
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_values = shap_values[0, :, 1]  # Positive class, first sample
                else:
                    shap_values = shap_values[0]
                
                top_drivers = get_top_drivers(shap_values, feature_cols, n=5)
            except Exception as e:
                logger.warning(f'SHAP computation failed: {e}')
                top_drivers = []
        
        # Extract unique recommended actions from drivers
        recommended_actions = list(dict.fromkeys(
            [d['action'] for d in top_drivers if d['action']]
        ))
        
        # Build response
        risk_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(risk_level, '⚪')
        
        explanations = {
            'High': f'Employee shows high resignation risk ({risk_score:.0%}). Immediate manager and HR intervention recommended.',
            'Medium': f'Employee shows medium resignation risk ({risk_score:.0%}). Check in and discuss career development.',
            'Low': f'Employee shows low resignation risk ({risk_score:.0%}). Standard engagement practices sufficient.'
        }
        
        return jsonify({
            'success': True,
            'risk_score': round(risk_score, 4),
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'explanation': explanations.get(risk_level, 'Unknown risk level'),
            'top_drivers': top_drivers,
            'recommended_actions': recommended_actions,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'detail': str(e)
        }), 500


@app.route('/explainer', methods=['POST'])
def explainer_endpoint():
    """
    Get detailed SHAP explanations for a prediction.
    
    Request JSON format: (same as /predict)
    {
      "age": 35,
      "tenure_years": 5.2,
      ...
    }
    
    Returns:
    {
      "risk_score": 0.32,
      "shap_values": {
        "feature_1": 0.05,
        "feature_2": -0.02,
        ...
      },
      "base_value": 0.25,
      "feature_values": {
        "age": 35,
        "tenure_years": 5.2,
        ...
      },
      "summary": "Top 3 drivers: Feature X (+0.15), Feature Y (+0.08), Feature Z (+0.05)"
    }
    """
    try:
        if explainer is None:
            return jsonify({
                'error': 'SHAP explainer not available',
                'detail': 'Explainer model not loaded'
            }), 503
        
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400
        
        # Prepare features
        success, X = prepare_features(data, feature_cols)
        if not success:
            return jsonify({'error': X}), 400
        
        # Make prediction
        risk_score = float(model.predict_proba(X)[0, 1])
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]  # Positive class
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_positive = shap_values[0, :, 1]  # Positive class, first sample
        else:
            shap_values_positive = shap_values[0]
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.array(base_value).flat[1])
        
        # Build SHAP dictionary
        shap_dict = {
            feature_cols[i]: float(shap_values_positive[i])
            for i in range(len(feature_cols))
        }
        
        # Sort by absolute SHAP value
        shap_sorted = dict(sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        # Get top 3 positive drivers for summary
        top_3 = dict(list(shap_sorted.items())[:3])
        summary = 'Top drivers: ' + ', '.join(
            f"{feat} ({val:+.3f})" for feat, val in top_3.items()
        )
        
        # Feature values used in this prediction
        feature_values = {col: float(X[col].iloc[0]) for col in feature_cols}
        
        return jsonify({
            'success': True,
            'risk_score': round(risk_score, 4),
            'shap_values': shap_sorted,
            'base_value': float(base_value),
            'feature_values': feature_values,
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f'Explainer error: {e}')
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Explainer failed',
            'detail': str(e)
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/health', '/predict', '/explainer']
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error'
    }), 500


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info('Starting Flask API...')
    
    # Load model artifacts
    if load_model_artifacts():
        logger.info('✓ Model artifacts loaded successfully')
    else:
        logger.error('⚠ Warning: Model artifacts not fully loaded. Some endpoints may fail.')
    
    # Run Flask app
    # For production, use Gunicorn: gunicorn -w 4 -b 0.0.0.0:5001 app:app
    app.run(host='0.0.0.0', port=5001, debug=False)
