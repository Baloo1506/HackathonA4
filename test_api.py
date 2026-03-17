"""
Test Script for Flask API Endpoints

Run this script to test all three endpoints after starting the Flask app:
  python app.py

Then in another terminal:
  python test_api.py

Tests:
  1. GET /health — Health check
  2. POST /predict — Predict resignation risk for an employee
  3. POST /explainer — Get SHAP explanations for a prediction
"""

import requests
import json
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = 'http://localhost:5001'
TIMEOUT = 10  # seconds

# Example employee profiles to test
TEST_EMPLOYEES = [
    {
        'name': 'High Risk Employee',
        'profile': {
            'age': 28,
            'tenure_years': 1.5,
            'Salary': 65000,
            'EngagementSurvey': 2.1,
            'EmpSatisfaction': 2,
            'Absences': 8,
            'DaysLateLast30': 3,
            'SpecialProjectsCount': 0,
            'Department_enc': 1,
            'Position_enc': 2,
            'MaritalDesc_enc': 1,
            'CitizenDesc_enc': 0,
            'RecruitmentSource_enc': 2,
            'PerformanceScore_enc': 1,
            'salary_vs_dept_mean': 0.85,
            'engagement_x_satisfaction': 4.2,
            'has_transfer_request': 1,
            'transfer_request_sentiment': 0.0,
            'feedback_sentiment': -0.3,
            'feedback_has_departure_intent': 1,
        }
    },
    {
        'name': 'Medium Risk Employee',
        'profile': {
            'age': 38,
            'tenure_years': 5.2,
            'Salary': 95000,
            'EngagementSurvey': 3.5,
            'EmpSatisfaction': 3,
            'Absences': 4,
            'DaysLateLast30': 1,
            'SpecialProjectsCount': 2,
            'Department_enc': 2,
            'Position_enc': 3,
            'MaritalDesc_enc': 2,
            'CitizenDesc_enc': 0,
            'RecruitmentSource_enc': 1,
            'PerformanceScore_enc': 2,
            'salary_vs_dept_mean': 1.05,
            'engagement_x_satisfaction': 10.5,
            'has_transfer_request': 0,
            'transfer_request_sentiment': 0.0,
            'feedback_sentiment': 0.1,
            'feedback_has_departure_intent': 0,
        }
    },
    {
        'name': 'Low Risk Employee',
        'profile': {
            'age': 45,
            'tenure_years': 12.8,
            'Salary': 125000,
            'EngagementSurvey': 4.6,
            'EmpSatisfaction': 4.5,
            'Absences': 1,
            'DaysLateLast30': 0,
            'SpecialProjectsCount': 5,
            'Department_enc': 3,
            'Position_enc': 4,
            'MaritalDesc_enc': 2,
            'CitizenDesc_enc': 0,
            'RecruitmentSource_enc': 0,
            'PerformanceScore_enc': 3,
            'salary_vs_dept_mean': 1.15,
            'engagement_x_satisfaction': 20.7,
            'has_transfer_request': 0,
            'transfer_request_sentiment': 0.0,
            'feedback_sentiment': 0.4,
            'feedback_has_departure_intent': 0,
        }
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Test Functions
# ─────────────────────────────────────────────────────────────────────────────

def print_header(text):
    """Print a formatted test section header."""
    print(f'\n{"="*70}')
    print(f'  {text}')
    print(f'{"="*70}')


def print_result(endpoint, status_code, response, elapsed_time=None):
    """Pretty-print API response."""
    status_symbol = '✓' if 200 <= status_code < 300 else '✗'
    print(f'\n{status_symbol} {endpoint} — HTTP {status_code}')
    
    if elapsed_time:
        print(f'  Response time: {elapsed_time:.3f}s')
    
    if response:
        try:
            if isinstance(response, str):
                response = json.loads(response)
            print(f'  Response:')
            print(json.dumps(response, indent=2)[:500] + '...' if len(str(response)) > 500 else json.dumps(response, indent=2))
        except Exception as e:
            print(f'  {response}')


def test_health():
    """Test GET /health endpoint."""
    print_header('Test 1: Health Check')
    
    try:
        start = time.time()
        response = requests.get(f'{BASE_URL}/health', timeout=TIMEOUT)
        elapsed = time.time() - start
        
        print_result('/health', response.status_code, response.json(), elapsed)
        return response.status_code == 200
    
    except requests.exceptions.ConnectionError:
        print(f'✗ /health — Connection refused')
        print('  Make sure Flask app is running: python app.py')
        return False
    except Exception as e:
        print(f'✗ /health — Error: {e}')
        return False


def test_predict(employee_name, profile):
    """Test POST /predict endpoint."""
    print_header(f'Test 2: Predict — {employee_name}')
    
    try:
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/predict',
            json=profile,
            timeout=TIMEOUT,
            headers={'Content-Type': 'application/json'}
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f'\n✓ /predict — HTTP {response.status_code}')
            print(f'  Response time: {elapsed:.3f}s')
            print(f'  Risk score: {data.get("risk_score", "N/A")}')
            print(f'  Risk level: {data.get("risk_emoji", "")} {data.get("risk_level", "N/A")}')
            print(f'  Explanation: {data.get("explanation", "N/A")}')
            
            if data.get('top_drivers'):
                print(f'  Top drivers:')
                for i, driver in enumerate(data['top_drivers'][:3], 1):
                    print(f'    {i}. {driver.get("label", driver.get("feature"))} ({driver.get("shap_value", 0):+.3f})')
                    print(f'       Action: {driver.get("action", "N/A")}')
            
            return True
        else:
            print_result('/predict', response.status_code, response.json(), elapsed)
            return False
    
    except Exception as e:
        print(f'✗ /predict — Error: {e}')
        return False


def test_explainer(employee_name, profile):
    """Test POST /explainer endpoint."""
    print_header(f'Test 3: Explainer — {employee_name}')
    
    try:
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/explainer',
            json=profile,
            timeout=TIMEOUT,
            headers={'Content-Type': 'application/json'}
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f'\n✓ /explainer — HTTP {response.status_code}')
            print(f'  Response time: {elapsed:.3f}s')
            print(f'  Risk score: {data.get("risk_score", "N/A")}')
            print(f'  Base value: {data.get("base_value", "N/A")}')
            print(f'  Summary: {data.get("summary", "N/A")}')
            
            if data.get('shap_values'):
                print(f'  SHAP values (top 5):')
                shap_vals = data['shap_values']
                for i, (feat, val) in enumerate(list(shap_vals.items())[:5], 1):
                    print(f'    {i}. {feat}: {val:+.4f}')
            
            return True
        else:
            print_result('/explainer', response.status_code, response.json(), elapsed)
            return False
    
    except Exception as e:
        print(f'✗ /explainer — Error: {e}')
        return False


def run_all_tests():
    """Run all tests."""
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                 HR ATTRITION API — TEST SUITE                        ║
╚══════════════════════════════════════════════════════════════════════╝

Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
API URL: {BASE_URL}
Timeout: {TIMEOUT}s

""")
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health()
    
    if not results['health']:
        print('\n⚠ Health check failed. Cannot continue with other tests.')
        return results
    
    time.sleep(1)  # Brief delay between tests
    
    # Test 2 & 3: Predict and Explainer for each employee
    for emp in TEST_EMPLOYEES:
        emp_name = emp['name']
        profile = emp['profile']
        
        results[f'predict_{emp_name}'] = test_predict(emp_name, profile)
        time.sleep(0.5)
        
        results[f'explainer_{emp_name}'] = test_explainer(emp_name, profile)
        time.sleep(0.5)
    
    # Summary
    print_header('Summary')
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f'\n✓ Passed: {passed}/{total}')
    
    for test_name, passed_flag in results.items():
        symbol = '✓' if passed_flag else '✗'
        print(f'  {symbol} {test_name}')
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print('\n\nTests interrupted by user.')
    except Exception as e:
        print(f'\nUnexpected error: {e}')
