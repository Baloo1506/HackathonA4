#!/usr/bin/env python3
"""
Quick test of Flask API endpoints
"""
import requests
import json
import time

BASE_URL = 'http://localhost:5001'

# Give API 2 seconds to start
time.sleep(2)

print("\n" + "="*70)
print("TESTING HR ATTRITION API")
print("="*70)

# Test 1: Health Check
print("\n✓ TEST 1: Health Check")
print("-" * 70)
try:
    response = requests.get(f'{BASE_URL}/health', timeout=5)
    result = response.json()
    print(f"  Status: {result['status']}")
    print(f"  Model Type: {result['model_type']}")
    print(f"  Features: {result['features']}")
    print("  ✓ PASS\n")
except Exception as e:
    print(f"  ✗ FAIL: {e}\n")
    exit(1)

# Test 2: Predict (High-Risk Employee)
print("✓ TEST 2: Predict Endpoint (High-Risk Employee)")
print("-" * 70)
payload = {
    "age": 28,
    "tenure_years": 1.5,
    "Salary": 65000,
    "EngagementSurvey": 2.1,
    "EmpSatisfaction": 2,
    "Absences": 8,
    "DaysLateLast30": 3,
    "SpecialProjectsCount": 0,
    "salary_vs_dept_mean": 0.85,
    "engagement_x_satisfaction": 4.2,
    "Department_enc": 2,
    "Position_enc": 5,
    "MaritalDesc_enc": 0,
    "CitizenDesc_enc": 0,
    "RecruitmentSource_enc": 3,
    "PerformanceScore_enc": 1
}
try:
    response = requests.post(f'{BASE_URL}/predict', json=payload, timeout=5)
    result = response.json()
    print(f"  Risk Score: {result.get('risk_score', 'N/A'):.3f}")
    print(f"  Risk Level: {result.get('risk_level', 'N/A')}")
    print(f"  Drivers: {result.get('top_3_drivers', {})}")
    print("  ✓ PASS\n")
except Exception as e:
    print(f"  ✗ FAIL: {e}\n")

# Test 3: Explainer
print("✓ TEST 3: Explainer Endpoint")
print("-" * 70)
try:
    response = requests.post(f'{BASE_URL}/explainer', json=payload, timeout=5)
    result = response.json()
    print(f"  SHAP Base Value: {result.get('base_value', 'N/A'):.4f}")
    print(f"  Num Features: {len(result.get('shap_values', {}))}")
    print(f"  Top Feature: {list(result.get('shap_values', {}).items())[0] if result.get('shap_values') else 'N/A'}")
    print("  ✓ PASS\n")
except Exception as e:
    print(f"  ✗ FAIL: {e}\n")

print("="*70)
print("ALL TESTS COMPLETED")
print("="*70 + "\n")
