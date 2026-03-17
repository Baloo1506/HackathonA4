from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data
df_full = pd.read_csv("HRDataset_with_review_and_ancienete.csv")
df_full["EmpID"] = df_full["EmpID"].astype(int)
feature_importance = pd.read_csv("importance_variables_departs.csv")
at_risk_employees = pd.read_csv("employes_a_risque_depart.csv")
at_risk_employees["EmpID"] = at_risk_employees["EmpID"].astype(int)

# Prepare active reference
active_reference = df_full[df_full["Termd"] == 0].copy()

# Variables to compare
vars_to_compare = [
    "Absences",
    "EngagementSurvey",
    "EmpSatisfaction",
    "Salary",
    "ancienete"
]

risk_if_higher = ["Absences"]
risk_if_lower = ["Salary", "EmpSatisfaction", "EngagementSurvey", "ancienete"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/feature_importance')
def get_feature_importance():
    top_features = feature_importance.head(12).to_dict('records')
    return jsonify(top_features)

@app.route('/api/top_risk_employees')
def get_top_risk_employees():
    display_cols = [
        col for col in [
            "EmpID", "Employee_Name", "Department", "Position",
            "Salary", "Absences", "EngagementSurvey",
            "EmpSatisfaction", "DaysLateLast30", "ancienete",
            "risk_of_leaving"
        ] if col in at_risk_employees.columns
    ]
    top10 = at_risk_employees[display_cols].head(10).to_dict('records')
    return jsonify(top10)

@app.route('/api/employee/<int:emp_id>')
def get_employee_details(emp_id):
    employee = df_full[df_full["EmpID"] == emp_id]
    if employee.empty:
        return jsonify({"error": "Employee not found"}), 404
    
    employee = employee.iloc[0]
    employee_name = employee["Employee_Name"]
    
    available_vars = [v for v in vars_to_compare if v in employee.index]
    
    comparison_data = []
    for var in available_vars:
        emp_val = employee[var]
        mean_val = active_reference[var].mean()
        ecart_pct = ((emp_val - mean_val) / mean_val) * 100 if mean_val != 0 else 0
        
        if var in risk_if_higher:
            color = "crimson" if ecart_pct > 0 else "seagreen"
        elif var in risk_if_lower:
            color = "crimson" if ecart_pct < 0 else "seagreen"
        else:
            color = "gray"
        
        comparison_data.append({
            "Variable": var,
            "Employe": float(emp_val),
            "Moyenne": float(mean_val),
            "Ecart_pct": float(ecart_pct),
            "Color": color
        })
    
    # Sort by Ecart_pct
    comparison_data.sort(key=lambda x: x["Ecart_pct"])
    
    return jsonify({
        "name": employee_name,
        "comparison": comparison_data
    })

if __name__ == '__main__':
    app.run(debug=True)