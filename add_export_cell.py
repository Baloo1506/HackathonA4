import json

nb_path = r'C:\Users\gaspa\Desktop\Git\HackathonA4\hr_attribution_7.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if already added
last_src = ''.join(nb['cells'][-1]['source'])
if 'Export model artifacts' in last_src:
    print('Export cell already present.')
else:
    export_cell_source = [
        "# -- Export model artifacts for Flask API --\n",
        "import joblib, os\n",
        "\n",
        "EXPORT_DIR = os.path.join(os.getcwd(), 'api')\n",
        "os.makedirs(EXPORT_DIR, exist_ok=True)\n",
        "\n",
        "# Use calibrated model for API (best probability estimates)\n",
        "joblib.dump(best_model_calibrated, os.path.join(EXPORT_DIR, 'model.pkl'))\n",
        "joblib.dump(FEATURE_COLS,          os.path.join(EXPORT_DIR, 'feature_cols.pkl'))\n",
        "joblib.dump(label_encoders,        os.path.join(EXPORT_DIR, 'label_encoders.pkl'))\n",
        "joblib.dump(scaler,                os.path.join(EXPORT_DIR, 'scaler.pkl'))\n",
        "joblib.dump(best_name,             os.path.join(EXPORT_DIR, 'best_model_name.pkl'))\n",
        "\n",
        "# Medians from original X_train (before SMOTE) for API imputation\n",
        "feature_medians = X_train.median().to_dict()\n",
        "joblib.dump(feature_medians, os.path.join(EXPORT_DIR, 'feature_medians.pkl'))\n",
        "\n",
        "dept_mean_salaries = (\n",
        "    df_raw[df_raw['EmploymentStatus'].isin(['Active','Voluntarily Terminated'])]\n",
        "    .groupby('Department')['Salary'].mean().to_dict()\n",
        ")\n",
        "joblib.dump(dept_mean_salaries, os.path.join(EXPORT_DIR, 'dept_mean_salaries.pkl'))\n",
        "\n",
        "encoder_classes = {col: list(le.classes_) for col, le in label_encoders.items()}\n",
        "joblib.dump(encoder_classes, os.path.join(EXPORT_DIR, 'encoder_classes.pkl'))\n",
        "\n",
        "print('Model artifacts saved in:', EXPORT_DIR)\n",
        "print('  model.pkl  (calibrated', best_name, ')')\n",
        "print('  Features:', len(FEATURE_COLS), FEATURE_COLS)\n",
        "print('  Departments:', list(dept_mean_salaries.keys()))\n"
    ]

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": export_cell_source
    }

    nb['cells'].append(new_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print('Export cell added. Total cells:', len(nb['cells']))

