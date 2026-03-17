# HackathonA4 Setup Guide

## Environment Status ✓

All dependencies have been installed and verified for the main branch.

### Installed Packages
- **pandas** 3.0.1 — Data manipulation
- **numpy** 2.4.3 — Numerical computing
- **matplotlib** 3.9.4 — Visualization
- **seaborn** 0.13.2 — Statistical visualization
- **scikit-learn** 1.8.0 — Machine learning
- **xgboost** 3.2.0 — Gradient boosting
- **shap** 0.51.0 — Model interpretability
- **vaderSentiment** 3.3.2 — Sentiment analysis
- **imbalanced-learn** 0.14.1 — Handling imbalanced datasets

### Python Environment
- **Type**: Virtual Environment (.venv)
- **Python Version**: 3.14.1
- **Location**: `/Users/Apple/Downloads/Data/HackathonA4/.venv`

## Quick Start

The notebooks are ready to run. You can open either:
- `hr_attribution_7.ipynb` — Main analysis notebook
- `hrattrition2.ipynb` — Secondary analysis

All required dependencies are pre-installed in the virtual environment.

## Install Additional Packages (if needed)

If you need to install additional packages later:

```bash
pip install -r requirements.txt
```

Or install individual packages:

```bash
pip install package_name
```

## Data Files
- `HRDataset_v14.csv` — Original HR dataset
- `HRDataset_v14_enriched.csv` — Enriched dataset with additional features

## Project Structure
This is a Group 17 hackathon project focused on:
- Identifying employees at risk of voluntary turnover
- Using AI/ML to predict attrition
- Providing actionable retention solutions
