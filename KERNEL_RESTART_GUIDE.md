# Kernel Restart Guide

## Issue
You encountered a `KeyboardInterrupt` during pandas import in the Jupyter notebook.

## Root Cause
- The notebook kernel was using the system Python instead of the virtual environment `.venv`
- The pip install command in the first cell was attempting to reinstall packages, which caused a hang

## Solution Applied
✓ Removed the `pip install` command (packages are already installed in `.venv`)
✓ Removed Google Colab-specific code (`files.upload()`, `files.download()`)
✓ Updated notebook to work with local file system

## How to Restart the Kernel in VS Code

### Option 1: Quick Restart (Recommended)
1. Open the notebook
2. Click the **Restart Kernel** button in the toolbar (circular arrow icon)
3. Or: Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
4. Type "Jupyter: Restart Kernel" and press Enter

### Option 2: Clear & Restart
1. Press `Ctrl+Shift+P` / `Cmd+Shift+P`
2. Type "Jupyter: Clear All Cells and Run All"
3. This clears output and restarts from scratch

### Option 3: Change Kernel (if above doesn't work)
1. Click the kernel selector in the top-right (usually shows Python version)
2. Select "Python 3.14.1 (/Users/Apple/Downloads/Data/HackathonA4/.venv/bin/python)"
3. Click "Restart" when prompted

## Environment Check
The `.venv` Python is configured and includes:
- ✓ pandas
- ✓ numpy
- ✓ matplotlib
- ✓ seaborn
- ✓ scikit-learn
- ✓ xgboost
- ✓ shap
- ✓ vaderSentiment
- ✓ imbalanced-learn

## Next Steps
1. Restart the kernel using one of the options above
2. Run the notebook from the top
3. Cell 1 should now show "✓ Environment ready. All dependencies pre-installed."
4. The notebook will run all the way through without hanging

## If Issues Persist
If the kernel still hangs:
1. Open Terminal: `Ctrl+` ` (backtick)
2. Run: `pkill -f ipykernel`
3. Restart VS Code
4. Reopen the notebook
