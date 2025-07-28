# Environment Setup Guide

This guide helps you reproduce the `mimic-legacy` environment used for the EHR Embeddings project.

## Files Created

- `mimic-legacy_environment.yml` - Complete conda environment specification
- `mimic-legacy_requirements.txt` - Pip requirements file
- `setup_environment.sh` - Linux/Mac setup script
- `setup_environment.bat` - Windows setup script

## Quick Setup

### Option 1: Using Conda (Recommended)

1. Install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [Anaconda](https://www.anaconda.com/products/distribution)

2. Run the appropriate setup script:
   - **Windows**: Double-click `setup_environment.bat` or run in command prompt
   - **Linux/Mac**: Run `bash setup_environment.sh` or `chmod +x setup_environment.sh && ./setup_environment.sh`

3. Activate the environment:
   ```bash
   conda activate mimic-legacy
   ```

### Option 2: Using pip

If you prefer using pip instead of conda:

```bash
pip install -r mimic-legacy_requirements.txt
```

## Environment Details

The `mimic-legacy` environment includes:
- Python 3.7.16
- XGBoost 1.6.2
- NumPy, Pandas, Scikit-learn
- Jupyter and other data science packages
- All dependencies with exact versions for reproducibility

## Troubleshooting

- If conda environment creation fails, try updating conda: `conda update conda`
- For permission issues on Windows, run command prompt as administrator
- If packages fail to install, check your internet connection and try again

## Verification

After setup, verify the environment by running:
```python
import xgboost as xgb
print("XGBoost version:", xgb.__version__)
```

## Original Environment Files

- `environment.yml` - Original environment specification
- `requirements.txt` - Project requirements (may not include all conda packages)

