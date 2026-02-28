# Quick Start Guide

## Current Issue

You're getting a `ModuleNotFoundError: No module named 'pandas'` because the required packages aren't installed in your conda environment.

## Solution

### Step 1: Install Dependencies

**Option A - Using the installation script:**
```bash
./install_dependencies.sh
```

**Option B - Manual installation with conda:**
```bash
conda install pandas pyarrow numpy scikit-learn matplotlib seaborn -y
```

**Option C - Manual installation with pip:**
```bash
pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
```

### Step 2: Verify Installation

```bash
python -c "import pandas; print('✓ pandas installed')"
python -c "import numpy; print('✓ numpy installed')"
python -c "import sklearn; print('✓ scikit-learn installed')"
```

### Step 3: Run Quick Training

Once packages are installed:

```bash
python train_quick.py
```

This will train on a sample of the data (1000 time series) to verify everything works.

### Step 4: Full Training (Optional)

For full training on all data:

```bash
python train.py
```

**Note:** This will take 30+ minutes due to the large dataset size.

### Step 5: Generate Predictions

```bash
python predict.py
```

This creates `submission.csv` ready for Kaggle submission.

## Troubleshooting

If you encounter network issues:
- Check your internet connection
- Try using a VPN if behind a firewall
- Install packages from a machine with internet access, then copy the environment

## Project Files

All code is ready:
- ✅ `train_quick.py` - Quick training script
- ✅ `train.py` - Full training script  
- ✅ `predict.py` - Prediction script
- ✅ `feature_engineering.py` - Feature creation
- ✅ `models.py` - Model implementations
- ✅ `utils.py` - Utility functions
- ✅ `config.py` - Configuration

You just need to install the dependencies!
