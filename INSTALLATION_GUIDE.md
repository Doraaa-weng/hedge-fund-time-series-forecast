# Installation Guide

## Current Situation

There's a **network connectivity issue** preventing automatic package installation. The error "nodename nor servname provided, or not known" indicates DNS resolution problems.

## Solutions

### Option 1: Fix Network Connection (Recommended)

1. **Check your internet connection:**
   ```bash
   ping google.com
   ```

2. **If behind a firewall/VPN:**
   - Ensure VPN is connected
   - Check proxy settings
   - Try disabling VPN temporarily

3. **Once network is working, install packages:**

   **Using conda:**
   ```bash
   conda install pandas pyarrow numpy scikit-learn matplotlib seaborn -y
   ```

   **Using pip:**
   ```bash
   pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
   ```

### Option 2: Use Offline Installation

If you have access to another machine with internet:

1. **On the machine with internet:**
   ```bash
   pip download pandas pyarrow numpy scikit-learn matplotlib seaborn -d ./packages
   ```

2. **Copy the packages folder to this machine**

3. **Install from local files:**
   ```bash
   pip install --no-index --find-links ./packages pandas pyarrow numpy scikit-learn matplotlib seaborn
   ```

### Option 3: Manual Installation Steps

When you have network access, run these commands in order:

```bash
# Navigate to project directory
cd "/Users/doraweng/Documents/Kaggle Project/Hedege Fund-Time series forecast"

# Activate conda environment (if using conda)
conda activate base

# Install packages one by one
conda install pandas -y
conda install pyarrow -y
conda install numpy -y
conda install scikit-learn -y
conda install matplotlib -y
conda install seaborn -y

# Or use pip
pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
```

### Option 4: Create New Conda Environment

```bash
# Create a fresh environment
conda create -n ts_forecast python=3.9 -y
conda activate ts_forecast

# Install packages
conda install pandas pyarrow numpy scikit-learn matplotlib seaborn -y
```

## Verify Installation

After installing, verify with:

```bash
python -c "import pandas; print('✓ pandas:', pandas.__version__)"
python -c "import numpy; print('✓ numpy:', numpy.__version__)"
python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)"
python -c "import pyarrow; print('✓ pyarrow:', pyarrow.__version__)"
```

## Test the Project

Once packages are installed:

```bash
# Quick test
python train_quick.py

# Full training (takes 30+ minutes)
python train.py

# Generate predictions
python predict.py
```

## Troubleshooting Network Issues

### Check DNS:
```bash
nslookup repo.anaconda.com
nslookup pypi.org
```

### Check Proxy Settings:
```bash
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

### Try Different Package Sources:
```bash
# Use conda-forge channel
conda install -c conda-forge pandas pyarrow numpy scikit-learn matplotlib seaborn -y

# Use pip with different index
pip install -i https://pypi.org/simple pandas pyarrow numpy scikit-learn matplotlib seaborn
```

## Required Packages Summary

- **pandas** - Data manipulation
- **pyarrow** - Reading parquet files
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning models
- **matplotlib** - Visualization (optional)
- **seaborn** - Visualization (optional)

All code is ready - you just need to install these packages when network connectivity is restored!
