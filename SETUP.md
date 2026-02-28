# Setup Instructions

## Installation Issue

There appears to be a network connectivity issue preventing automatic package installation. Here are several ways to install the required packages:

## Option 1: Using Conda (Recommended)

If you have conda installed and internet access:

```bash
conda install pandas pyarrow numpy scikit-learn matplotlib seaborn -y
```

Or create a new conda environment:

```bash
conda create -n ts_forecast python=3.9 pandas pyarrow numpy scikit-learn matplotlib seaborn -y
conda activate ts_forecast
```

## Option 2: Using pip (if network is available)

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
```

## Option 3: Check if packages are already installed

Try running:

```bash
python -c "import sys; print(sys.path)"
```

This will show where Python looks for packages. You might need to install packages in a specific location.

## Option 4: Use a different Python environment

If you have multiple Python installations:

```bash
# Check available Python versions
which -a python
which -a python3

# Try with python3 explicitly
python3 -m pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
```

## Required Packages

The project requires:
- pandas (>=2.0.0)
- pyarrow (>=12.0.0) - for reading parquet files
- numpy (>=1.24.0)
- scikit-learn (>=1.3.0)
- matplotlib (>=3.7.0) - optional, for visualization
- seaborn (>=0.12.0) - optional, for visualization

## Verify Installation

After installing, verify with:

```bash
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"
```

## Troubleshooting Network Issues

If you're behind a firewall or proxy:
1. Configure pip/conda to use proxy settings
2. Use offline installation if packages are available locally
3. Check DNS settings if domain resolution is failing

## Quick Test

Once packages are installed, test the quick training script:

```bash
python train_quick.py
```
