#!/bin/bash
# Installation script for time series forecasting project dependencies

echo "Installing dependencies for Time Series Forecasting project..."
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda to install packages..."
    conda install pandas pyarrow numpy scikit-learn matplotlib seaborn -y
    if [ $? -eq 0 ]; then
        echo "✓ Packages installed successfully with conda"
        exit 0
    else
        echo "Conda installation failed, trying pip..."
    fi
fi

# Fallback to pip
if command -v pip &> /dev/null; then
    echo "Using pip to install packages..."
    pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
    if [ $? -eq 0 ]; then
        echo "✓ Packages installed successfully with pip"
        exit 0
    fi
fi

# Try pip3
if command -v pip3 &> /dev/null; then
    echo "Using pip3 to install packages..."
    pip3 install pandas pyarrow numpy scikit-learn matplotlib seaborn
    if [ $? -eq 0 ]; then
        echo "✓ Packages installed successfully with pip3"
        exit 0
    fi
fi

echo "❌ Installation failed. Please install packages manually."
echo "See SETUP.md for detailed instructions."
