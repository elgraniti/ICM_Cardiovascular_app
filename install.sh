#!/bin/bash

# ICM Cardiovascular App Installation Script
# This script sets up the environment and installs the package

echo "==================================="
echo "ICM Cardiovascular App Installer"
echo "==================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found"

# Ask user for environment name
read -p "Enter conda environment name (default: icm-cardio): " ENV_NAME
ENV_NAME=${ENV_NAME:-icm-cardio}

echo "📦 Creating conda environment: $ENV_NAME"

# Create conda environment with Python 3.9
conda create -n $ENV_NAME python=3.9 -y

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment"
    exit 1
fi

echo "✅ Conda environment created successfully"

# Activate environment
echo "🔄 Activating environment"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

echo "✅ Environment activated"

# Install the package in development mode
echo "📦 Installing ICM Cardiovascular package..."
pip install -e .

if [ $? -ne 0 ]; then
    echo "❌ Failed to install package"
    exit 1
fi

echo "✅ Package installed successfully"

# Test installation
echo "🧪 Testing installation..."
python -c "import icm_cardiovascular; print('✅ Package import successful')"

if [ $? -ne 0 ]; then
    echo "❌ Package import failed"
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To use the applications:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Choose one of these methods:"
echo ""
echo "   Method 1 - Direct script execution:"
echo "   python database_creator.py"
echo "   python icm_cardiovascular/icm_database.py"
echo ""
echo "   Method 2 - Command line tools:"
echo "   icm-database-creator"
echo "   icm-database-viewer"
echo ""
echo "📖 See README.md for detailed usage instructions"
