#!/bin/bash

# Setup script for reproducing the mimic-legacy environment
# This script provides multiple options for setting up the environment

echo "Setting up mimic-legacy environment..."
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda detected. Setting up environment..."

    # Create environment from YAML file (recommended)
    echo "Creating environment from mimic-legacy_environment.yml..."
    conda env create -f mimic-legacy_environment.yml

    echo ""
    echo "Environment created successfully!"
    echo "To activate: conda activate mimic-legacy"
    echo ""

else
    echo "Conda not found. Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
    echo ""
    echo "Alternatively, you can use pip to install packages:"
    echo "pip install -r mimic-legacy_requirements.txt"
fi

echo "Setup complete!"
echo ""
echo "For more information, see the README.md file."

