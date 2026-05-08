@echo off

REM Setup script for reproducing the mimic-legacy environment
REM This script provides multiple options for setting up the environment

echo Setting up mimic-legacy environment...
echo.

REM Check if conda is available
conda --version >nul 2>&1
if %errorlevel% == 0 (
    echo Conda detected. Setting up environment...

    REM Create environment from YAML file (recommended)
    echo Creating environment from environment.yml...
    conda env create -f environment.yml

    echo.
    echo Environment created successfully!
    echo To activate: conda activate mimic_legacy
    echo.

) else (
    echo Conda not found. Please install Miniconda or Anaconda first:
    echo https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
    echo.
    echo Alternatively, you can use pip to install packages:
    echo pip install -r requirements.txt
)

echo Setup complete!
echo.
echo For more information, see the README.md file.

pause

