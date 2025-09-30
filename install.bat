@echo off
REM ICM Cardiovascular App Installation Script
REM This script sets up the environment and installs the package

echo ===================================
echo ICM Cardiovascular App Installer
echo ===================================

REM Check if conda is installed
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda is not installed or not in PATH.
    echo Please install Anaconda or Miniconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Conda found

REM Ask user for environment name
set /p ENV_NAME="Enter conda environment name (default: icm-cardio): "
if "%ENV_NAME%"=="" set ENV_NAME=icm-cardio

echo Creating conda environment: %ENV_NAME%

REM Create conda environment with Python 3.9
conda create -n %ENV_NAME% python=3.9 -y

if %errorlevel% neq 0 (
    echo Failed to create conda environment
    pause
    exit /b 1
)

echo Conda environment created successfully

REM Activate environment
echo Activating environment
call conda activate %ENV_NAME%

if %errorlevel% neq 0 (
    echo Failed to activate conda environment
    pause
    exit /b 1
)

echo Environment activated

REM Install the package in development mode
echo Installing ICM Cardiovascular package...
pip install -e .

if %errorlevel% neq 0 (
    echo Failed to install package
    pause
    exit /b 1
)

echo Package installed successfully

REM Test installation
echo Testing installation...
python -c "import icm_cardiovascular; print('Package import successful')"

if %errorlevel% neq 0 (
    echo Package import failed
    pause
    exit /b 1
)

echo.
echo Installation completed successfully!
echo.
echo To use the applications:
echo 1. Activate the environment: conda activate %ENV_NAME%
echo 2. Choose one of these methods:
echo.
echo    Method 1 - Direct script execution:
echo    python database_creator.py
echo    python icm_cardiovascular/icm_database.py
echo.
echo    Method 2 - Command line tools:
echo    icm-database-creator
echo    icm-database-viewer
echo.
echo See README.md for detailed usage instructions
pause
