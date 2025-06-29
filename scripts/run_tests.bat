@echo off
REM Test Runner Script for Windows
REM Runs the full test suite with coverage and shows a clean summary

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Please run setup first.
    echo    python -m venv venv
    echo    venv\Scripts\activate
    echo    pip install -r requirements.txt
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the test script
python scripts\run_tests_with_coverage.py

REM Exit with the same code as the test script
exit /b %ERRORLEVEL% 