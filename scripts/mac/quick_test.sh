#!/bin/bash

# Minesweeper CSP & Probabilistic Solver - Mac Quick Test Script
# This script runs essential tests to verify the installation

set -e  # Exit on any error

echo "🧪 Running Minesweeper CSP & Probabilistic Solver tests..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run install_and_run.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Run CSP-specific tests
echo "🧪 Running CSP tests..."
python -m pytest tests/unit/csp/ -v

# Run core environment tests
echo "🧪 Running core environment tests..."
python -m pytest tests/unit/core/ -v

# Run functional tests
echo "🧪 Running functional tests..."
python -m pytest tests/functional/ -v

echo "✅ All tests completed successfully!"
echo ""
echo "🎯 CSP solver is ready to use!"
