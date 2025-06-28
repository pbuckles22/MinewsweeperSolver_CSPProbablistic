#!/bin/bash

# Minesweeper CSP & Probabilistic Solver - Mac Installation Script
# This script sets up the virtual environment and installs dependencies
# Designed to be easily adaptable for Windows (PowerShell) later

set -e  # Exit on any error

echo "🚀 Setting up Minesweeper CSP & Probabilistic Solver on Mac..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or later."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📦 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Installation complete!"
echo ""
echo "🎯 To activate the environment in the future:"
echo "   source venv/bin/activate"
echo ""
echo "🧪 To run tests:"
echo "   python -m pytest tests/ -v"
echo ""
echo "🚀 To run CSP solver:"
echo "   python -c \"from src.core.minesweeper_env import MinesweeperEnv; from src.core.csp_agent import CSPAgent; env = MinesweeperEnv(height=9, width=9, num_mines=10); agent = CSPAgent(); agent.solve(env)\""
