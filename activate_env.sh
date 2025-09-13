#!/bin/bash
# Activation script for the AI trading system virtual environment

echo "Activating AI Trading System virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo ""
echo "Available packages:"
echo "- pandas: $(python -c 'import pandas; print(pandas.__version__)')"
echo "- numpy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "- scikit-learn: $(python -c 'import sklearn; print(sklearn.__version__)')"
echo "- xgboost: $(python -c 'import xgboost; print(xgboost.__version__)')"
echo "- lightgbm: $(python -c 'import lightgbm; print(lightgbm.__version__)')"
echo "- tensorflow: $(python -c 'import tensorflow; print(tensorflow.__version__)')"
echo "- optuna: $(python -c 'import optuna; print(optuna.__version__)')"
echo ""
echo "To run the trading system:"
echo "  python main.py"
echo "  python run_trader.py"
echo "  python scripts/backtest.py"
echo ""
echo "To deactivate: deactivate"