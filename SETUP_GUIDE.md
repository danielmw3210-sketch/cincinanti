# AI Trading System Setup Guide

## Prerequisites

The system has been set up with all necessary dependencies. Here's what you need to know:

## Virtual Environment

A Python virtual environment has been created with all required packages installed:

- **Location**: `/workspace/venv/`
- **Python Version**: 3.13
- **Activation**: Run `source venv/bin/activate` or use the provided script

## Quick Start

### 1. Activate the Environment
```bash
# Option 1: Use the activation script
./activate_env.sh

# Option 2: Manual activation
source venv/bin/activate
```

### 2. Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, tensorflow, optuna; print('All packages installed successfully!')"
```

### 3. Run the System
```bash
# Start the main trading system
python main.py

# Run backtesting
python scripts/backtest.py --pair BTC/USD --start 2023-01-01 --end 2023-12-31

# Run the trader
python run_trader.py
```

## Installed Packages

### Core Dependencies
- **pandas** (2.3.2): Data manipulation and analysis
- **numpy** (2.3.3): Numerical computing
- **scikit-learn** (1.7.2): Machine learning algorithms
- **scipy** (1.16.2): Scientific computing

### Machine Learning
- **xgboost** (3.0.5): Gradient boosting framework
- **lightgbm** (4.6.0): Light gradient boosting machine
- **tensorflow** (2.20.0): Deep learning framework
- **keras** (3.11.3): High-level neural networks API

### Optimization & Analysis
- **optuna** (4.5.0): Hyperparameter optimization
- **imbalanced-learn** (0.14.0): Imbalanced dataset handling

### Data & Visualization
- **matplotlib** (3.10.6): Plotting library
- **seaborn** (0.13.2): Statistical data visualization
- **ta** (0.11.0): Technical analysis library

### Trading & APIs
- **ccxt** (4.5.4): Cryptocurrency exchange trading library
- **requests** (2.32.5): HTTP library
- **websocket-client** (1.8.0): WebSocket client

### Web Framework
- **fastapi** (0.116.1): Modern web framework
- **uvicorn** (0.35.0): ASGI server
- **pydantic** (2.11.9): Data validation

### Utilities
- **loguru** (0.7.3): Logging
- **python-dotenv** (1.1.1): Environment variables
- **schedule** (1.2.2): Job scheduling

## New Features Available

### 1. Enhanced Feature Engineering
- 100+ technical indicators across multiple timeframes
- Market microstructure features
- Cross-timeframe analysis
- Advanced feature engineering with lagged features and rolling statistics

### 2. Advanced ML Models
- XGBoost with hyperparameter optimization
- LightGBM with balanced class weights
- LSTM neural networks for time series
- Ensemble methods with dynamic model selection

### 3. Sophisticated Risk Management
- Kelly Criterion position sizing
- Dynamic risk multipliers
- Portfolio heat management
- Correlation-based position adjustment

### 4. Comprehensive Performance Analysis
- Detailed performance metrics
- Risk analysis (VaR, CVaR, Sharpe ratio)
- Model performance tracking
- Market regime analysis

### 5. Hyperparameter Optimization
- Optuna-based optimization
- Time series cross-validation
- Feature selection optimization
- Model-specific parameter tuning

## Configuration

### Environment Variables
Create a `.env` file with your configuration:

```env
# Kraken API
KRAKEN_API_KEY=your_api_key
KRAKEN_SECRET_KEY=your_secret_key
KRAKEN_SANDBOX=true

# Trading Parameters
DEFAULT_PAIR=BTC/USD
INITIAL_BALANCE=1000.0
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02

# AI Model
MODEL_TYPE=ensemble
RETRAIN_INTERVAL=24
FEATURE_WINDOW=100

# Safety
EMERGENCY_STOP=false
MAX_DAILY_LOSS=0.05
```

## Usage Examples

### Training Models
```python
from ai_trader import AITrader
from market_analyzer import MarketAnalyzer
from kraken_client import KrakenClient

# Initialize components
client = KrakenClient()
analyzer = MarketAnalyzer(client)
trader = AITrader(analyzer)

# Train models
accuracies = trader.train_models("BTC/USD")
print(f"Model accuracies: {accuracies}")
```

### Hyperparameter Optimization
```python
from hyperparameter_optimizer import HyperparameterOptimizer

# Initialize optimizer
optimizer = HyperparameterOptimizer(trader, analyzer)

# Optimize all models
results = optimizer.optimize_all_models("BTC/USD", n_trials=100)
print(optimizer.get_optimization_summary())
```

### Performance Analysis
```python
from performance_analyzer import PerformanceAnalyzer

# Initialize analyzer
perf_analyzer = PerformanceAnalyzer()

# Add trade data
perf_analyzer.add_trade(trade_data)
perf_analyzer.add_portfolio_snapshot(portfolio_data)

# Generate report
report = perf_analyzer.generate_performance_report()
print(report)
```

## File Structure

```
/workspace/
├── venv/                          # Virtual environment
├── activate_env.sh               # Activation script
├── requirements.txt              # Dependencies
├── main.py                       # Main application
├── ai_trader.py                  # Enhanced AI trader
├── market_analyzer.py            # Enhanced market analyzer
├── risk_manager.py               # Enhanced risk manager
├── trading_executor.py           # Trading executor
├── performance_analyzer.py       # Performance analysis
├── hyperparameter_optimizer.py   # Hyperparameter optimization
├── config.py                     # Configuration
├── kraken_client.py              # Kraken API client
├── scripts/
│   ├── backtest.py               # Backtesting script
│   └── install.sh                # Installation script
├── models/                       # Saved models directory
├── optimization_results/         # Optimization results
└── ALGORITHM_IMPROVEMENTS.md     # Detailed improvements
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the virtual environment is activated
2. **API Errors**: Check your Kraken API credentials
3. **Memory Issues**: Reduce feature window size or use smaller datasets
4. **Model Training**: Ensure sufficient data is available

### Getting Help

1. Check the logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure your API credentials are valid
4. Check the configuration settings

## Next Steps

1. **Configure API Keys**: Set up your Kraken API credentials
2. **Run Backtesting**: Test the system with historical data
3. **Optimize Models**: Run hyperparameter optimization
4. **Monitor Performance**: Use the performance analyzer
5. **Deploy**: Start live trading (with caution!)

## Safety Reminders

- **Start with Paper Trading**: Test thoroughly before live trading
- **Set Conservative Limits**: Use small position sizes initially
- **Monitor Closely**: Keep an eye on performance and risk metrics
- **Emergency Stop**: Know how to stop trading immediately if needed

The system is now ready to use with all the enhanced features and improvements!