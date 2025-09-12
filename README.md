# AI Crypto Trader for Kraken Pro

A sophisticated AI-powered cryptocurrency trading bot that integrates with Kraken Pro exchange. This system uses machine learning models to analyze market data, generate trading signals, and execute trades with comprehensive risk management.

## 🚀 Features

- **AI-Powered Trading**: Ensemble ML models (Random Forest, Gradient Boosting, Logistic Regression)
- **Real-time Market Analysis**: Technical indicators, pattern recognition, sentiment analysis
- **Risk Management**: Position sizing, stop losses, take profits, drawdown protection
- **Kraken Pro Integration**: Full API integration with sandbox and live trading support
- **Monitoring & Alerts**: Comprehensive logging, webhook alerts, performance tracking
- **Portfolio Management**: Real-time portfolio tracking and performance metrics

## 📋 Prerequisites

- Python 3.8 or higher
- Kraken Pro API credentials
- Basic understanding of cryptocurrency trading

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-crypto-trader
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Kraken API credentials
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p logs models
   ```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Kraken API Configuration
KRAKEN_API_KEY=your_api_key_here
KRAKEN_SECRET_KEY=your_secret_key_here
KRAKEN_SANDBOX=true

# Trading Configuration
DEFAULT_PAIR=BTC/USD
INITIAL_BALANCE=1000
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02

# AI Model Configuration
MODEL_TYPE=ensemble
RETRAIN_INTERVAL=24
FEATURE_WINDOW=100

# Monitoring
LOG_LEVEL=INFO
WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_EMAIL=your-email@example.com

# Safety
EMERGENCY_STOP=false
MAX_DAILY_LOSS=0.05
```

### Kraken API Setup

1. **Create Kraken Pro Account**: Sign up at [kraken.com](https://www.kraken.com)
2. **Generate API Keys**: 
   - Go to Account → API
   - Create new API key
   - Enable trading permissions
   - Copy API Key and Secret
3. **Test in Sandbox**: Start with `KRAKEN_SANDBOX=true` for testing

## 🚀 Usage

### 1. Single Analysis
```bash
python main.py analyze BTC/USD
```

### 2. Check System Status
```bash
python main.py status
```

### 3. Manual Trade
```bash
python main.py trade BTC/USD buy 0.01
```

### 4. Start Automated Trading
```bash
python main.py start
```

## 📊 Trading Strategy

### AI Model Architecture
- **Ensemble Approach**: Combines multiple ML models for robust predictions
- **Feature Engineering**: 50+ technical indicators and market features
- **Signal Generation**: Buy/Sell/Hold recommendations with confidence scores
- **Risk Adjustment**: Position sizing based on signal confidence and market conditions

### Technical Indicators Used
- Moving Averages (SMA, EMA)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- Volume indicators
- Support/Resistance levels
- Candlestick patterns

### Risk Management
- **Position Sizing**: Kelly Criterion-based sizing
- **Stop Losses**: ATR-based dynamic stop losses
- **Take Profits**: Risk-reward ratio of 2:1
- **Daily Loss Limits**: Configurable maximum daily loss
- **Drawdown Protection**: Automatic trading halt on excessive drawdown

## 📈 Monitoring & Alerts

### Performance Metrics
- Total return and daily P&L
- Maximum drawdown tracking
- Win rate and profit factor
- Sharpe ratio and risk-adjusted returns

### Alert Types
- High daily loss alerts
- Drawdown warnings
- Consecutive loss notifications
- API connectivity issues
- Model performance degradation

### Logging
- Comprehensive trade logging
- AI signal generation logs
- Error tracking and debugging
- Performance reports

## 🔧 Advanced Configuration

### Model Customization
```python
# In ai_trader.py, modify model parameters
self.models = {
    'random_forest': RandomForestClassifier(
        n_estimators=200,  # Increase for better accuracy
        max_depth=15,      # Adjust complexity
        random_state=42
    ),
    # ... other models
}
```

### Risk Parameters
```python
# In risk_manager.py, adjust risk settings
self.risk_per_trade = 0.01      # 1% risk per trade
self.max_position_size = 0.05   # 5% max position
self.max_daily_loss = 0.03      # 3% daily loss limit
```

### Trading Pairs
```python
# Add more trading pairs
self.trading_pairs = [
    'BTC/USD', 'ETH/USD', 'ADA/USD', 
    'DOT/USD', 'LINK/USD'
]
```

## 🛡️ Safety Features

### Emergency Controls
- **Emergency Stop**: Immediate halt of all trading
- **Daily Loss Limits**: Automatic shutdown on excessive losses
- **API Error Handling**: Graceful handling of connection issues
- **Sandbox Mode**: Safe testing environment

### Risk Controls
- **Position Limits**: Maximum position size per trade
- **Drawdown Limits**: Trading halt on excessive drawdown
- **Consecutive Loss Limits**: Reduced trading after losses
- **Market Hours**: Optional trading time restrictions

## 📚 API Reference

### Core Classes

#### `KrakenClient`
- `get_account_balance()`: Get account balance
- `get_ticker(pair)`: Get current price
- `place_market_order(pair, side, amount)`: Execute trade
- `get_ohlc_data(pair, interval)`: Get historical data

#### `MarketAnalyzer`
- `collect_market_data(pair)`: Collect OHLC data
- `calculate_technical_indicators(df)`: Calculate indicators
- `detect_patterns(df)`: Detect candlestick patterns
- `get_market_summary(pair)`: Comprehensive market analysis

#### `AITrader`
- `predict_signal(pair)`: Generate AI trading signal
- `train_models(pair)`: Train ML models
- `get_trading_recommendation(pair)`: Get trading advice

#### `RiskManager`
- `calculate_position_size(...)`: Calculate optimal position size
- `validate_trade(...)`: Validate trade parameters
- `check_stop_losses()`: Monitor stop loss levels

#### `TradingExecutor`
- `execute_trade(pair, signal, confidence)`: Execute trades
- `get_portfolio_summary()`: Get portfolio status
- `start_trading_session(pair)`: Begin automated trading

## 🧪 Testing

### Sandbox Testing
1. Set `KRAKEN_SANDBOX=true` in .env
2. Use Kraken sandbox API credentials
3. Test with small amounts
4. Monitor logs for any issues

### Backtesting
```python
# Example backtesting script
from market_analyzer import MarketAnalyzer
from ai_trader import AITrader

analyzer = MarketAnalyzer(kraken_client)
ai_trader = AITrader(analyzer)

# Get historical data
df = analyzer.collect_market_data('BTC/USD', limit=1000)

# Generate signals
signals = []
for i in range(100, len(df)):
    historical_data = df.iloc[:i]
    signal = ai_trader.predict_signal('BTC/USD')
    signals.append(signal)
```

## 🚨 Important Disclaimers

### Risk Warning
- **High Risk**: Cryptocurrency trading involves substantial risk
- **No Guarantees**: Past performance doesn't guarantee future results
- **Loss Potential**: You may lose your entire investment
- **Test First**: Always test in sandbox mode before live trading

### Legal Compliance
- **Regulatory Compliance**: Ensure compliance with local regulations
- **Tax Implications**: Cryptocurrency trading may have tax implications
- **Terms of Service**: Follow Kraken's terms of service
- **Responsible Trading**: Only trade with money you can afford to lose

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**API Connection Errors**
- Verify API credentials
- Check internet connection
- Ensure API permissions are correct

**Model Training Errors**
- Check data availability
- Verify feature calculations
- Monitor memory usage

**Trading Execution Errors**
- Verify account balance
- Check position limits
- Monitor API rate limits

### Getting Help
- Check the logs in `logs/` directory
- Review error messages carefully
- Test in sandbox mode first
- Start with small position sizes

## 🔮 Future Enhancements

- **Advanced ML Models**: Deep learning models, reinforcement learning
- **Multi-Exchange Support**: Binance, Coinbase Pro, etc.
- **Strategy Optimization**: Genetic algorithms, Bayesian optimization
- **Web Interface**: Dashboard for monitoring and control
- **Mobile Alerts**: Push notifications for important events
- **Social Trading**: Copy trading features

---

**Remember**: This is a sophisticated trading system. Start small, test thoroughly, and always prioritize risk management over profits.