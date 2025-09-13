# AI Crypto Trader Dashboard

A real-time web dashboard for monitoring and controlling the AI-powered cryptocurrency trading system.

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
# or use the activation script
./activate_env.sh
```

### 2. Start the Dashboard
```bash
python start_dashboard.py
```

### 3. Access the Dashboard
- **Main Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 🎯 Features

### Real-time Trading Dashboard
- **Portfolio Overview**: Total balance, available balance, daily P&L
- **Current Positions**: Live view of open trading positions
- **AI Signals**: Real-time AI trading recommendations
- **Performance Metrics**: Win rate, profit factor, Sharpe ratio
- **Risk Metrics**: Portfolio heat, drawdown, VaR

### Trading Controls
- **Start/Stop Trading**: Toggle automated trading on/off
- **Real-time Updates**: Auto-refresh every 30 seconds
- **Manual Refresh**: Force refresh dashboard data

### Advanced Analytics
- **Multi-timeframe Analysis**: 1m, 5m, 15m, 60m data
- **100+ Technical Indicators**: Comprehensive market analysis
- **AI Model Ensemble**: XGBoost, LightGBM, LSTM, Voting Classifier
- **Risk Management**: Kelly Criterion, dynamic position sizing

## 📊 Dashboard Components

### Portfolio Stats
- Total Balance
- Available Balance  
- Daily P&L
- Win Rate

### Current Positions
- Trading pair
- Position size and type
- Entry price vs current price
- Unrealized P&L

### AI Signals
- Real-time buy/sell/hold recommendations
- Confidence scores
- Reasoning explanations
- Multi-pair analysis

### Performance Metrics
- Total trades executed
- Win rate percentage
- Profit factor
- Sharpe ratio

### Risk Metrics
- Portfolio heat (risk exposure)
- Maximum drawdown
- Value at Risk (VaR 95%)
- Current drawdown

## 🔧 API Endpoints

### Dashboard Data
- `GET /api/trading/dashboard` - Complete dashboard data
- `GET /api/trading/status` - Trading status
- `GET /api/health` - System health check

### Trading Controls
- `POST /api/trading/start` - Start automated trading
- `POST /api/trading/stop` - Stop automated trading
- `GET /api/trading/signal/{pair}` - Get AI signal for pair

### Analytics
- `GET /api/trading/performance` - Performance metrics
- `GET /api/trading/risk` - Risk metrics
- `POST /api/trading/train` - Train AI models

## 🛠️ Technical Details

### Backend
- **FastAPI**: Modern Python web framework
- **Real-time Data**: Live market data integration
- **AI Models**: Advanced machine learning ensemble
- **Risk Management**: Sophisticated position sizing

### Frontend
- **HTML/CSS/JavaScript**: Lightweight, responsive design
- **Tailwind CSS**: Modern styling framework
- **Lucide Icons**: Beautiful icon set
- **Real-time Updates**: Auto-refresh capabilities

### Data Sources
- **Kraken Pro API**: Live market data
- **Multi-timeframe**: 1m, 5m, 15m, 60m intervals
- **Technical Indicators**: 100+ indicators
- **Market Metadata**: Session analysis, volatility regimes

## 🔒 Security Features

- **API Key Management**: Secure credential handling
- **Risk Limits**: Maximum position sizes and daily losses
- **Portfolio Heat**: Total risk exposure monitoring
- **Stop Losses**: Automatic risk protection

## 📈 Performance Monitoring

### Real-time Metrics
- Live P&L tracking
- Position monitoring
- AI signal confidence
- Risk exposure levels

### Historical Analysis
- Performance trends
- Win/loss ratios
- Drawdown analysis
- Sharpe ratio tracking

## 🚨 Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if virtual environment is activated
   - Verify all dependencies are installed
   - Check for port conflicts (8000)

2. **No data showing**
   - Verify API keys are configured
   - Check network connectivity
   - Review server logs for errors

3. **Trading not working**
   - Ensure API keys have trading permissions
   - Check account balance
   - Verify risk settings

### Debug Commands
```bash
# Test dashboard
python test_dashboard.py

# Check server status
curl http://localhost:8000/api/health

# View server logs
tail -f logs/trading.log
```

## 📝 Configuration

### Environment Variables
Create a `.env` file with:
```env
KRAKEN_API_KEY=your_api_key
KRAKEN_SECRET=your_secret
KRAKEN_SANDBOX=true
```

### Trading Settings
- Default pair: BTC/USD
- Risk per trade: 2%
- Max position size: 10%
- Max daily loss: 5%

## 🎯 Next Steps

1. **Configure API Keys**: Set up your Kraken Pro credentials
2. **Test with Demo Mode**: Use sandbox mode for testing
3. **Monitor Performance**: Watch the dashboard for insights
4. **Adjust Settings**: Fine-tune risk parameters
5. **Scale Up**: Gradually increase position sizes

## 📞 Support

For issues or questions:
- Check the logs in `logs/` directory
- Review the API documentation at `/docs`
- Test individual endpoints with the test script

---

**Happy Trading! 🚀📈**