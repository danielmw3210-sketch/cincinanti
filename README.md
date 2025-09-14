# 🚀 Crypto Price Forecast System

A fast, robust baseline for cryptocurrency price forecasting using XGBoost with multi-horizon regression (15/30/60 minutes) and a modern React frontend with ticker-style display.

## ✨ Features

- **AI-Powered Forecasting**: XGBoost-based multi-horizon price prediction (15/30/60 min)
- **Robust Fallback**: Automatically falls back to sklearn's HistGradientBoosting if XGBoost unavailable
- **Real-time Data**: Live cryptocurrency data fetching with sample data fallback
- **Modern UI**: React frontend with ticker-style display and real-time updates
- **Feature Engineering**: Advanced lag features, rolling statistics, and technical indicators
- **FastAPI Backend**: High-performance API with automatic documentation
- **Auto-refresh**: Automatic forecast updates every 30 seconds

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  Data Sources   │
│                 │    │                 │    │                 │
│ • Ticker Display │◄──►│ • XGBoost Model │◄──►│ • Binance API   │
│ • Symbol Selector│    │ • Data Processor│    │ • Sample Data   │
│ • Forecast Panel │    │ • Crypto Fetcher│    │ • CCXT Library  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: One-Command Setup
```bash
./start.sh
```

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
pip install -r ../requirements.txt
python main.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 📊 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /forecast` - Get price forecasts
- `GET /symbols` - Available cryptocurrency symbols
- `GET /model/info` - Model information
- `POST /retrain` - Retrain the model
- `GET /data/summary` - Data quality summary

### Example API Usage

```bash
# Get forecast for Bitcoin
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC/USDT", "lookback_hours": 24}'

# Check health
curl "http://localhost:8000/health"
```

## 🎯 Model Details

### XGBoost Configuration
- **Algorithm**: XGBoost Regressor
- **Horizons**: 15, 30, 60 minutes
- **Features**: 50+ engineered features including:
  - Price lags (1, 5, 15, 30, 60 min)
  - Rolling statistics (5, 15, 30, 60 min windows)
  - Technical indicators (RSI, Bollinger Bands, MACD)
  - Volume features and volatility measures

### Fallback Model
- **Algorithm**: HistGradientBoostingRegressor
- **Activation**: When XGBoost is not available
- **Performance**: Similar accuracy with different training characteristics

## 🎨 Frontend Features

### Ticker Display
- Real-time price updates
- Multi-horizon forecast cards
- Confidence indicators
- Animated transitions

### Symbol Selector
- Searchable dropdown
- 10+ major cryptocurrencies
- Symbol descriptions and icons

### Forecast Panel
- Detailed prediction information
- Price change calculations
- Model type and data quality indicators

## 🔧 Configuration

### Backend Settings (`backend/config/settings.py`)
```python
# Forecast horizons (minutes)
forecast_horizons = [15, 30, 60]

# Feature engineering
lag_periods = [1, 5, 15, 30, 60]
rolling_windows = [5, 15, 30, 60]

# Data requirements
min_data_points = 100
default_lookback_hours = 24
```

### Environment Variables
```bash
# Optional: Set API URL for frontend
REACT_APP_API_URL=http://localhost:8000

# Optional: Enable debug mode
DEBUG=true
```

## 📈 Performance

- **Training Time**: ~30 seconds for 24 hours of 1-minute data
- **Prediction Time**: <100ms per forecast
- **Memory Usage**: ~200MB for model + data
- **Accuracy**: Varies by market conditions and data quality

## 🛠️ Development

### Project Structure
```
├── backend/
│   ├── config/          # Configuration settings
│   ├── data/            # Data processing and fetching
│   ├── models/          # ML models and training
│   ├── schemas/         # Pydantic models
│   ├── utils/           # Utilities and logging
│   └── main.py          # FastAPI application
├── frontend/
│   ├── public/          # Static assets
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/    # API services
│   │   └── App.js       # Main application
│   └── package.json
└── requirements.txt     # Python dependencies
```

### Adding New Features

1. **New Forecast Horizon**: Update `forecast_horizons` in settings
2. **New Features**: Add to `_create_features()` in `forecast_model.py`
3. **New Symbols**: Update `supported_symbols` in settings
4. **New UI Components**: Add to `frontend/src/components/`

## 🔍 Troubleshooting

### Common Issues

1. **XGBoost Installation Error**
   ```bash
   pip install xgboost
   # or
   conda install -c conda-forge xgboost
   ```

2. **Port Already in Use**
   ```bash
   # Change ports in settings.py and package.json
   # Backend: settings.port = 8001
   # Frontend: "start": "PORT=3001 react-scripts start"
   ```

3. **CORS Issues**
   - Ensure frontend URL is in `cors_origins` in settings
   - Check that backend is running on correct port

4. **Model Not Loading**
   - Check `backend/models/saved/` directory exists
   - Verify file permissions
   - Check logs for specific error messages

### Logs
- Backend logs: Console output
- Frontend logs: Browser developer console
- API logs: FastAPI automatic logging

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Open an issue on GitHub

---

**Note**: This is a baseline implementation for educational and research purposes. Cryptocurrency trading involves significant risk, and this tool should not be used as the sole basis for trading decisions.