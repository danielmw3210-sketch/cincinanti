from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import asyncio
import logging

from models.forecast_model import CryptoForecastModel
from data.data_processor import DataProcessor
from data.crypto_data_fetcher import CryptoDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Price Forecast API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and data processor instances
forecast_model = None
data_processor = None
crypto_fetcher = None

class ForecastRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    lookback_hours: int = 24

class ForecastResponse(BaseModel):
    symbol: str
    current_price: float
    forecasts: Dict[str, Dict[str, float]]  # {horizon: {price: float, confidence: float}}
    timestamp: str
    model_type: str

@app.on_event("startup")
async def startup_event():
    """Initialize the model and data processor on startup"""
    global forecast_model, data_processor, crypto_fetcher
    
    try:
        logger.info("Initializing crypto forecast system...")
        
        # Initialize components
        data_processor = DataProcessor()
        crypto_fetcher = CryptoDataFetcher()
        forecast_model = CryptoForecastModel()
        
        # Load or train initial model
        model_path = "models/saved/crypto_forecast_model.pkl"
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            forecast_model.load_model(model_path)
        else:
            logger.info("Training new model with sample data...")
            await train_initial_model()
        
        logger.info("Crypto forecast system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

async def train_initial_model():
    """Train the model with initial data"""
    try:
        # Fetch some sample data for initial training
        sample_data = await crypto_fetcher.fetch_historical_data("BTC/USDT", "1m", 24)
        if sample_data is not None and len(sample_data) > 100:
            processed_data = data_processor.process_data(sample_data)
            forecast_model.train(processed_data)
            forecast_model.save_model("models/saved/crypto_forecast_model.pkl")
            logger.info("Initial model training completed")
        else:
            logger.warning("Insufficient data for initial training, using default model")
    except Exception as e:
        logger.error(f"Initial training failed: {e}")

@app.get("/")
async def root():
    return {"message": "Crypto Price Forecast API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": forecast_model is not None}

@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    """Get price forecasts for the specified cryptocurrency"""
    try:
        if forecast_model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Fetch recent data
        logger.info(f"Fetching data for {request.symbol}")
        raw_data = await crypto_fetcher.fetch_historical_data(
            request.symbol, 
            request.timeframe, 
            request.lookback_hours
        )
        
        if raw_data is None or len(raw_data) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data for forecasting")
        
        # Process data
        processed_data = data_processor.process_data(raw_data)
        
        # Get current price
        current_price = float(processed_data['close'].iloc[-1])
        
        # Generate forecasts
        forecasts = forecast_model.predict(processed_data)
        
        # Format response
        response = ForecastResponse(
            symbol=request.symbol,
            current_price=current_price,
            forecasts=forecasts,
            timestamp=datetime.now().isoformat(),
            model_type=forecast_model.model_type
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with fresh data"""
    try:
        if forecast_model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        logger.info("Starting model retraining...")
        
        # Fetch fresh data
        raw_data = await crypto_fetcher.fetch_historical_data("BTC/USDT", "1m", 48)
        if raw_data is None or len(raw_data) < 200:
            raise HTTPException(status_code=400, detail="Insufficient data for retraining")
        
        # Process and train
        processed_data = data_processor.process_data(raw_data)
        forecast_model.train(processed_data)
        forecast_model.save_model("models/saved/crypto_forecast_model.pkl")
        
        logger.info("Model retraining completed")
        return {"message": "Model retrained successfully", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_available_symbols():
    """Get list of available cryptocurrency symbols"""
    return {
        "symbols": [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
            "XRP/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)