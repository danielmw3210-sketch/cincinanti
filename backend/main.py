from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import asyncio
import os
from datetime import datetime
from contextlib import asynccontextmanager

from models.forecast_model import CryptoForecastModel
from data.data_processor import DataProcessor
from data.crypto_data_fetcher import CryptoDataFetcher
from schemas.models import (
    ForecastRequest, ForecastResponse, ModelInfo, HealthResponse,
    ErrorResponse, RetrainResponse, SymbolListResponse, DataSummary
)
from config.settings import settings
from utils.logger import app_logger

# Global instances
forecast_model: Optional[CryptoForecastModel] = None
data_processor: Optional[DataProcessor] = None
crypto_fetcher: Optional[CryptoDataFetcher] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def startup_event():
    """Initialize the model and data processor on startup"""
    global forecast_model, data_processor, crypto_fetcher
    
    try:
        app_logger.info("Initializing crypto forecast system...")
        
        # Initialize components
        data_processor = DataProcessor()
        crypto_fetcher = CryptoDataFetcher()
        forecast_model = CryptoForecastModel()
        
        # Load or train initial model
        if os.path.exists(settings.model_path):
            app_logger.info("Loading existing model...")
            forecast_model.load_model(settings.model_path)
        else:
            app_logger.info("Training new model with sample data...")
            await train_initial_model()
        
        app_logger.info("Crypto forecast system initialized successfully!")
        
    except Exception as e:
        app_logger.error(f"Failed to initialize system: {e}")
        raise

async def shutdown_event():
    """Cleanup on shutdown"""
    global crypto_fetcher
    if crypto_fetcher:
        await crypto_fetcher.close()
    app_logger.info("Application shutdown completed")

async def train_initial_model():
    """Train the model with initial data"""
    try:
        # Try to fetch real data first
        sample_data = await crypto_fetcher.fetch_historical_data("BTC/USDT", "1m", 24)
        
        # If no real data, generate sample data
        if sample_data is None or len(sample_data) < settings.min_data_points:
            app_logger.info("Using sample data for initial training...")
            sample_data = await crypto_fetcher.generate_sample_data("BTC/USDT", 24)
        
        if sample_data and len(sample_data) >= settings.min_data_points:
            processed_data = data_processor.process_data(sample_data)
            forecast_model.train(processed_data)
            forecast_model.save_model(settings.model_path)
            app_logger.info("Initial model training completed")
        else:
            app_logger.warning("Insufficient data for initial training")
    except Exception as e:
        app_logger.error(f"Initial training failed: {e}")

def get_model() -> CryptoForecastModel:
    """Dependency to get the forecast model"""
    if forecast_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return forecast_model

def get_data_processor() -> DataProcessor:
    """Dependency to get the data processor"""
    if data_processor is None:
        raise HTTPException(status_code=503, detail="Data processor not initialized")
    return data_processor

def get_crypto_fetcher() -> CryptoDataFetcher:
    """Dependency to get the crypto data fetcher"""
    if crypto_fetcher is None:
        raise HTTPException(status_code=503, detail="Crypto data fetcher not initialized")
    return crypto_fetcher

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=forecast_model is not None and forecast_model.is_trained,
        database_connected=True,  # We don't use a database currently
        last_update=None  # Could be added if we track model updates
    )

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def get_forecast(
    request: ForecastRequest,
    model: CryptoForecastModel = Depends(get_model),
    processor: DataProcessor = Depends(get_data_processor),
    fetcher: CryptoDataFetcher = Depends(get_crypto_fetcher)
):
    """Get price forecasts for the specified cryptocurrency"""
    try:
        app_logger.info(f"Forecasting request for {request.symbol}")
        
        # Validate symbol
        if request.symbol not in settings.supported_symbols:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported symbol. Available symbols: {settings.supported_symbols}"
            )
        
        # Fetch recent data
        raw_data = await fetcher.fetch_historical_data(
            request.symbol, 
            request.timeframe, 
            request.lookback_hours
        )
        
        # If no real data, try sample data
        if raw_data is None or len(raw_data) < settings.min_data_points:
            app_logger.warning(f"Insufficient real data for {request.symbol}, using sample data")
            raw_data = await fetcher.generate_sample_data(request.symbol, request.lookback_hours)
        
        if raw_data is None or len(raw_data) < settings.min_data_points:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for forecasting. Need at least {settings.min_data_points} points, got {len(raw_data) if raw_data else 0}"
            )
        
        # Process data
        processed_data = processor.process_data(raw_data)
        
        # Get data summary for quality assessment
        data_summary = processor.get_data_summary(processed_data)
        
        # Get current price
        current_price = float(processed_data['close'].iloc[-1])
        
        # Generate forecasts
        forecasts_raw = model.predict(processed_data)
        
        # Format forecasts for response
        forecasts = {}
        for horizon, forecast_data in forecasts_raw.items():
            forecasts[horizon] = {
                "price": forecast_data["price"],
                "confidence": forecast_data["confidence"]
            }
        
        # Format response
        response = ForecastResponse(
            symbol=request.symbol,
            current_price=current_price,
            forecasts=forecasts,
            timestamp=datetime.now().isoformat(),
            model_type=model.model_type,
            data_quality=data_summary["data_quality"]
        )
        
        app_logger.info(f"Forecast completed for {request.symbol}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Forecast error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.post("/retrain", response_model=RetrainResponse, tags=["Model Management"])
async def retrain_model(
    model: CryptoForecastModel = Depends(get_model),
    processor: DataProcessor = Depends(get_data_processor),
    fetcher: CryptoDataFetcher = Depends(get_crypto_fetcher)
):
    """Retrain the model with fresh data"""
    try:
        app_logger.info("Starting model retraining...")
        
        # Fetch fresh data
        raw_data = await fetcher.fetch_historical_data("BTC/USDT", "1m", 48)
        if raw_data is None or len(raw_data) < 200:
            app_logger.warning("Insufficient real data, using sample data for retraining")
            raw_data = await fetcher.generate_sample_data("BTC/USDT", 48)
        
        if raw_data is None or len(raw_data) < 200:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for retraining. Need at least 200 points, got {len(raw_data) if raw_data else 0}"
            )
        
        # Process and train
        processed_data = processor.process_data(raw_data)
        training_results = model.train(processed_data)
        model.save_model(settings.model_path)
        
        app_logger.info("Model retraining completed")
        return RetrainResponse(
            message="Model retrained successfully",
            timestamp=datetime.now().isoformat(),
            training_results=training_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/symbols", response_model=SymbolListResponse, tags=["Data"])
async def get_available_symbols(
    fetcher: CryptoDataFetcher = Depends(get_crypto_fetcher)
):
    """Get list of available cryptocurrency symbols"""
    try:
        symbols = await fetcher.get_available_symbols()
        return SymbolListResponse(
            symbols=symbols,
            total_count=len(symbols)
        )
    except Exception as e:
        app_logger.error(f"Failed to get symbols: {e}")
        return SymbolListResponse(
            symbols=settings.supported_symbols,
            total_count=len(settings.supported_symbols)
        )

@app.get("/model/info", response_model=ModelInfo, tags=["Model Management"])
async def get_model_info(model: CryptoForecastModel = Depends(get_model)):
    """Get information about the current model"""
    return ModelInfo(**model.get_model_info())

@app.get("/data/summary", response_model=DataSummary, tags=["Data"])
async def get_data_summary(
    symbol: str = "BTC/USDT",
    hours_back: int = 24,
    processor: DataProcessor = Depends(get_data_processor),
    fetcher: CryptoDataFetcher = Depends(get_crypto_fetcher)
):
    """Get summary of the latest data for a symbol"""
    try:
        raw_data = await fetcher.fetch_historical_data(symbol, "1m", hours_back)
        if raw_data is None or len(raw_data) < 10:
            raw_data = await fetcher.generate_sample_data(symbol, hours_back)
        
        if raw_data is None:
            raise HTTPException(status_code=400, detail="No data available")
        
        processed_data = processor.process_data(raw_data)
        summary = processor.get_data_summary(processed_data)
        
        return DataSummary(**summary)
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Data summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.host, 
        port=settings.port, 
        reload=settings.debug
    )