from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class ForecastRequest(BaseModel):
    """Request model for price forecasting"""
    symbol: str = Field(default="BTC/USDT", description="Cryptocurrency symbol to forecast")
    timeframe: str = Field(default="1m", description="Data timeframe (1m, 5m, 15m, etc.)")
    lookback_hours: int = Field(default=24, ge=1, le=168, description="Hours of historical data to use")
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "lookback_hours": 24
            }
        }

class ForecastData(BaseModel):
    """Individual forecast data point"""
    price: float = Field(description="Predicted price")
    confidence: float = Field(ge=0.0, le=1.0, description="Prediction confidence (0-1)")

class ForecastResponse(BaseModel):
    """Response model for price forecasting"""
    symbol: str = Field(description="Cryptocurrency symbol")
    current_price: float = Field(description="Current market price")
    forecasts: Dict[str, ForecastData] = Field(description="Forecasts for different horizons")
    timestamp: str = Field(description="Prediction timestamp")
    model_type: str = Field(description="Type of model used")
    data_quality: str = Field(description="Quality of input data")
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "current_price": 45000.0,
                "forecasts": {
                    "15min": {"price": 45100.0, "confidence": 0.85},
                    "30min": {"price": 45200.0, "confidence": 0.78},
                    "60min": {"price": 45300.0, "confidence": 0.72}
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "model_type": "xgboost",
                "data_quality": "good"
            }
        }

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str = Field(description="Type of model (xgboost/hist_gradient_boosting)")
    is_trained: bool = Field(description="Whether the model is trained")
    horizons: List[int] = Field(description="Available forecast horizons in minutes")
    feature_count: int = Field(description="Number of features used")
    xgboost_available: bool = Field(description="Whether XGBoost is available")
    last_trained: Optional[str] = Field(description="Last training timestamp")

class DataSummary(BaseModel):
    """Data summary response"""
    total_rows: int = Field(description="Total number of data points")
    date_range: Dict[str, Optional[str]] = Field(description="Start and end timestamps")
    price_stats: Dict[str, float] = Field(description="Price statistics")
    volume_stats: Dict[str, float] = Field(description="Volume statistics")
    missing_values: Dict[str, int] = Field(description="Missing value counts")
    data_quality: str = Field(description="Overall data quality assessment")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether model is loaded")
    database_connected: bool = Field(description="Whether database is connected")
    last_update: Optional[str] = Field(description="Last model update timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(description="Detailed error information")
    timestamp: str = Field(description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Insufficient data for forecasting",
                "detail": "Need at least 100 data points, got 50",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class RetrainResponse(BaseModel):
    """Model retraining response"""
    message: str = Field(description="Retraining status message")
    timestamp: str = Field(description="Retraining timestamp")
    training_results: Optional[Dict[str, float]] = Field(description="Training metrics")
    
class SymbolListResponse(BaseModel):
    """Available symbols response"""
    symbols: List[str] = Field(description="List of available cryptocurrency symbols")
    total_count: int = Field(description="Total number of symbols")
    
class BatchForecastRequest(BaseModel):
    """Request model for batch forecasting multiple symbols"""
    symbols: List[str] = Field(description="List of cryptocurrency symbols")
    timeframe: str = Field(default="1m", description="Data timeframe")
    lookback_hours: int = Field(default=24, ge=1, le=168, description="Hours of historical data")
    
class BatchForecastResponse(BaseModel):
    """Response model for batch forecasting"""
    forecasts: Dict[str, ForecastResponse] = Field(description="Forecasts for each symbol")
    timestamp: str = Field(description="Batch prediction timestamp")
    total_symbols: int = Field(description="Total number of symbols processed")
    successful_symbols: int = Field(description="Number of successfully processed symbols")
    failed_symbols: List[str] = Field(description="List of symbols that failed to process")