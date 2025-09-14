from pydantic import BaseSettings
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Settings
    app_name: str = "Crypto Price Forecast API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Settings
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Model Settings
    model_path: str = "models/saved/crypto_forecast_model.pkl"
    default_lookback_hours: int = 24
    min_data_points: int = 100
    retrain_threshold_hours: int = 6
    
    # Forecast Horizons (in minutes)
    forecast_horizons: List[int] = [15, 30, 60]
    
    # Feature Engineering
    lag_periods: List[int] = [1, 5, 15, 30, 60]
    rolling_windows: List[int] = [5, 15, 30, 60]
    
    # Data Sources
    default_exchange: str = "binance"
    supported_symbols: List[str] = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
        "XRP/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT"
    ]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()