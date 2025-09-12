"""Configuration management for the AI crypto trader."""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class TradingConfig(BaseSettings):
    """Trading configuration settings."""
    
    # Kraken API
    kraken_api_key: str = Field(..., env="KRAKEN_API_KEY")
    kraken_secret_key: str = Field(..., env="KRAKEN_SECRET_KEY")
    kraken_sandbox: bool = Field(True, env="KRAKEN_SANDBOX")
    
    # Trading parameters
    default_pair: str = Field("BTC/USD", env="DEFAULT_PAIR")
    initial_balance: float = Field(1000.0, env="INITIAL_BALANCE")
    max_position_size: float = Field(0.1, env="MAX_POSITION_SIZE")
    risk_per_trade: float = Field(0.02, env="RISK_PER_TRADE")
    
    # AI Model
    model_type: str = Field("ensemble", env="MODEL_TYPE")
    retrain_interval: int = Field(24, env="RETRAIN_INTERVAL")
    feature_window: int = Field(100, env="FEATURE_WINDOW")
    
    # Monitoring
    log_level: str = Field("INFO", env="LOG_LEVEL")
    webhook_url: Optional[str] = Field(None, env="WEBHOOK_URL")
    alert_email: Optional[str] = Field(None, env="ALERT_EMAIL")
    
    # Safety
    emergency_stop: bool = Field(False, env="EMERGENCY_STOP")
    max_daily_loss: float = Field(0.05, env="MAX_DAILY_LOSS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global config instance
config = TradingConfig()