"""Enhanced configuration for multi-platform trading system."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class MT5Config(BaseSettings):
    """MT5 configuration."""
    login: Optional[int] = Field(None, env="MT5_LOGIN")
    password: Optional[str] = Field(None, env="MT5_PASSWORD")
    server: Optional[str] = Field(None, env="MT5_SERVER")
    timeout: int = Field(60000, env="MT5_TIMEOUT")
    portable: bool = Field(False, env="MT5_PORTABLE")
    enabled: bool = Field(True, env="MT5_ENABLED")

class IBConfig(BaseSettings):
    """Interactive Brokers configuration."""
    host: str = Field("127.0.0.1", env="IB_HOST")
    port: int = Field(7497, env="IB_PORT")
    client_id: int = Field(1, env="IB_CLIENT_ID")
    timeout: float = Field(10.0, env="IB_TIMEOUT")
    enabled: bool = Field(True, env="IB_ENABLED")

class CCXTConfig(BaseSettings):
    """CCXT configuration."""
    binance_api_key: Optional[str] = Field(None, env="BINANCE_API_KEY")
    binance_secret: Optional[str] = Field(None, env="BINANCE_SECRET")
    binance_sandbox: bool = Field(True, env="BINANCE_SANDBOX")
    
    kraken_api_key: Optional[str] = Field(None, env="KRAKEN_API_KEY")
    kraken_secret: Optional[str] = Field(None, env="KRAKEN_SECRET")
    kraken_sandbox: bool = Field(True, env="KRAKEN_SANDBOX")
    
    enabled: bool = Field(True, env="CCXT_ENABLED")

class DatabaseConfig(BaseSettings):
    """Database configuration."""
    url: str = Field("postgresql://trader:password@localhost:5432/trading_db", env="DATABASE_URL")
    echo: bool = Field(False, env="DATABASE_ECHO")
    pool_size: int = Field(10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(20, env="DATABASE_MAX_OVERFLOW")

class RiskConfig(BaseSettings):
    """Risk management configuration."""
    max_daily_loss: float = Field(0.05, env="MAX_DAILY_LOSS")  # 5%
    max_position_size: float = Field(0.1, env="MAX_POSITION_SIZE")  # 10%
    risk_per_trade: float = Field(0.02, env="RISK_PER_TRADE")  # 2%
    max_drawdown: float = Field(0.15, env="MAX_DRAWDOWN")  # 15%
    kill_switch_enabled: bool = Field(True, env="KILL_SWITCH_ENABLED")
    position_monitoring_interval: int = Field(60, env="POSITION_MONITORING_INTERVAL")  # seconds

class TradingConfig(BaseSettings):
    """Trading configuration."""
    mode: str = Field("mt5_only", env="TRADING_MODE")  # mt5_only, ib_only, ccxt_only, multi_platform
    micro_lot_size: float = Field(0.01, env="MICRO_LOT_SIZE")
    default_symbols: list = Field(["EURUSD", "GBPUSD", "USDJPY"], env="DEFAULT_SYMBOLS")
    trading_hours: Dict[str, str] = Field({
        "start": "00:00",
        "end": "23:59"
    }, env="TRADING_HOURS")
    enabled: bool = Field(True, env="TRADING_ENABLED")

class APIConfig(BaseSettings):
    """API configuration."""
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="API_DEBUG")
    cors_origins: list = Field(["*"], env="CORS_ORIGINS")
    api_key: Optional[str] = Field(None, env="API_KEY")
    rate_limit: int = Field(100, env="RATE_LIMIT")  # requests per minute

class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field("INFO", env="LOG_LEVEL")
    file: Optional[str] = Field(None, env="LOG_FILE")
    max_size: str = Field("10MB", env="LOG_MAX_SIZE")
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")
    format: str = Field("{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}", env="LOG_FORMAT")

class EnhancedConfig(BaseSettings):
    """Enhanced trading system configuration."""
    
    # Sub-configurations
    mt5: MT5Config = MT5Config()
    ib: IBConfig = IBConfig()
    ccxt: CCXTConfig = CCXTConfig()
    database: DatabaseConfig = DatabaseConfig()
    risk: RiskConfig = RiskConfig()
    trading: TradingConfig = TradingConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Global settings
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    timezone: str = Field("UTC", env="TIMEZONE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_mt5_config(self) -> Dict[str, Any]:
        """Get MT5 configuration as dictionary."""
        return {
            'login': self.mt5.login,
            'password': self.mt5.password,
            'server': self.mt5.server,
            'timeout': self.mt5.timeout,
            'portable': self.mt5.portable
        }
    
    def get_ib_config(self) -> Dict[str, Any]:
        """Get IB configuration as dictionary."""
        return {
            'host': self.ib.host,
            'port': self.ib.port,
            'client_id': self.ib.client_id,
            'timeout': self.ib.timeout
        }
    
    def get_ccxt_config(self) -> Dict[str, Any]:
        """Get CCXT configuration as dictionary."""
        config = {}
        
        if self.ccxt.binance_api_key and self.ccxt.binance_secret:
            config['binance'] = {
                'api_key': self.ccxt.binance_api_key,
                'secret': self.ccxt.binance_secret,
                'sandbox': self.ccxt.binance_sandbox
            }
        
        if self.ccxt.kraken_api_key and self.ccxt.kraken_secret:
            config['kraken'] = {
                'api_key': self.ccxt.kraken_api_key,
                'secret': self.ccxt.kraken_secret,
                'sandbox': self.ccxt.kraken_sandbox
            }
        
        return config
    
    def get_trading_mode(self):
        """Get trading mode enum."""
        from enhanced_trading_executor import TradingMode
        
        mode_mapping = {
            'mt5_only': TradingMode.MT5_ONLY,
            'ib_only': TradingMode.IB_ONLY,
            'ccxt_only': TradingMode.CCXT_ONLY,
            'multi_platform': TradingMode.MULTI_PLATFORM
        }
        
        return mode_mapping.get(self.trading.mode, TradingMode.MT5_ONLY)
    
    def is_platform_enabled(self, platform: str) -> bool:
        """Check if platform is enabled."""
        platform_configs = {
            'mt5': self.mt5.enabled,
            'ib': self.ib.enabled,
            'ccxt': self.ccxt.enabled
        }
        
        return platform_configs.get(platform, False)
    
    def get_enabled_platforms(self) -> list:
        """Get list of enabled platforms."""
        platforms = []
        
        if self.mt5.enabled:
            platforms.append('mt5')
        if self.ib.enabled:
            platforms.append('ib')
        if self.ccxt.enabled:
            platforms.append('ccxt')
        
        return platforms
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        errors = []
        
        # Check if at least one platform is enabled
        if not any([self.mt5.enabled, self.ib.enabled, self.ccxt.enabled]):
            errors.append("At least one trading platform must be enabled")
        
        # Check MT5 configuration if enabled
        if self.mt5.enabled:
            if not self.mt5.login or not self.mt5.password or not self.mt5.server:
                errors.append("MT5 configuration incomplete: login, password, and server are required")
        
        # Check IB configuration if enabled
        if self.ib.enabled:
            if not self.ib.host or not self.ib.port:
                errors.append("IB configuration incomplete: host and port are required")
        
        # Check CCXT configuration if enabled
        if self.ccxt.enabled:
            if not any([
                (self.ccxt.binance_api_key and self.ccxt.binance_secret),
                (self.ccxt.kraken_api_key and self.ccxt.kraken_secret)
            ]):
                errors.append("CCXT configuration incomplete: at least one exchange API key/secret pair is required")
        
        # Check risk parameters
        if self.risk.max_daily_loss <= 0 or self.risk.max_daily_loss > 1:
            errors.append("max_daily_loss must be between 0 and 1")
        
        if self.risk.max_position_size <= 0 or self.risk.max_position_size > 1:
            errors.append("max_position_size must be between 0 and 1")
        
        if self.risk.risk_per_trade <= 0 or self.risk.risk_per_trade > 1:
            errors.append("risk_per_trade must be between 0 and 1")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True

# Global configuration instance
config = EnhancedConfig()

# Validate configuration on import
if not config.validate_config():
    print("Warning: Configuration validation failed. Please check your settings.")