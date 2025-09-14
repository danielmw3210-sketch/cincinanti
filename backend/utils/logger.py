import logging
import sys
from typing import Optional
from config.settings import settings

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Set up a logger with consistent formatting"""
    
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level or settings.log_level))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(settings.log_format)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

# Create main application logger
app_logger = setup_logger("crypto_forecast")