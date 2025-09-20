"""Setup script for enhanced trading system."""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
from loguru import logger

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    try:
        logger.info(f"Running: {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def create_env_file():
    """Create .env file with default values."""
    env_content = """# Enhanced Trading System Configuration

# Environment
ENVIRONMENT=development
DEBUG=false
TIMEZONE=UTC

# Database
DATABASE_URL=postgresql://trader:password@localhost:5432/trading_db

# MT5 Configuration
MT5_ENABLED=true
MT5_LOGIN=123456
MT5_PASSWORD=your_mt5_password
MT5_SERVER=MetaQuotes-Demo
MT5_TIMEOUT=60000
MT5_PORTABLE=false

# Interactive Brokers Configuration
IB_ENABLED=true
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
IB_TIMEOUT=10.0

# CCXT Configuration
CCXT_ENABLED=true
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
BINANCE_SANDBOX=true
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret
KRAKEN_SANDBOX=true

# Trading Configuration
TRADING_MODE=mt5_only
MICRO_LOT_SIZE=0.01
DEFAULT_SYMBOLS=["EURUSD", "GBPUSD", "USDJPY"]
TRADING_ENABLED=true

# Risk Management
MAX_DAILY_LOSS=0.05
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02
MAX_DRAWDOWN=0.15
KILL_SWITCH_ENABLED=true
POSITION_MONITORING_INTERVAL=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
CORS_ORIGINS=["*"]
API_KEY=
RATE_LIMIT=100

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_system.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        logger.info("Created .env file with default configuration")
    else:
        logger.info(".env file already exists")

def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "data",
        "backups",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    logger.info("Installing Python dependencies...")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    # Install additional dependencies for MT5
    if not run_command("pip install MetaTrader5", "Installing MetaTrader5"):
        logger.warning("Failed to install MetaTrader5 - MT5 functionality will not be available")
    
    return True

def setup_database():
    """Setup PostgreSQL database."""
    logger.info("Setting up database...")
    
    # Check if PostgreSQL is running
    if not run_command("pg_isready", "Checking PostgreSQL status"):
        logger.warning("PostgreSQL is not running. Please start PostgreSQL and create the database manually.")
        logger.info("To create the database manually:")
        logger.info("1. Start PostgreSQL")
        logger.info("2. Create database: createdb trading_db")
        logger.info("3. Create user: createuser -s trader")
        logger.info("4. Set password: psql -c \"ALTER USER trader PASSWORD 'password';\"")
        return False
    
    # Create database if it doesn't exist
    run_command("createdb trading_db 2>/dev/null || true", "Creating database (if not exists)")
    
    logger.info("Database setup completed")
    return True

def create_docker_compose():
    """Create Docker Compose file for PostgreSQL."""
    docker_compose_content = """version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: trading_postgres
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

volumes:
  postgres_data:
"""
    
    with open("docker-compose.enhanced.yml", "w") as f:
        f.write(docker_compose_content)
    
    logger.info("Created docker-compose.enhanced.yml for PostgreSQL")

def create_init_sql():
    """Create SQL initialization file."""
    init_sql_content = """-- Trading System Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better performance
-- (Tables will be created by SQLAlchemy)

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader;
"""
    
    with open("init.sql", "w") as f:
        f.write(init_sql_content)
    
    logger.info("Created init.sql for database initialization")

def create_startup_scripts():
    """Create startup scripts."""
    
    # Linux/Mac startup script
    startup_script = """#!/bin/bash
# Enhanced Trading System Startup Script

echo "Starting Enhanced Trading System..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please run setup_enhanced.py first."
    exit 1
fi

# Start PostgreSQL if using Docker
if [ "$1" = "--with-db" ]; then
    echo "Starting PostgreSQL with Docker..."
    docker-compose -f docker-compose.enhanced.yml up -d postgres
    sleep 5
fi

# Start the trading system
echo "Starting trading system..."
python enhanced_main.py start
"""
    
    with open("start_enhanced.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod("start_enhanced.sh", 0o755)
    
    # Windows startup script
    windows_script = """@echo off
REM Enhanced Trading System Startup Script

echo Starting Enhanced Trading System...

REM Check if .env exists
if not exist .env (
    echo Error: .env file not found. Please run setup_enhanced.py first.
    exit /b 1
)

REM Start PostgreSQL if using Docker
if "%1"=="--with-db" (
    echo Starting PostgreSQL with Docker...
    docker-compose -f docker-compose.enhanced.yml up -d postgres
    timeout /t 5
)

REM Start the trading system
echo Starting trading system...
python enhanced_main.py start
"""
    
    with open("start_enhanced.bat", "w") as f:
        f.write(windows_script)
    
    logger.info("Created startup scripts")

def create_readme():
    """Create README for enhanced system."""
    readme_content = """# Enhanced Multi-Platform Trading System

A sophisticated trading system that supports MetaTrader5, Interactive Brokers, and CCXT exchanges with PostgreSQL logging and DreamFactory REST API.

## Features

- **Multi-Platform Support**: MT5, Interactive Brokers, CCXT
- **Micro-Lot Trading**: Safe testing with small position sizes
- **Hard Kill Switch**: Emergency stop functionality
- **PostgreSQL Logging**: Comprehensive order and trade logging
- **Risk Management**: Built-in risk controls and monitoring
- **REST API**: DreamFactory API for external integration
- **Real-time Monitoring**: Position and risk monitoring

## Quick Start

1. **Setup the system:**
   ```bash
   python setup_enhanced.py
   ```

2. **Configure your settings:**
   Edit `.env` file with your API keys and credentials

3. **Start with database:**
   ```bash
   ./start_enhanced.sh --with-db
   ```

4. **Or start without database:**
   ```bash
   ./start_enhanced.sh
   ```

## Configuration

Edit the `.env` file to configure:

- **MT5 Settings**: Login, password, server
- **IB Settings**: Host, port, client ID
- **CCXT Settings**: Exchange API keys
- **Risk Management**: Loss limits, position sizes
- **Trading Mode**: Single or multi-platform

## API Endpoints

The system provides a REST API on port 8000:

- `GET /health` - Health check
- `POST /api/v1/trade/signal` - Execute trading signal
- `GET /api/v1/trade/portfolio` - Get portfolio summary
- `POST /api/v1/trade/kill-switch` - Toggle kill switch
- `GET /api/v1/risk/metrics` - Get risk metrics
- `GET /api/v1/orders` - Get orders
- `GET /api/v1/positions` - Get positions

## Commands

```bash
# Check system status
python enhanced_main.py status

# Execute manual trade
python enhanced_main.py trade EURUSD buy 0.8

# Toggle kill switch
python enhanced_main.py kill-switch

# Start automated trading
python enhanced_main.py start
```

## Safety Features

- **Kill Switch**: Immediately stops all trading
- **Risk Limits**: Daily loss and position size limits
- **Micro Lots**: Safe testing with small sizes
- **Comprehensive Logging**: All actions logged to database
- **Real-time Monitoring**: Continuous position and risk monitoring

## Requirements

- Python 3.8+
- PostgreSQL 12+
- MetaTrader5 (for MT5 support)
- Interactive Brokers TWS/Gateway (for IB support)
- Docker (optional, for database)

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Setup database:
   ```bash
   createdb trading_db
   createuser -s trader
   psql -c "ALTER USER trader PASSWORD 'password';"
   ```

3. Configure MT5 (if using):
   - Install MetaTrader5
   - Login to your account
   - Enable automated trading

4. Configure Interactive Brokers (if using):
   - Install TWS or Gateway
   - Enable API connections
   - Configure port and client ID

## Support

For issues and questions, please check the logs in the `logs/` directory.
"""
    
    with open("README_ENHANCED.md", "w") as f:
        f.write(readme_content)
    
    logger.info("Created README_ENHANCED.md")

def main():
    """Main setup function."""
    logger.info("Setting up Enhanced Trading System...")
    
    # Create directories
    create_directories()
    
    # Create configuration files
    create_env_file()
    create_docker_compose()
    create_init_sql()
    create_startup_scripts()
    create_readme()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Setup database
    setup_database()
    
    logger.info("Setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Edit .env file with your API keys and credentials")
    logger.info("2. Start the system: ./start_enhanced.sh --with-db")
    logger.info("3. Or start without database: ./start_enhanced.sh")
    
    return True

if __name__ == "__main__":
    main()