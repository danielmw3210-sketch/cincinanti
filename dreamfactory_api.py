"""DreamFactory REST API layer for fills and risk management."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import asyncio
import uvicorn
from contextlib import asynccontextmanager

# Import our modules
from enhanced_trading_executor import EnhancedTradingExecutor, TradingSignal, TradingMode
from database import DatabaseManager, TradingPlatform, OrderStatus, PositionStatus
from mt5_client import MT5Client
from ib_client import IBClient

# Pydantic models for API
class TradingSignalRequest(BaseModel):
    """Trading signal request model."""
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="Action: buy, sell, hold")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level 0-1")
    price: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    volume: Optional[float] = Field(None, description="Volume/quantity")
    platform: Optional[str] = Field(None, description="Preferred platform")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class KillSwitchRequest(BaseModel):
    """Kill switch request model."""
    activate: bool = Field(..., description="Activate or deactivate kill switch")
    reason: Optional[str] = Field(None, description="Reason for kill switch")

class RiskEventRequest(BaseModel):
    """Risk event request model."""
    event_type: str = Field(..., description="Type of risk event")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    message: str = Field(..., description="Event message")
    symbol: Optional[str] = Field(None, description="Related symbol")
    platform: Optional[str] = Field(None, description="Related platform")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class OrderResponse(BaseModel):
    """Order response model."""
    success: bool
    order_id: Optional[int] = None
    ticket: Optional[int] = None
    platform: Optional[str] = None
    symbol: str
    volume: Optional[float] = None
    price: Optional[float] = None
    order_type: Optional[str] = None
    error: Optional[str] = None

class PortfolioSummary(BaseModel):
    """Portfolio summary model."""
    timestamp: str
    platforms: Dict[str, Any]
    total_equity: float
    total_pnl: float
    total_positions: int
    kill_switch_active: bool
    trading_active: bool

class RiskMetrics(BaseModel):
    """Risk metrics model."""
    daily_pnl: float
    max_drawdown: float
    position_count: int
    exposure_by_platform: Dict[str, float]
    risk_events_count: int
    kill_switch_active: bool

# Global variables
trading_executor: Optional[EnhancedTradingExecutor] = None
db_manager: Optional[DatabaseManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global trading_executor, db_manager
    
    # Startup
    logger.info("Starting DreamFactory API...")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Initialize trading executor
    trading_executor = EnhancedTradingExecutor(
        mt5_config={
            'login': 123456,  # Replace with actual MT5 login
            'password': 'password',  # Replace with actual password
            'server': 'MetaQuotes-Demo',  # Replace with actual server
            'timeout': 60000,
            'portable': False
        },
        ib_config={
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1,
            'timeout': 10.0
        },
        ccxt_config={
            'binance': {
                'api_key': 'your_api_key',
                'secret': 'your_secret',
                'sandbox': True
            }
        },
        trading_mode=TradingMode.MT5_ONLY,
        micro_lot_size=0.01
    )
    
    # Initialize trading executor
    await trading_executor.initialize()
    
    logger.info("DreamFactory API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DreamFactory API...")
    
    if trading_executor:
        await trading_executor.cleanup()
    
    logger.info("DreamFactory API shut down")

# Create FastAPI app
app = FastAPI(
    title="DreamFactory Trading API",
    description="REST API for trading operations, fills, and risk management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get trading executor
def get_trading_executor() -> EnhancedTradingExecutor:
    """Get trading executor instance."""
    if not trading_executor:
        raise HTTPException(status_code=503, detail="Trading executor not initialized")
    return trading_executor

# Dependency to get database manager
def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database manager not initialized")
    return db_manager

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "trading_active": trading_executor.is_trading_active if trading_executor else False,
        "kill_switch_active": trading_executor.is_kill_switch_active() if trading_executor else False
    }

# Trading endpoints
@app.post("/api/v1/trade/signal", response_model=OrderResponse)
async def execute_trading_signal(
    signal_request: TradingSignalRequest,
    background_tasks: BackgroundTasks,
    executor: EnhancedTradingExecutor = Depends(get_trading_executor)
):
    """Execute a trading signal."""
    try:
        # Convert request to trading signal
        signal = TradingSignal(
            symbol=signal_request.symbol,
            action=signal_request.action,
            confidence=signal_request.confidence,
            price=signal_request.price,
            stop_loss=signal_request.stop_loss,
            take_profit=signal_request.take_profit,
            volume=signal_request.volume,
            platform=TradingPlatform(signal_request.platform) if signal_request.platform else None,
            metadata=signal_request.metadata or {}
        )
        
        # Execute trade
        result = await executor.execute_trade(signal)
        
        # Log trade execution
        background_tasks.add_task(
            log_trade_execution,
            result,
            signal_request.dict()
        )
        
        return OrderResponse(**result)
        
    except Exception as e:
        logger.error(f"Error executing trading signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trade/portfolio", response_model=PortfolioSummary)
async def get_portfolio_summary(
    executor: EnhancedTradingExecutor = Depends(get_trading_executor)
):
    """Get portfolio summary."""
    try:
        summary = await executor.get_portfolio_summary()
        return PortfolioSummary(**summary)
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/trade/kill-switch")
async def toggle_kill_switch(
    kill_switch_request: KillSwitchRequest,
    background_tasks: BackgroundTasks,
    executor: EnhancedTradingExecutor = Depends(get_trading_executor),
    db: DatabaseManager = Depends(get_db_manager)
):
    """Toggle kill switch."""
    try:
        if kill_switch_request.activate:
            executor.activate_kill_switch()
            message = "Kill switch activated"
        else:
            executor.deactivate_kill_switch()
            message = "Kill switch deactivated"
        
        # Log kill switch event
        background_tasks.add_task(
            log_kill_switch_event,
            kill_switch_request.activate,
            kill_switch_request.reason
        )
        
        return {
            "success": True,
            "message": message,
            "kill_switch_active": executor.is_kill_switch_active(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error toggling kill switch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk management endpoints
@app.post("/api/v1/risk/event")
async def log_risk_event(
    risk_event_request: RiskEventRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Log a risk management event."""
    try:
        event_data = {
            'event_type': risk_event_request.event_type,
            'severity': risk_event_request.severity,
            'message': risk_event_request.message,
            'symbol': risk_event_request.symbol,
            'platform': risk_event_request.platform,
            'metadata': risk_event_request.metadata or {}
        }
        
        event_id = db.log_risk_event(event_data)
        
        return {
            "success": True,
            "event_id": event_id,
            "message": "Risk event logged successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error logging risk event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/metrics", response_model=RiskMetrics)
async def get_risk_metrics(
    db: DatabaseManager = Depends(get_db_manager),
    executor: EnhancedTradingExecutor = Depends(get_trading_executor)
):
    """Get risk metrics."""
    try:
        # Get recent risk events
        risk_events = db.get_risk_events(limit=100)
        
        # Get positions by platform
        positions = db.get_positions(limit=1000)
        exposure_by_platform = {}
        for pos in positions:
            platform = pos['platform']
            if platform not in exposure_by_platform:
                exposure_by_platform[platform] = 0
            exposure_by_platform[platform] += abs(pos['volume'] * pos['price_open'])
        
        # Calculate daily PnL (simplified)
        daily_pnl = sum(pos['profit'] for pos in positions)
        
        # Get portfolio summary for additional metrics
        portfolio_summary = await executor.get_portfolio_summary()
        
        metrics = RiskMetrics(
            daily_pnl=daily_pnl,
            max_drawdown=0.0,  # Calculate from historical data
            position_count=len(positions),
            exposure_by_platform=exposure_by_platform,
            risk_events_count=len(risk_events),
            kill_switch_active=executor.is_kill_switch_active()
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/events")
async def get_risk_events(
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 100,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get risk events."""
    try:
        events = db.get_risk_events(
            severity=severity,
            resolved=resolved,
            limit=limit
        )
        
        return {
            "success": True,
            "events": events,
            "count": len(events),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting risk events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Order and position management endpoints
@app.get("/api/v1/orders")
async def get_orders(
    platform: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get orders."""
    try:
        platform_enum = TradingPlatform(platform) if platform else None
        status_enum = OrderStatus(status) if status else None
        
        orders = db.get_orders(
            platform=platform_enum,
            status=status_enum,
            limit=limit
        )
        
        return {
            "success": True,
            "orders": orders,
            "count": len(orders),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/positions")
async def get_positions(
    platform: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get positions."""
    try:
        platform_enum = TradingPlatform(platform) if platform else None
        status_enum = PositionStatus(status) if status else None
        
        positions = db.get_positions(
            platform=platform_enum,
            status=status_enum,
            limit=limit
        )
        
        return {
            "success": True,
            "positions": positions,
            "count": len(positions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/positions/close-all")
async def close_all_positions(
    background_tasks: BackgroundTasks,
    executor: EnhancedTradingExecutor = Depends(get_trading_executor)
):
    """Close all positions across all platforms."""
    try:
        results = await executor.close_all_positions()
        
        # Log the action
        background_tasks.add_task(
            log_close_all_positions,
            results
        )
        
        return {
            "success": True,
            "message": "Close all positions command executed",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def log_trade_execution(result: Dict[str, Any], signal_data: Dict[str, Any]):
    """Log trade execution as background task."""
    try:
        if db_manager:
            # Log trade execution
            trade_data = {
                'order_id': result.get('order_id'),
                'platform': result.get('platform', 'unknown'),
                'symbol': result.get('symbol'),
                'trade_type': signal_data.get('action'),
                'volume': result.get('volume'),
                'price': result.get('price'),
                'metadata': {
                    'signal_data': signal_data,
                    'execution_result': result
                }
            }
            
            db_manager.log_trade(trade_data)
            
    except Exception as e:
        logger.error(f"Error logging trade execution: {e}")

async def log_kill_switch_event(activated: bool, reason: Optional[str]):
    """Log kill switch event as background task."""
    try:
        if db_manager:
            event_data = {
                'event_type': 'kill_switch_toggle',
                'severity': 'critical' if activated else 'info',
                'message': f"Kill switch {'activated' if activated else 'deactivated'}",
                'metadata': {
                    'reason': reason,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            db_manager.log_risk_event(event_data)
            
    except Exception as e:
        logger.error(f"Error logging kill switch event: {e}")

async def log_close_all_positions(results: Dict[str, int]):
    """Log close all positions action as background task."""
    try:
        if db_manager:
            event_data = {
                'event_type': 'close_all_positions',
                'severity': 'high',
                'message': f"Close all positions executed: {results}",
                'metadata': {
                    'results': results,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            db_manager.log_risk_event(event_data)
            
    except Exception as e:
        logger.error(f"Error logging close all positions: {e}")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "dreamfactory_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )