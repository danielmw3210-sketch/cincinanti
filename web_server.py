"""Web server for the AI Crypto Trader dashboard."""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from ai_trader import AITrader
from market_analyzer import MarketAnalyzer
from kraken_client import KrakenClient
from risk_manager import RiskManager
from trading_executor import TradingExecutor
from performance_analyzer import PerformanceAnalyzer
from config import config

# Initialize FastAPI app
app = FastAPI(
    title="AI Crypto Trader Dashboard",
    description="Real-time trading dashboard for AI-powered cryptocurrency trading",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize trading components
kraken_client = KrakenClient()
market_analyzer = MarketAnalyzer(kraken_client)
ai_trader = AITrader(market_analyzer)
risk_manager = RiskManager()
trading_executor = TradingExecutor()
performance_analyzer = PerformanceAnalyzer()

# Global state
trading_active = False
last_update = None

# Pydantic models
class TradeRequest(BaseModel):
    pair: str = "BTC/USD"
    action: str = "buy"
    size: Optional[float] = None

class StartTradingRequest(BaseModel):
    pair: str = "BTC/USD"

# Serve static files
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard."""
    try:
        with open("frontend/dist/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Crypto Trader</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { padding: 20px; background: #f0f0f0; border-radius: 8px; margin: 20px 0; }
                .error { color: #d32f2f; }
                .success { color: #2e7d32; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Crypto Trader Dashboard</h1>
                <div class="status">
                    <h2>Status</h2>
                    <p>Backend server is running successfully!</p>
                    <p>Frontend build not found. Please run: <code>cd frontend && npm run build</code></p>
                </div>
                <div class="status">
                    <h2>API Endpoints</h2>
                    <ul>
                        <li><a href="/api/trading/dashboard">GET /api/trading/dashboard</a> - Get dashboard data</li>
                        <li><a href="/api/trading/status">GET /api/trading/status</a> - Get trading status</li>
                        <li><a href="/docs">GET /docs</a> - API documentation</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """)

@app.get("/api/trading/dashboard")
async def get_trading_dashboard():
    """Get comprehensive trading dashboard data."""
    try:
        global last_update
        last_update = datetime.now()
        
        # Get portfolio summary
        portfolio_summary = trading_executor.get_portfolio_summary()
        
        # Get AI signals for major pairs
        pairs = ['BTC/USD', 'ETH/USD', 'ADA/USD']
        ai_signals = []
        
        for pair in pairs:
            try:
                signal = ai_trader.predict_signal(pair)
                if signal and signal.get('signal') != 'hold':
                    ai_signals.append({
                        'pair': pair,
                        'signal': signal.get('signal', 'hold'),
                        'confidence': signal.get('confidence', 0.0),
                        'timestamp': datetime.now().isoformat(),
                        'reasoning': signal.get('reason', 'AI analysis')
                    })
            except Exception as e:
                print(f"Error getting signal for {pair}: {e}")
        
        # Get performance metrics
        try:
            performance_metrics = performance_analyzer.calculate_comprehensive_metrics()
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            performance_metrics = {}
        
        # Get risk metrics
        try:
            risk_metrics = risk_manager.get_risk_metrics()
        except Exception as e:
            print(f"Error getting risk metrics: {e}")
            risk_metrics = {}
        
        # Mock recent trades for demo
        recent_trades = [
            {
                'id': 1,
                'pair': 'BTC/USD',
                'type': 'buy',
                'size': 0.1,
                'price': 45000,
                'timestamp': datetime.now().isoformat(),
                'pnl': 50,
                'status': 'completed'
            }
        ]
        
        dashboard_data = {
            'portfolio': {
                'totalBalance': portfolio_summary.get('portfolio', {}).get('total_balance', 10000),
                'availableBalance': portfolio_summary.get('portfolio', {}).get('available_balance', 8000),
                'unrealizedPnl': portfolio_summary.get('unrealized_pnl', 0),
                'totalReturn': performance_metrics.get('total_return_pct', 0),
                'dailyPnl': risk_metrics.get('daily_pnl', 0)
            },
            'positions': [
                {
                    'pair': 'BTC/USD',
                    'size': 0.5,
                    'entryPrice': 45000,
                    'currentPrice': 46500,
                    'pnl': 750,
                    'pnlPercent': 3.33,
                    'type': 'long'
                }
            ],
            'recentTrades': recent_trades,
            'performance': {
                'totalTrades': performance_metrics.get('total_trades', 0),
                'winningTrades': performance_metrics.get('winning_trades', 0),
                'losingTrades': performance_metrics.get('losing_trades', 0),
                'winRate': performance_metrics.get('win_rate_pct', 0),
                'profitFactor': performance_metrics.get('profit_factor', 0),
                'sharpeRatio': performance_metrics.get('sharpe_ratio', 0),
                'maxDrawdown': performance_metrics.get('max_drawdown_pct', 0)
            },
            'aiSignals': ai_signals,
            'riskMetrics': {
                'portfolioHeat': risk_metrics.get('portfolio_heat', 0),
                'maxDrawdown': risk_metrics.get('max_drawdown', 0),
                'var95': performance_metrics.get('var_95', 0),
                'currentDrawdown': risk_metrics.get('current_drawdown', 0)
            },
            'tradingStatus': {
                'active': trading_active,
                'lastUpdate': last_update.isoformat() if last_update else None
            }
        }
        
        return {
            'success': True,
            'data': dashboard_data
        }
        
    except Exception as e:
        print(f"Error getting trading dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trading dashboard data: {str(e)}")

@app.get("/api/trading/status")
async def get_trading_status():
    """Get current trading status."""
    try:
        return {
            'success': True,
            'data': {
                'tradingActive': trading_active,
                'lastUpdate': last_update.isoformat() if last_update else None,
                'systemStatus': 'running'
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trading status: {str(e)}")

@app.post("/api/trading/start")
async def start_trading(request: StartTradingRequest):
    """Start automated trading."""
    try:
        global trading_active
        
        if trading_active:
            return {
                'success': False,
                'message': 'Trading is already active'
            }
        
        # Start trading session
        result = trading_executor.start_trading_session(request.pair)
        
        if result.get('success', False):
            trading_active = True
            return {
                'success': True,
                'message': 'Trading started successfully',
                'data': result
            }
        else:
            return {
                'success': False,
                'message': result.get('reason', 'Failed to start trading')
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start trading: {str(e)}")

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop automated trading."""
    try:
        global trading_active
        
        trading_executor.stop_trading()
        trading_active = False
        
        return {
            'success': True,
            'message': 'Trading stopped successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop trading: {str(e)}")

@app.get("/api/trading/signal/{pair}")
async def get_ai_signal(pair: str):
    """Get AI signal for a specific trading pair."""
    try:
        signal = ai_trader.predict_signal(pair)
        
        return {
            'success': True,
            'data': signal
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AI signal: {str(e)}")

@app.get("/api/trading/performance")
async def get_performance_metrics():
    """Get performance metrics."""
    try:
        metrics = performance_analyzer.calculate_comprehensive_metrics()
        
        return {
            'success': True,
            'data': metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.get("/api/trading/risk")
async def get_risk_metrics():
    """Get risk metrics."""
    try:
        metrics = risk_manager.get_risk_metrics()
        
        return {
            'success': True,
            'data': metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get risk metrics: {str(e)}")

@app.post("/api/trading/train")
async def train_models(request: StartTradingRequest):
    """Train AI models."""
    try:
        accuracies = ai_trader.train_models(request.pair)
        
        return {
            'success': True,
            'data': accuracies,
            'message': 'Models trained successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train models: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }

if __name__ == "__main__":
    print("Starting AI Crypto Trader Web Server...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )