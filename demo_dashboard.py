#!/usr/bin/env python3
"""Demo mode for the AI Crypto Trader Dashboard."""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI Crypto Trader Dashboard - Demo Mode",
    description="Demo dashboard with simulated trading data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

# Demo state
trading_active = False
demo_data = {
    'portfolio': {
        'totalBalance': 10000.0,
        'availableBalance': 8000.0,
        'unrealizedPnl': 0.0,
        'totalReturn': 0.0,
        'dailyPnl': 0.0
    },
    'positions': [],
    'recentTrades': [],
    'performance': {
        'totalTrades': 0,
        'winningTrades': 0,
        'losingTrades': 0,
        'winRate': 0.0,
        'profitFactor': 0.0,
        'sharpeRatio': 0.0,
        'maxDrawdown': 0.0
    },
    'aiSignals': [],
    'riskMetrics': {
        'portfolioHeat': 0.0,
        'maxDrawdown': 0.0,
        'var95': 0.0,
        'currentDrawdown': 0.0
    }
}

# Pydantic models
class StartTradingRequest(BaseModel):
    pair: str = "BTC/USD"

def generate_demo_data():
    """Generate realistic demo trading data."""
    global demo_data
    
    # Simulate some market movement
    btc_price = 45000 + random.uniform(-2000, 2000)
    eth_price = 3200 + random.uniform(-200, 200)
    
    # Generate AI signals
    signals = []
    pairs = ['BTC/USD', 'ETH/USD', 'ADA/USD']
    
    for pair in pairs:
        if random.random() > 0.7:  # 30% chance of signal
            signal_type = random.choice(['buy', 'sell', 'hold'])
            confidence = random.uniform(0.6, 0.95)
            
            reasons = {
                'buy': [
                    'Strong bullish momentum with high volume',
                    'Breakout above resistance level',
                    'Positive divergence in RSI',
                    'Golden cross formation detected'
                ],
                'sell': [
                    'Bearish divergence in MACD',
                    'Price rejected at resistance',
                    'Volume declining on uptrend',
                    'Overbought conditions detected'
                ],
                'hold': [
                    'Mixed signals, waiting for clearer direction',
                    'Consolidation phase, no clear trend',
                    'Low volume, insufficient conviction',
                    'Waiting for breakout confirmation'
                ]
            }
            
            signals.append({
                'pair': pair,
                'signal': signal_type,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'reasoning': random.choice(reasons[signal_type])
            })
    
    # Update portfolio with some random movement
    daily_change = random.uniform(-500, 800)
    demo_data['portfolio']['dailyPnl'] = daily_change
    demo_data['portfolio']['totalBalance'] = max(5000, demo_data['portfolio']['totalBalance'] + daily_change)
    demo_data['portfolio']['totalReturn'] = ((demo_data['portfolio']['totalBalance'] - 10000) / 10000) * 100
    
    # Update performance metrics
    if random.random() > 0.8:  # 20% chance of new trade
        demo_data['performance']['totalTrades'] += 1
        if random.random() > 0.4:  # 60% win rate
            demo_data['performance']['winningTrades'] += 1
        else:
            demo_data['performance']['losingTrades'] += 1
        
        demo_data['performance']['winRate'] = (
            demo_data['performance']['winningTrades'] / 
            demo_data['performance']['totalTrades'] * 100
        )
    
    # Update risk metrics
    demo_data['riskMetrics']['portfolioHeat'] = random.uniform(0.05, 0.25)
    demo_data['riskMetrics']['currentDrawdown'] = random.uniform(0.0, 0.05)
    
    demo_data['aiSignals'] = signals

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
            <title>AI Crypto Trader - Demo</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { padding: 20px; background: #f0f0f0; border-radius: 8px; margin: 20px 0; }
                .demo { background: #e3f2fd; border-left: 4px solid #2196f3; }
                .error { color: #d32f2f; }
                .success { color: #2e7d32; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Crypto Trader Dashboard - Demo Mode</h1>
                <div class="status demo">
                    <h2>🎮 Demo Mode Active</h2>
                    <p>This is a demonstration of the AI Crypto Trader dashboard with simulated data.</p>
                    <p>No real trading is occurring - this is for testing and demonstration purposes only.</p>
                </div>
                <div class="status">
                    <h2>API Endpoints</h2>
                    <ul>
                        <li><a href="/api/trading/dashboard">GET /api/trading/dashboard</a> - Get demo dashboard data</li>
                        <li><a href="/api/trading/status">GET /api/trading/status</a> - Get trading status</li>
                        <li><a href="/docs">GET /docs</a> - API documentation</li>
                    </ul>
                </div>
                <div class="status">
                    <h2>Demo Features</h2>
                    <ul>
                        <li>✅ Simulated portfolio data</li>
                        <li>✅ Mock AI trading signals</li>
                        <li>✅ Realistic performance metrics</li>
                        <li>✅ Risk management simulation</li>
                        <li>✅ Real-time data updates</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """)

@app.get("/api/trading/dashboard")
async def get_trading_dashboard():
    """Get demo trading dashboard data."""
    try:
        # Generate new demo data
        generate_demo_data()
        
        # Add trading status
        demo_data['tradingStatus'] = {
            'active': trading_active,
            'lastUpdate': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'data': demo_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get demo dashboard data: {str(e)}")

@app.get("/api/trading/status")
async def get_trading_status():
    """Get current trading status."""
    return {
        'success': True,
        'data': {
            'tradingActive': trading_active,
            'lastUpdate': datetime.now().isoformat(),
            'systemStatus': 'demo_mode'
        }
    }

@app.post("/api/trading/start")
async def start_trading(request: StartTradingRequest):
    """Start demo trading."""
    global trading_active
    trading_active = True
    
    return {
        'success': True,
        'message': 'Demo trading started successfully',
        'data': {
            'pair': request.pair,
            'mode': 'demo'
        }
    }

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop demo trading."""
    global trading_active
    trading_active = False
    
    return {
        'success': True,
        'message': 'Demo trading stopped successfully'
    }

@app.get("/api/trading/signal/{pair}")
async def get_ai_signal(pair: str):
    """Get demo AI signal for a trading pair."""
    signals = [s for s in demo_data['aiSignals'] if s['pair'] == pair]
    
    if signals:
        return {
            'success': True,
            'data': signals[0]
        }
    else:
        return {
            'success': True,
            'data': {
                'pair': pair,
                'signal': 'hold',
                'confidence': 0.5,
                'reason': 'No clear signal in demo mode'
            }
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'mode': 'demo',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }

if __name__ == "__main__":
    print("🎮 Starting AI Crypto Trader Dashboard - Demo Mode...")
    print("=" * 60)
    print("🌐 Dashboard will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🎯 This is DEMO MODE - No real trading occurs!")
    print("=" * 60)
    
    uvicorn.run(
        "demo_dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )