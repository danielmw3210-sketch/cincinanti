"""
Enhanced web server for the AI Crypto Trader with continuous learning capabilities.
Implements data storage, performance tracking, automatic retraining, 
incremental learning, and feedback loops.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel
import uvicorn

# Import the enhanced AI trader
from enhanced_ai_trader import EnhancedAITrader

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced AI Crypto Trader",
    description="AI-powered cryptocurrency trading signals with continuous learning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI trader instance
trader = None

def get_ai_trader():
    """Get or create AI trader instance with lazy initialization."""
    global trader
    if trader is None:
        print("🚀 Initializing Enhanced AI Trader...")
        trader = EnhancedAITrader()
        print("✅ Enhanced AI Trader initialized")
    return trader

# Pydantic models
class ForecastRequest(BaseModel):
    symbol: str

class RetrainRequest(BaseModel):
    pair: Optional[str] = None
    force: Optional[bool] = False

# Static file serving
frontend_path = Path(__file__).parent / "frontend" / "build"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

# Root endpoint - serve React app
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React frontend."""
    if frontend_path.exists():
        index_file = frontend_path / "index.html"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
    
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced AI Crypto Trader</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .container { max-width: 600px; margin: 0 auto; }
            .status { color: #059669; font-size: 1.2em; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Enhanced AI Crypto Trader</h1>
            <p class="status">✅ Backend is running with continuous learning capabilities!</p>
            <p>Features: Data Storage, Performance Tracking, Automatic Retraining, Incremental Learning, Feedback Loops</p>
            <p>API Documentation: <a href="/docs">/docs</a></p>
        </div>
    </body>
    </html>
    """)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check with learning status."""
    trader = get_ai_trader()
    # Calculate overall model status
    trained_pairs = sum(1 for pair in trader.available_pairs if trader.is_trained.get(pair, False))
    overall_trained = trained_pairs > 0
    
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'model_loaded': overall_trained,
        'model_version': trader.model_version,
        'trained_pairs': trained_pairs,
        'total_pairs': len(trader.available_pairs),
        'last_retrain': {pair: trader.last_retrain[pair].isoformat() if trader.last_retrain[pair] else None for pair in trader.available_pairs},
        'learning_active': True
    }

# Symbols endpoint
@app.get("/symbols")
async def get_symbols():
    """Get available trading symbols."""
    trader = get_ai_trader()
    return {
        'symbols': trader.available_pairs
    }

# Enhanced forecast endpoint
@app.post("/forecast")
async def get_forecast(request: ForecastRequest):
    """Get AI forecast with enhanced learning capabilities."""
    try:
        trader = get_ai_trader()
        
        print(f"🔮 Enhanced forecast request for symbol: {request.symbol}")
        
        # Get prediction with learning capabilities
        prediction_result = trader.predict_signal(request.symbol)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result["error"])
        
        # Get performance metrics
        performance = trader.get_performance_metrics(request.symbol)
        
        response = {
            'success': True,
            'data': {
                'symbol': request.symbol,
                'forecast': prediction_result,
                'model_info': {
                    'type': 'Enhanced XGBoost',
                    'version': trader.model_version,
                    'features': ['sma_5', 'sma_20', 'rsi', 'price_change', 'volatility'],
                    'trained': trader.is_trained,
                    'learning_active': True
                },
                'performance': performance.get('performance_by_pair', {}).get(request.symbol, {})
            }
        }
        
        print(f"✅ Enhanced forecast generated for {request.symbol}")
        return response
        
    except Exception as e:
        print(f"❌ Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced retrain endpoint
@app.post("/retrain")
async def retrain_model(request: RetrainRequest = None):
    """Retrain the AI model with enhanced capabilities."""
    try:
        trader = get_ai_trader()
        
        if request and request.force:
            print("🔄 Force retraining model...")
            result = trader.train_model(request.pair or "BTC/GBP")
        else:
            print("🔄 Retraining with learning data...")
            trader.retrain_with_learning_data()
            result = {"success": True, "message": "Model retrained with learning data"}
        
        if result.get("success"):
            return {
                'success': True,
                'message': 'Model retrained successfully',
                'accuracy': result.get('accuracy', 0.0),
                'samples': result.get('samples', 0),
                'model_version': trader.model_version,
                'last_retrain': trader.last_retrain.isoformat() if trader.last_retrain else None
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Retraining failed"))
            
    except Exception as e:
        print(f"❌ Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrain model: {str(e)}")

# Performance metrics endpoint
@app.get("/performance")
async def get_performance_metrics():
    """Get comprehensive performance metrics."""
    try:
        trader = get_ai_trader()
        metrics = trader.get_performance_metrics()
        return {
            'success': True,
            'data': metrics
        }
    except Exception as e:
        print(f"❌ Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get detailed model information."""
    trader = get_ai_trader()
    return {
        'model_type': 'Enhanced XGBoost',
        'version': trader.model_version,
        'features': ['sma_5', 'sma_20', 'rsi', 'price_change', 'volatility'],
        'trained': trader.is_trained,
        'last_retrained': trader.last_retrain.isoformat() if trader.last_retrain else None,
        'learning_active': True,
        'retrain_interval_hours': trader.retrain_interval_hours,
        'performance_threshold': trader.performance_threshold
    }

# Data summary endpoint
@app.get("/data/summary")
async def get_data_summary():
    """Get data quality and learning summary."""
    try:
        trader = get_ai_trader()
        
        # Get performance metrics
        performance = trader.get_performance_metrics()
        
        # Get learning data counts
        learning_data = {}
        pairs = ['BTC/GBP', 'ETH/GBP', 'SOL/GBP', 'XRP/GBP', 'DOGE/GBP']
        
        for pair in pairs:
            df = trader.storage.get_learning_data(pair, days=30)
            learning_data[pair] = {
                'total_records': len(df),
                'recent_records': len(df[df['timestamp'] >= datetime.now() - timedelta(days=7)])
            }
        
        return {
            'success': True,
            'data': {
                'learning_data': learning_data,
                'performance': performance,
                'model_status': {
                    'trained': trader.is_trained,
                    'version': trader.model_version,
                    'last_retrain': trader.last_retrain.isoformat() if trader.last_retrain else None
                }
            }
        }
    except Exception as e:
        print(f"❌ Data summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup endpoint
@app.post("/cleanup")
async def cleanup_old_data():
    """Clean up old data to maintain performance."""
    try:
        trader = get_ai_trader()
        trader.cleanup_old_data()
        return {
            'success': True,
            'message': 'Old data cleaned up successfully'
        }
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Favicon and manifest endpoints
@app.get("/favicon.ico")
async def favicon():
    return Response(content="", media_type="image/x-icon")

@app.get("/manifest.json")
async def manifest():
    return JSONResponse(content={
        "name": "Enhanced AI Crypto Trader",
        "short_name": "AI Trader",
        "version": "2.0.0",
        "description": "AI-powered cryptocurrency trading with continuous learning"
    })

# Top 10 cryptocurrencies endpoint
@app.get("/api/crypto/top10/gbp")
async def get_top10_cryptos():
    """Get top 10 cryptocurrencies in GBP."""
    return {
        'cryptos': [
            {'symbol': 'BTC/GBP', 'name': 'Bitcoin', 'rank': 1},
            {'symbol': 'ETH/GBP', 'name': 'Ethereum', 'rank': 2},
            {'symbol': 'SOL/GBP', 'name': 'Solana', 'rank': 3},
            {'symbol': 'XRP/GBP', 'name': 'XRP', 'rank': 4},
            {'symbol': 'LTC/GBP', 'name': 'Litecoin', 'rank': 5},
            {'symbol': 'DOT/GBP', 'name': 'Polkadot', 'rank': 6},
            {'symbol': 'LINK/GBP', 'name': 'Chainlink', 'rank': 7},
            {'symbol': 'AVAX/GBP', 'name': 'Avalanche', 'rank': 8},
            {'symbol': 'ADA/GBP', 'name': 'Cardano', 'rank': 9}
        ]
    }

# Individual crypto endpoint
@app.get("/api/crypto/{crypto_id}/gbp")
async def get_crypto_price(crypto_id: str):
    """Get current price for a specific cryptocurrency."""
    try:
        trader = get_ai_trader()
        pair = f"{crypto_id.upper()}/GBP"
        
        # Get current price
        current_price = trader.get_current_price_for_pair(pair)
        
        if current_price is None:
            raise HTTPException(status_code=404, detail=f"Price not found for {crypto_id}")
        
        return {
            'symbol': pair,
            'price': current_price,
            'currency': 'GBP',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Price error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Enhanced AI Crypto Trader Server...")
    print("📊 Features: Continuous Learning, Performance Tracking, Automatic Retraining")
    print("🔗 API Documentation: http://localhost:8001/docs")
    print("🌐 Frontend: http://localhost:8001")
    
    uvicorn.run(
        "enhanced_web_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
