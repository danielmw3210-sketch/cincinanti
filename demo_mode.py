#!/usr/bin/env python3
"""Demo mode for AI Crypto Trader - No API keys required."""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_analyzer import MarketAnalyzer
from ai_trader import AITrader
from config import config

class DemoKrakenClient:
    """Demo Kraken client that generates sample data."""
    
    def __init__(self):
        self.base_url = "https://api.kraken.com"
        
    def get_server_time(self):
        """Get demo server time."""
        return {'unixtime': int(datetime.now().timestamp())}
    
    def get_ticker(self, pair):
        """Get demo ticker data."""
        # Generate realistic sample data
        base_price = 50000 if 'BTC' in pair else 3000
        price_variation = np.random.normal(0, 0.02)  # 2% variation
        current_price = base_price * (1 + price_variation)
        
        return {
            pair.replace('/', ''): {
                'c': [str(current_price), '0.1'],  # [price, lot volume]
                'v': ['100.0', '200.0'],  # [volume today, volume 24h]
                'p': [str(current_price * 1.01), str(current_price * 0.99)],  # [high, low]
                't': [100, 200],  # [trades today, trades 24h]
                'l': [str(current_price * 0.98)],  # [low today]
                'h': [str(current_price * 1.02)],  # [high today]
                'o': str(current_price * 0.99)  # [open today]
            }
        }
    
    def get_ohlc_data(self, pair, interval=1, limit=1000):
        """Generate demo OHLC data."""
        # Generate realistic price data
        np.random.seed(42)  # For consistent demo data
        base_price = 50000 if 'BTC' in pair else 3000
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0, 0.02, limit)  # 2% volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        ohlc_data = []
        current_time = int(datetime.now().timestamp()) - (limit * interval * 60)
        
        for i in range(limit):
            price = prices[i]
            volatility = abs(np.random.normal(0, 0.01))
            
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + volatility)
            low_price = min(open_price, close_price) * (1 - volatility)
            volume = np.random.randint(1000, 10000)
            
            ohlc_data.append({
                'timestamp': current_time + (i * interval * 60),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return ohlc_data

def run_demo_analysis(pair="BTC/USD"):
    """Run demo analysis without API keys."""
    print(f"🔍 Running Demo Analysis for {pair}")
    print("=" * 50)
    
    try:
        # Create demo client
        demo_client = DemoKrakenClient()
        
        # Create market analyzer with demo client
        analyzer = MarketAnalyzer(demo_client)
        
        # Get market data
        print(f"📊 Collecting market data for {pair}...")
        df = analyzer.collect_market_data(pair, limit=200)
        
        if df.empty:
            print("❌ No data collected")
            return
        
        print(f"✅ Collected {len(df)} data points")
        
        # Calculate technical indicators
        print("🧮 Calculating technical indicators...")
        df_with_indicators = analyzer.calculate_technical_indicators(df)
        
        # Detect patterns
        print("🔍 Detecting candlestick patterns...")
        patterns = analyzer.detect_patterns(df_with_indicators)
        
        # Calculate market sentiment
        print("📈 Analyzing market sentiment...")
        sentiment = analyzer.calculate_market_sentiment(df_with_indicators)
        
        # Get current ticker
        ticker = demo_client.get_ticker(pair)
        
        # Display results
        print("\n📊 MARKET ANALYSIS RESULTS")
        print("=" * 50)
        
        # Current price info
        ticker_data = list(ticker.values())[0]
        current_price = float(ticker_data['c'][0])
        volume_24h = float(ticker_data['v'][1])
        
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"📊 24h Volume: {volume_24h:,.0f}")
        print(f"📅 Data Points: {len(df)}")
        
        # Technical indicators
        if not df_with_indicators.empty:
            latest = df_with_indicators.iloc[-1]
            
            print(f"\n📈 TECHNICAL INDICATORS")
            print("-" * 30)
            print(f"SMA 20: ${latest.get('sma_20', 0):,.2f}")
            print(f"SMA 50: ${latest.get('sma_50', 0):,.2f}")
            print(f"RSI: {latest.get('rsi', 0):.2f}")
            print(f"MACD: {latest.get('macd', 0):.4f}")
            print(f"BB Upper: ${latest.get('bb_upper', 0):,.2f}")
            print(f"BB Lower: ${latest.get('bb_lower', 0):,.2f}")
            print(f"ATR: ${latest.get('atr', 0):,.2f}")
        
        # Patterns
        if patterns:
            print(f"\n🔍 DETECTED PATTERNS")
            print("-" * 30)
            for pattern, detected in patterns.items():
                if detected:
                    print(f"✅ {pattern.replace('_', ' ').title()}")
        
        # Sentiment
        if sentiment:
            print(f"\n📊 MARKET SENTIMENT")
            print("-" * 30)
            trend_sentiment = sentiment.get('trend_sentiment', 0)
            momentum_sentiment = sentiment.get('momentum_sentiment', 0)
            volatility_sentiment = sentiment.get('volatility_sentiment', 0)
            
            print(f"Trend: {'Bullish' if trend_sentiment > 0 else 'Bearish' if trend_sentiment < 0 else 'Neutral'}")
            print(f"Momentum: {'Overbought' if momentum_sentiment > 0.5 else 'Oversold' if momentum_sentiment < -0.5 else 'Neutral'}")
            print(f"Volatility: {'High' if volatility_sentiment > 0.1 else 'Low' if volatility_sentiment < -0.1 else 'Normal'}")
        
        # AI Signal (simulated)
        print(f"\n🤖 AI TRADING SIGNAL (DEMO)")
        print("-" * 30)
        
        # Simple signal based on indicators
        if not df_with_indicators.empty:
            latest = df_with_indicators.iloc[-1]
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            sma_20 = latest.get('sma_20', current_price)
            sma_50 = latest.get('sma_50', current_price)
            
            signal_score = 0
            
            # RSI signal
            if rsi < 30:
                signal_score += 0.3  # Oversold - buy signal
            elif rsi > 70:
                signal_score -= 0.3  # Overbought - sell signal
            
            # MACD signal
            if macd > 0:
                signal_score += 0.2  # Bullish MACD
            else:
                signal_score -= 0.2  # Bearish MACD
            
            # Moving average signal
            if sma_20 > sma_50:
                signal_score += 0.2  # Bullish trend
            else:
                signal_score -= 0.2  # Bearish trend
            
            # Price vs moving averages
            if current_price > sma_20:
                signal_score += 0.1
            else:
                signal_score -= 0.1
            
            # Determine signal
            if signal_score > 0.3:
                signal = "BUY"
                confidence = min(signal_score, 1.0)
            elif signal_score < -0.3:
                signal = "SELL"
                confidence = min(abs(signal_score), 1.0)
            else:
                signal = "HOLD"
                confidence = 0.5
            
            print(f"Signal: {signal}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Score: {signal_score:.2f}")
            
            # Reasoning
            print(f"\n💭 REASONING")
            print("-" * 30)
            if rsi < 30:
                print("• RSI indicates oversold conditions")
            elif rsi > 70:
                print("• RSI indicates overbought conditions")
            
            if macd > 0:
                print("• MACD shows bullish momentum")
            else:
                print("• MACD shows bearish momentum")
            
            if sma_20 > sma_50:
                print("• Short-term trend is bullish")
            else:
                print("• Short-term trend is bearish")
        
        print(f"\n✅ Demo analysis completed!")
        print(f"💡 This is simulated data - get API keys for real market data")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Crypto Trader Demo Mode")
    parser.add_argument("--pair", "-p", default="BTC/USD", help="Trading pair to analyze")
    
    args = parser.parse_args()
    
    print("🤖 AI Crypto Trader - Demo Mode")
    print("📊 No API keys required - using simulated data")
    print("=" * 50)
    
    run_demo_analysis(args.pair)

if __name__ == "__main__":
    main()