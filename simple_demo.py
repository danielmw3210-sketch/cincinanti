#!/usr/bin/env python3
"""Simple demo of AI Crypto Trader - No dependencies required."""

import math
import random
from datetime import datetime, timedelta

def generate_sample_data(pair="BTC/USD", days=30):
    """Generate realistic sample market data."""
    base_price = 50000 if 'BTC' in pair else 3000
    data = []
    
    # Generate price series with trend and volatility
    current_price = base_price
    current_time = datetime.now() - timedelta(days=days)
    
    for i in range(days * 24):  # Hourly data
        # Add some trend and volatility
        change = random.gauss(0, 0.02)  # 2% volatility
        current_price *= (1 + change)
        
        # Generate OHLC for this hour
        volatility = abs(random.gauss(0, 0.01))
        open_price = current_price
        close_price = current_price * (1 + random.gauss(0, 0.005))
        high_price = max(open_price, close_price) * (1 + volatility)
        low_price = min(open_price, close_price) * (1 - volatility)
        volume = random.randint(1000, 10000)
        
        data.append({
            'timestamp': current_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_time += timedelta(hours=1)
    
    return data

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    if len(prices) < window:
        return None
    
    return sum(prices[-window:]) / window

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index."""
    if len(prices) < window + 1:
        return 50
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < window:
        return 50
    
    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices):
    """Calculate MACD (simplified)."""
    if len(prices) < 26:
        return 0
    
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    
    if ema_12 is None or ema_26 is None:
        return 0
    
    return ema_12 - ema_26

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    if len(prices) < window:
        return None
    
    multiplier = 2 / (window + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands."""
    if len(prices) < window:
        return None, None, None
    
    sma = calculate_sma(prices, window)
    recent_prices = prices[-window:]
    
    # Calculate standard deviation
    variance = sum((price - sma) ** 2 for price in recent_prices) / window
    std = math.sqrt(variance)
    
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    return upper_band, sma, lower_band

def analyze_market(data):
    """Analyze market data and generate trading signal."""
    if len(data) < 50:
        return "Insufficient data for analysis"
    
    # Extract prices
    closes = [d['close'] for d in data]
    volumes = [d['volume'] for d in data]
    
    # Calculate indicators
    sma_20 = calculate_sma(closes, 20)
    sma_50 = calculate_sma(closes, 50)
    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
    
    current_price = closes[-1]
    
    # Generate signal
    signal_score = 0
    reasoning = []
    
    # RSI analysis
    if rsi < 30:
        signal_score += 0.3
        reasoning.append("RSI indicates oversold conditions (buy signal)")
    elif rsi > 70:
        signal_score -= 0.3
        reasoning.append("RSI indicates overbought conditions (sell signal)")
    else:
        reasoning.append(f"RSI is neutral ({rsi:.1f})")
    
    # MACD analysis
    if macd > 0:
        signal_score += 0.2
        reasoning.append("MACD shows bullish momentum")
    else:
        signal_score -= 0.2
        reasoning.append("MACD shows bearish momentum")
    
    # Moving average analysis
    if sma_20 and sma_50:
        if sma_20 > sma_50:
            signal_score += 0.2
            reasoning.append("Short-term trend is bullish")
        else:
            signal_score -= 0.2
            reasoning.append("Short-term trend is bearish")
    
    # Bollinger Bands analysis
    if bb_upper and bb_lower:
        if current_price > bb_upper:
            signal_score -= 0.1
            reasoning.append("Price above upper Bollinger Band (overbought)")
        elif current_price < bb_lower:
            signal_score += 0.1
            reasoning.append("Price below lower Bollinger Band (oversold)")
        else:
            reasoning.append("Price within Bollinger Bands (normal)")
    
    # Determine final signal
    if signal_score > 0.3:
        signal = "BUY"
        confidence = min(signal_score, 1.0)
    elif signal_score < -0.3:
        signal = "SELL"
        confidence = min(abs(signal_score), 1.0)
    else:
        signal = "HOLD"
        confidence = 0.5
    
    return {
        'signal': signal,
        'confidence': confidence,
        'score': signal_score,
        'reasoning': reasoning,
        'indicators': {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'macd': macd,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower
        }
    }

def run_demo(pair="BTC/USD"):
    """Run the demo analysis."""
    print(f"🤖 AI Crypto Trader - Demo Mode")
    print(f"📊 Analyzing {pair} (Simulated Data)")
    print("=" * 60)
    
    # Generate sample data
    print("📈 Generating sample market data...")
    data = generate_sample_data(pair, days=30)
    print(f"✅ Generated {len(data)} data points")
    
    # Analyze market
    print("🧮 Calculating technical indicators...")
    analysis = analyze_market(data)
    
    if isinstance(analysis, str):
        print(f"❌ {analysis}")
        return
    
    # Display results
    print(f"\n📊 MARKET ANALYSIS RESULTS")
    print("=" * 60)
    
    indicators = analysis['indicators']
    print(f"💰 Current Price: ${indicators['current_price']:,.2f}")
    print(f"📅 Data Points: {len(data)}")
    
    print(f"\n📈 TECHNICAL INDICATORS")
    print("-" * 40)
    print(f"SMA 20: ${indicators['sma_20']:,.2f}" if indicators['sma_20'] else "SMA 20: N/A")
    print(f"SMA 50: ${indicators['sma_50']:,.2f}" if indicators['sma_50'] else "SMA 50: N/A")
    print(f"RSI: {indicators['rsi']:.1f}")
    print(f"MACD: {indicators['macd']:.4f}")
    
    if indicators['bb_upper']:
        print(f"Bollinger Upper: ${indicators['bb_upper']:,.2f}")
        print(f"Bollinger Middle: ${indicators['bb_middle']:,.2f}")
        print(f"Bollinger Lower: ${indicators['bb_lower']:,.2f}")
    
    print(f"\n🤖 AI TRADING SIGNAL")
    print("-" * 40)
    print(f"Signal: {analysis['signal']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Score: {analysis['score']:.2f}")
    
    print(f"\n💭 REASONING")
    print("-" * 40)
    for reason in analysis['reasoning']:
        print(f"• {reason}")
    
    print(f"\n✅ Demo completed!")
    print(f"💡 This uses simulated data - get API keys for real market data")
    print(f"🔗 To get API keys: https://pro.kraken.com/app/settings/api")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Crypto Trader Simple Demo")
    parser.add_argument("--pair", "-p", default="BTC/USD", help="Trading pair to analyze")
    
    args = parser.parse_args()
    
    run_demo(args.pair)

if __name__ == "__main__":
    main()