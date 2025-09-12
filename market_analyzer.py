"""Market data collection and technical analysis module."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import ta
from loguru import logger
from kraken_client import KrakenClient
from config import config

class MarketAnalyzer:
    """Handles market data collection and technical analysis."""
    
    def __init__(self, kraken_client: KrakenClient):
        self.client = kraken_client
        self.feature_window = config.feature_window
        
    def collect_market_data(self, pair: str, interval: int = 1, limit: int = 1000) -> pd.DataFrame:
        """Collect OHLC market data."""
        try:
            ohlc_data = self.client.get_ohlc_data(pair, interval, limit=limit)
            
            if not ohlc_data:
                logger.warning(f"No market data received for {pair}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlc_data)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            logger.info(f"Collected {len(df)} data points for {pair}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting market data for {pair}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        if df.empty:
            return df
            
        try:
            # Price-based indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_histogram'] = ta.trend.macd(df['close'])
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # Bollinger Bands
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Volume indicators
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Price patterns
            df['price_change'] = df['close'].pct_change()
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            
            # Trend indicators
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            
            # Support and resistance levels
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            
            logger.info(f"Calculated {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Detect common candlestick patterns."""
        if df.empty or len(df) < 3:
            return {}
            
        try:
            patterns = {}
            
            # Get last few candles
            recent = df.tail(3)
            
            # Doji pattern
            patterns['doji'] = abs(recent.iloc[-1]['close'] - recent.iloc[-1]['open']) < (recent.iloc[-1]['high'] - recent.iloc[-1]['low']) * 0.1
            
            # Hammer pattern
            if len(recent) >= 1:
                candle = recent.iloc[-1]
                body_size = abs(candle['close'] - candle['open'])
                lower_shadow = min(candle['open'], candle['close']) - candle['low']
                upper_shadow = candle['high'] - max(candle['open'], candle['close'])
                
                patterns['hammer'] = (lower_shadow > 2 * body_size and upper_shadow < body_size)
                patterns['shooting_star'] = (upper_shadow > 2 * body_size and lower_shadow < body_size)
            
            # Engulfing patterns
            if len(recent) >= 2:
                prev_candle = recent.iloc[-2]
                curr_candle = recent.iloc[-1]
                
                prev_bullish = prev_candle['close'] > prev_candle['open']
                curr_bullish = curr_candle['close'] > curr_candle['open']
                
                patterns['bullish_engulfing'] = (not prev_bullish and curr_bullish and 
                                               curr_candle['open'] < prev_candle['close'] and 
                                               curr_candle['close'] > prev_candle['open'])
                
                patterns['bearish_engulfing'] = (prev_bullish and not curr_bullish and 
                                                curr_candle['open'] > prev_candle['close'] and 
                                                curr_candle['close'] < prev_candle['open'])
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    def calculate_market_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market sentiment indicators."""
        if df.empty:
            return {}
            
        try:
            sentiment = {}
            
            # Trend sentiment
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                current_price = df['close'].iloc[-1]
                
                sentiment['trend_sentiment'] = 1.0 if sma_20 > sma_50 else -1.0
                sentiment['price_vs_sma20'] = (current_price - sma_20) / sma_20
                sentiment['price_vs_sma50'] = (current_price - sma_50) / sma_50
            
            # Momentum sentiment
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi > 70:
                    sentiment['momentum_sentiment'] = -1.0  # Overbought
                elif rsi < 30:
                    sentiment['momentum_sentiment'] = 1.0   # Oversold
                else:
                    sentiment['momentum_sentiment'] = (rsi - 50) / 50
            
            # Volatility sentiment
            if 'bb_width' in df.columns:
                bb_width = df['bb_width'].iloc[-1]
                bb_width_avg = df['bb_width'].rolling(window=20).mean().iloc[-1]
                sentiment['volatility_sentiment'] = bb_width / bb_width_avg - 1
            
            # Volume sentiment
            if 'volume_ratio' in df.columns:
                volume_ratio = df['volume_ratio'].iloc[-1]
                sentiment['volume_sentiment'] = min(volume_ratio - 1, 2.0)  # Cap at 2x
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        if df.empty:
            return pd.DataFrame()
            
        try:
            # Calculate all indicators
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Select relevant features
            feature_columns = [
                'close', 'volume', 'price_change', 'price_range', 'body_size',
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal',
                'rsi', 'bb_width', 'atr', 'volatility', 'adx', 'cci',
                'support_distance', 'resistance_distance', 'volume_ratio'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in df_with_indicators.columns]
            features_df = df_with_indicators[available_features].copy()
            
            # Add lagged features
            for lag in [1, 2, 3, 5]:
                for col in ['close', 'volume', 'price_change']:
                    if col in features_df.columns:
                        features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
            
            # Add rolling statistics
            for window in [5, 10, 20]:
                for col in ['close', 'volume']:
                    if col in features_df.columns:
                        features_df[f'{col}_mean_{window}'] = features_df[col].rolling(window).mean()
                        features_df[f'{col}_std_{window}'] = features_df[col].rolling(window).std()
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            logger.info(f"Prepared {len(features_df.columns)} features for ML model")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def get_market_summary(self, pair: str) -> Dict[str, any]:
        """Get comprehensive market summary."""
        try:
            # Collect data
            df = self.collect_market_data(pair, limit=self.feature_window)
            
            if df.empty:
                return {}
            
            # Calculate indicators
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Detect patterns
            patterns = self.detect_patterns(df_with_indicators)
            
            # Calculate sentiment
            sentiment = self.calculate_market_sentiment(df_with_indicators)
            
            # Get current ticker
            ticker = self.client.get_ticker(pair)
            
            summary = {
                'pair': pair,
                'timestamp': df.index[-1],
                'current_price': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
                'price_change_24h': df['price_change'].iloc[-1] if 'price_change' in df.columns else 0,
                'patterns': patterns,
                'sentiment': sentiment,
                'ticker': ticker,
                'data_points': len(df)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary for {pair}: {e}")
            return {}