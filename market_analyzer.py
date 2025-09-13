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
        """Collect comprehensive market data with multiple timeframes."""
        try:
            # Collect multiple timeframes for better analysis
            timeframes = [1, 5, 15, 60] if interval == 1 else [interval]
            all_data = {}
            
            for tf in timeframes:
                try:
                    ohlc_data = self.client.get_ohlc_data(pair, tf, limit=limit)
                    
                    if ohlc_data:
                        df_tf = pd.DataFrame(ohlc_data)
                        df_tf['datetime'] = pd.to_datetime(df_tf['timestamp'], unit='s')
                        df_tf.set_index('datetime', inplace=True)
                        
                        # Rename columns with timeframe suffix
                        if tf != interval:
                            df_tf.columns = [f"{col}_{tf}m" if col not in ['datetime', 'timestamp'] else col for col in df_tf.columns]
                        
                        all_data[f"tf_{tf}"] = df_tf
                        logger.info(f"Collected {len(df_tf)} data points for {pair} at {tf}m timeframe")
                    
                except Exception as e:
                    logger.warning(f"Error collecting {tf}m data for {pair}: {e}")
                    continue
            
            if not all_data:
                logger.warning(f"No market data received for {pair}")
                return pd.DataFrame()
            
            # Use primary timeframe as base
            primary_tf = f"tf_{interval}"
            if primary_tf in all_data:
                df = all_data[primary_tf].copy()
                
                # Merge other timeframes
                for tf_name, tf_data in all_data.items():
                    if tf_name != primary_tf:
                        # Resample to match primary timeframe
                        tf_data_resampled = tf_data.resample(f"{interval}min").last()
                        df = df.join(tf_data_resampled, how='left', rsuffix=f"_{tf_name}")
                
                # Add additional market data
                df = self._add_market_metadata(df, pair)
                
                logger.info(f"Collected comprehensive data: {len(df)} points, {len(df.columns)} features for {pair}")
                return df
            else:
                # Fallback to first available timeframe
                df = list(all_data.values())[0]
                logger.info(f"Using fallback data: {len(df)} points for {pair}")
                return df
            
        except Exception as e:
            logger.error(f"Error collecting market data for {pair}: {e}")
            return pd.DataFrame()
    
    def _add_market_metadata(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Add additional market metadata and features."""
        try:
            # Add market session information
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add market session indicators
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            # Add volatility regime indicators
            df['volatility_20'] = df['close'].rolling(window=20).std()
            df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8)).astype(int)
            
            # Add trend strength
            df['trend_strength'] = abs(df['close'].rolling(20).mean() - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
            
            # Add volume profile
            df['volume_profile'] = df['volume'].rolling(20).rank(pct=True)
            
            # Add price momentum
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)
            
            # Add market microstructure features
            df['price_impact'] = df['volume'] * abs(df['close'].pct_change())
            df['liquidity_proxy'] = df['volume'] / df['volatility_20']
            
            # Add cross-timeframe features
            if 'close_5m' in df.columns:
                df['price_ratio_1m_5m'] = df['close'] / df['close_5m']
            if 'close_15m' in df.columns:
                df['price_ratio_1m_15m'] = df['close'] / df['close_15m']
            if 'close_60m' in df.columns:
                df['price_ratio_1m_60m'] = df['close'] / df['close_60m']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding market metadata: {e}")
            return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        if df.empty:
            return df
            
        try:
            # Price-based indicators (multiple timeframes)
            for window in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
                df[f'ema_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
            
            # MACD with multiple timeframes
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_histogram'] = ta.trend.macd(df['close'])
            df['macd_12_26'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
            df['macd_5_35'] = ta.trend.macd(df['close'], window_slow=35, window_fast=5)
            
            # RSI with multiple timeframes
            for window in [7, 14, 21, 28]:
                df[f'rsi_{window}'] = ta.momentum.rsi(df['close'], window=window)
            
            # Stochastic Oscillator
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # Bollinger Bands (multiple timeframes)
            for window in [20, 50]:
                df[f'bb_upper_{window}'] = ta.volatility.bollinger_hband(df['close'], window=window)
                df[f'bb_middle_{window}'] = ta.volatility.bollinger_mavg(df['close'], window=window)
                df[f'bb_lower_{window}'] = ta.volatility.bollinger_lband(df['close'], window=window)
                df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
                df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            
            # Keltner Channels
            df['kc_upper'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'])
            df['kc_middle'] = ta.volatility.keltner_channel_mband(df['high'], df['low'], df['close'])
            df['kc_lower'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'])
            
            # Volume indicators (enhanced)
            df['volume_sma_20'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
            df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
            df['fi'] = ta.volume.force_index(df['close'], df['volume'])
            df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'])
            
            # Volatility indicators (enhanced)
            df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['atr_21'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=21)
            df['volatility_20'] = df['close'].rolling(window=20).std()
            df['volatility_50'] = df['close'].rolling(window=50).std()
            df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
            
            # Price patterns (enhanced)
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(periods=2)
            df['price_change_5'] = df['close'].pct_change(periods=5)
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
            df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
            
            # Trend indicators (enhanced)
            df['adx_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            df['adx_21'] = ta.trend.adx(df['high'], df['low'], df['close'], window=21)
            df['cci_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
            df['cci_50'] = ta.trend.cci(df['high'], df['low'], df['close'], window=50)
            df['dpo'] = ta.trend.dpo(df['close'])
            df['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
            df['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
            df['kst'] = ta.trend.kst(df['close'])
            df['mass_index'] = ta.trend.mass_index(df['high'], df['low'])
            df['psar'] = ta.trend.psar_down(df['high'], df['low'], df['close'])
            
            # Momentum indicators
            df['roc_10'] = ta.momentum.roc(df['close'], window=10)
            df['roc_20'] = ta.momentum.roc(df['close'], window=20)
            df['tsi'] = ta.momentum.tsi(df['close'])
            df['ultimate_oscillator'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
            
            # Support and resistance levels (enhanced)
            for window in [20, 50, 100]:
                df[f'support_{window}'] = df['low'].rolling(window=window).min()
                df[f'resistance_{window}'] = df['high'].rolling(window=window).max()
                df[f'support_distance_{window}'] = (df['close'] - df[f'support_{window}']) / df['close']
                df[f'resistance_distance_{window}'] = (df[f'resistance_{window}'] - df['close']) / df['close']
            
            # Market microstructure features
            df['bid_ask_spread'] = (df['high'] - df['low']) / df['close']  # Proxy for spread
            df['price_impact'] = df['volume'] * df['price_change'].abs()
            df['liquidity_ratio'] = df['volume'] / df['atr_14']
            
            # Cross-asset features (if available)
            df['price_momentum_5'] = df['close'].pct_change(5)
            df['price_momentum_10'] = df['close'].pct_change(10)
            df['price_momentum_20'] = df['close'].pct_change(20)
            
            # Volatility clustering
            df['volatility_cluster'] = df['price_change'].rolling(window=5).std()
            df['volatility_regime'] = (df['volatility_20'] > df['volatility_50']).astype(int)
            
            # Market regime indicators
            df['trend_regime'] = ((df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])).astype(int)
            df['volatility_regime_high'] = (df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8)).astype(int)
            
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
        """Prepare comprehensive features for ML model."""
        if df.empty:
            return pd.DataFrame()
            
        try:
            # Calculate all indicators
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Select comprehensive feature set
            feature_columns = [
                # Price and volume
                'close', 'volume', 'price_change', 'price_change_2', 'price_change_5',
                'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
                
                # Moving averages (multiple timeframes)
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
                'ema_5', 'ema_10', 'ema_12', 'ema_20', 'ema_26', 'ema_50', 'ema_100', 'ema_200',
                
                # MACD variants
                'macd', 'macd_signal', 'macd_histogram', 'macd_12_26', 'macd_5_35',
                
                # RSI variants
                'rsi_7', 'rsi_14', 'rsi_21', 'rsi_28',
                
                # Stochastic and Williams
                'stoch_k', 'stoch_d', 'williams_r',
                
                # Bollinger Bands
                'bb_width_20', 'bb_width_50', 'bb_position_20', 'bb_position_50',
                
                # Keltner Channels
                'kc_upper', 'kc_middle', 'kc_lower',
                
                # Volume indicators
                'volume_ratio', 'obv', 'ad', 'cmf', 'fi', 'eom',
                
                # Volatility
                'atr_14', 'atr_21', 'volatility_20', 'volatility_50', 'volatility_ratio',
                'volatility_cluster', 'volatility_regime',
                
                # Trend indicators
                'adx_14', 'adx_21', 'cci_20', 'cci_50', 'dpo', 'kst', 'mass_index',
                'ichimoku_a', 'ichimoku_b',
                
                # Momentum
                'roc_10', 'roc_20', 'tsi', 'ultimate_oscillator',
                
                # Support/Resistance
                'support_distance_20', 'support_distance_50', 'support_distance_100',
                'resistance_distance_20', 'resistance_distance_50', 'resistance_distance_100',
                
                # Market microstructure
                'bid_ask_spread', 'price_impact', 'liquidity_ratio',
                
                # Regime indicators
                'trend_regime', 'volatility_regime_high',
                
                # Price momentum
                'price_momentum_5', 'price_momentum_10', 'price_momentum_20'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in df_with_indicators.columns]
            features_df = df_with_indicators[available_features].copy()
            
            # Add lagged features (more comprehensive)
            lag_features = ['close', 'volume', 'price_change', 'rsi_14', 'macd', 'bb_position_20']
            for lag in [1, 2, 3, 5, 10]:
                for col in lag_features:
                    if col in features_df.columns:
                        features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
            
            # Add rolling statistics (enhanced)
            rolling_features = ['close', 'volume', 'price_change', 'rsi_14']
            for window in [3, 5, 10, 20, 50]:
                for col in rolling_features:
                    if col in features_df.columns:
                        features_df[f'{col}_mean_{window}'] = features_df[col].rolling(window).mean()
                        features_df[f'{col}_std_{window}'] = features_df[col].rolling(window).std()
                        features_df[f'{col}_min_{window}'] = features_df[col].rolling(window).min()
                        features_df[f'{col}_max_{window}'] = features_df[col].rolling(window).max()
                        features_df[f'{col}_quantile_25_{window}'] = features_df[col].rolling(window).quantile(0.25)
                        features_df[f'{col}_quantile_75_{window}'] = features_df[col].rolling(window).quantile(0.75)
            
            # Add technical indicator ratios
            if 'sma_20' in features_df.columns and 'sma_50' in features_df.columns:
                features_df['sma_ratio_20_50'] = features_df['sma_20'] / features_df['sma_50']
            if 'ema_12' in features_df.columns and 'ema_26' in features_df.columns:
                features_df['ema_ratio_12_26'] = features_df['ema_12'] / features_df['ema_26']
            if 'rsi_14' in features_df.columns:
                features_df['rsi_normalized'] = (features_df['rsi_14'] - 50) / 50
            
            # Add price position features
            if 'bb_upper_20' in features_df.columns and 'bb_lower_20' in features_df.columns:
                features_df['price_bb_position'] = (features_df['close'] - features_df['bb_lower_20']) / (features_df['bb_upper_20'] - features_df['bb_lower_20'])
            
            # Add volatility-adjusted features
            if 'atr_14' in features_df.columns:
                features_df['price_change_atr_ratio'] = features_df['price_change'].abs() / features_df['atr_14']
                features_df['volume_atr_ratio'] = features_df['volume'] / features_df['atr_14']
            
            # Add market regime interactions
            if 'trend_regime' in features_df.columns and 'volatility_regime_high' in features_df.columns:
                features_df['regime_interaction'] = features_df['trend_regime'] * features_df['volatility_regime_high']
            
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