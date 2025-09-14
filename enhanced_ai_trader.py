"""
Enhanced AI Trader with continuous learning capabilities.
Implements data storage, performance tracking, automatic retraining, 
incremental learning, and feedback loops.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import threading
import time
import logging
import traceback
from prediction_storage import PredictionStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedAITrader:
    def __init__(self):
        # Separate models for each cryptocurrency
        self.models = {}  # Dictionary to store models per pair
        self.scalers = {}  # Dictionary to store scalers per pair
        self.is_trained = {}  # Dictionary to track training status per pair
        self.model_version = "2.0"
        self.storage = PredictionStorage()
        self.retrain_interval_hours = 24  # Retrain every 24 hours
        self.last_retrain = {}
        self.performance_threshold = 0.6  # Minimum accuracy to consider model good
        self.learning_rate = 0.1  # How much to adjust model with new data
        
        # Available pairs for separate models
        self.available_pairs = ['BTC/GBP', 'ETH/GBP', 'SOL/GBP', 'XRP/GBP', 'LTC/GBP', 'DOT/GBP', 'LINK/GBP', 'AVAX/GBP', 'ADA/GBP']
        
        # Initialize models and scalers for each pair
        for pair in self.available_pairs:
            self.models[pair] = None
            self.scalers[pair] = StandardScaler()
            self.is_trained[pair] = False
            self.last_retrain[pair] = None
        
        # Start background learning process
        self.start_background_learning()
    
    def start_background_learning(self):
        """Start background processes for continuous learning."""
        try:
            # Start outcome evaluation thread
            outcome_thread = threading.Thread(target=self.evaluate_outcomes_loop, daemon=True, name="OutcomeEvaluator")
            outcome_thread.start()
            
            # Start retraining thread
            retrain_thread = threading.Thread(target=self.retrain_loop, daemon=True, name="ModelRetrainer")
            retrain_thread.start()
            
            # Start periodic learning data collection
            learning_thread = threading.Thread(target=self.learning_data_loop, daemon=True, name="LearningDataCollector")
            learning_thread.start()
            
            logger.info("🔄 Background learning processes started")
            logger.info(f"   - Outcome Evaluator: {outcome_thread.is_alive()}")
            logger.info(f"   - Model Retrainer: {retrain_thread.is_alive()}")
            logger.info(f"   - Learning Data Collector: {learning_thread.is_alive()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start background learning processes: {e}")
            logger.error(traceback.format_exc())
    
    def evaluate_outcomes_loop(self):
        """Continuously evaluate prediction outcomes."""
        print("🔄 Outcome evaluation loop started")
        while True:
            try:
                print("🔍 Evaluating pending outcomes...")
                self.evaluate_pending_outcomes()
                print("✅ Outcome evaluation completed")
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                print(f"❌ Error in outcome evaluation: {e}")
                time.sleep(300)
    
    def retrain_loop(self):
        """Continuously retrain model with new data."""
        print("🔄 Retraining loop started")
        while True:
            try:
                if self.should_retrain():
                    print("🔄 Starting automatic retraining...")
                    self.retrain_with_learning_data()
                    print("✅ Automatic retraining completed")
                else:
                    print("⏳ Retraining not needed yet")
                time.sleep(1800)  # Check every 30 minutes
            except Exception as e:
                print(f"❌ Error in retraining: {e}")
                time.sleep(1800)
    
    def learning_data_loop(self):
        """Continuously collect learning data and generate predictions for learning."""
        print("🔄 Learning data collection and prediction loop started")
        pairs = ['BTC/GBP', 'ETH/GBP', 'SOL/GBP', 'XRP/GBP', 'LTC/GBP', 'DOT/GBP', 'LINK/GBP', 'AVAX/GBP', 'ADA/GBP']  # All major cryptocurrencies except DOGE
        
        while True:
            try:
                print("📊 Collecting learning data and generating predictions...")
                successful_collections = 0
                successful_predictions = 0
                
                for pair in pairs:
                    try:
                        # Get fresh market data
                        df = self.get_market_data(pair)
                        if len(df) > 0:
                            print(f"✅ Collected data for {pair}: {len(df)} records")
                            successful_collections += 1
                            
                            # Generate prediction for learning (if model is trained)
                            if self.is_trained:
                                try:
                                    prediction_result = self.predict_signal(pair)
                                    if "error" not in prediction_result:
                                        print(f"🔮 Generated learning prediction for {pair}")
                                        successful_predictions += 1
                                    else:
                                        print(f"⚠️ Prediction failed for {pair}: {prediction_result['error']}")
                                except Exception as e:
                                    print(f"❌ Error generating prediction for {pair}: {e}")
                    except Exception as e:
                        print(f"❌ Error collecting data for {pair}: {e}")
                        # Continue with other pairs
                        continue
                
                print(f"✅ Learning cycle completed: {successful_collections}/{len(pairs)} data collections, {successful_predictions} predictions generated")
                time.sleep(900)  # Collect every 15 minutes
            except Exception as e:
                print(f"❌ Error in learning data collection: {e}")
                time.sleep(900)
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        if not self.last_retrain:
            return True
        
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain.total_seconds() > (self.retrain_interval_hours * 3600)
    
    def get_market_data(self, pair: str, limit: int = 100) -> pd.DataFrame:
        """Get real market data from Kraken API with learning data storage."""
        try:
            # Map pairs to Kraken symbols (corrected)
            pair_mapping = {
                'BTC/USD': 'XXBTZUSD', 'BTC/GBP': 'XXBTZGBP', 'BTC/USDT': 'XXBTZUSD',
                'ETH/USD': 'XETHZUSD', 'ETH/GBP': 'XETHZGBP', 'ETH/USDT': 'XETHZUSD',
                'ADA/USD': 'ADAUSD', 'ADA/GBP': 'ADAGBP', 'ADA/USDT': 'ADAUSD',
                'SOL/USD': 'SOLUSD', 'SOL/GBP': 'SOLGBP', 'SOL/USDT': 'SOLUSD',
                'DOGE/USD': 'XXDGZUSD', 'DOGE/GBP': 'XXDGZGBP', 'DOGE/USDT': 'XXDGZUSD',
                'XRP/USD': 'XXRPZUSD', 'XRP/GBP': 'XRPGBP', 'XRP/USDT': 'XXRPZUSD',
                'LTC/GBP': 'LTCGBP', 'DOT/GBP': 'DOTGBP', 'LINK/GBP': 'LINKGBP', 'AVAX/GBP': 'AVAXGBP'
            }
            
            # Fallback pairs for GBP (some might not exist)
            gbp_fallback = {
                'DOGE/GBP': 'XXDGZUSD',  # Use USD price as fallback
                'LINK/GBP': 'LINKUSD',   # Use USD price as fallback
                'AVAX/GBP': 'AVAXUSD'    # Use USD price as fallback
            }
            
            kraken_pair = pair_mapping.get(pair)
            if not kraken_pair:
                raise ValueError(f"Unknown pair: {pair}")
            
            # Get current price from ticker
            current_price = self.get_current_price(kraken_pair)
            
            # If GBP pair fails, try USD fallback
            if current_price is None and pair.endswith('/GBP') and pair in gbp_fallback:
                print(f"⚠️ Trying USD fallback for {pair}")
                usd_pair = gbp_fallback[pair]
                usd_price = self.get_current_price(usd_pair)
                if usd_price is not None:
                    # Convert USD to GBP (rough conversion)
                    current_price = usd_price * 0.8  # Approximate USD to GBP conversion
                    print(f"✅ Converted {usd_pair} price {usd_price} to GBP: {current_price}")
            
            if current_price is None:
                raise ValueError(f"Could not get current price for {pair}")
            
            # Get OHLC data
            df = self.get_ohlc_data(kraken_pair, current_price)
            
            # Store learning data for future retraining
            self.store_learning_data_from_df(pair, df)
            
            return df
                
        except Exception as e:
            print(f"❌ Error getting market data: {e}")
            raise ValueError(f"Failed to get market data for {pair}: {str(e)}")
    
    def get_current_price(self, kraken_pair: str) -> Optional[float]:
        """Get current price from Kraken ticker API."""
        try:
            ticker_url = "https://api.kraken.com/0/public/Ticker"
            ticker_params = {'pair': kraken_pair}
            response = requests.get(ticker_url, params=ticker_params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and kraken_pair in data['result']:
                    price = float(data['result'][kraken_pair]['c'][0])
                    print(f"✅ Got current price for {kraken_pair}: {price}")
                    return price
                elif 'error' in data:
                    print(f"❌ Kraken API error for {kraken_pair}: {data['error']}")
                else:
                    print(f"❌ No data found for {kraken_pair}")
            else:
                print(f"❌ HTTP error {response.status_code} for {kraken_pair}")
            return None
        except Exception as e:
            print(f"❌ Error getting current price for {kraken_pair}: {e}")
            return None
    
    def get_ohlc_data(self, kraken_pair: str, current_price: float) -> pd.DataFrame:
        """Get OHLC data from Kraken API."""
        try:
            ohlc_url = "https://api.kraken.com/0/public/OHLC"
            ohlc_params = {
                'pair': kraken_pair,
                'interval': 1440,  # Daily candles
                'since': int((datetime.now() - timedelta(days=30)).timestamp())
            }
            
            response = requests.get(ohlc_url, params=ohlc_params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and kraken_pair in data['result']:
                    ohlc_records = data['result'][kraken_pair]
                    
                    df = pd.DataFrame(ohlc_records, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df = df.set_index('timestamp')
                    df['price'] = df['close'].astype(float)
                    
                    # Update last price with current price
                    df.iloc[-1, df.columns.get_loc('price')] = current_price
                    
                    # Calculate technical indicators
                    df = self.calculate_technical_indicators(df)
                    
                    return df.dropna()
            
            # Fallback: create minimal data
            return self.create_minimal_data(current_price)
            
        except Exception as e:
            print(f"❌ Error getting OHLC data: {e}")
            return self.create_minimal_data(current_price)
    
    def create_minimal_data(self, current_price: float) -> pd.DataFrame:
        """Create minimal data with current price."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=29), end=datetime.now(), freq='D')
        prices = [current_price] * len(dates)
        
        df = pd.DataFrame({'price': prices}, index=dates)
        return self.calculate_technical_indicators(df)
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        # Basic indicators
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        df['rsi'] = self.calculate_rsi(df['price'])
        df['price_change'] = df['price'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=5).std()
        
        # MACD (Moving Average Convergence Divergence)
        df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df['price'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'], df['bb_position'] = self.calculate_bollinger_bands(df['price'])
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df)
        
        # Williams %R
        df['williams_r'] = self.calculate_williams_r(df)
        
        # Volume indicators (if volume data available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = self.calculate_obv(df)
            df['volume_price_trend'] = self.calculate_vpt(df)
        else:
            # Create mock volume data for demonstration
            df['volume'] = np.random.uniform(1000, 10000, len(df))
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = self.calculate_obv(df)
            df['volume_price_trend'] = self.calculate_vpt(df)
        
        # Additional momentum indicators
        df['momentum'] = df['price'] / df['price'].shift(10)
        df['rate_of_change'] = df['price'].pct_change(periods=10)
        
        return df
    
    def store_learning_data_from_df(self, pair: str, df: pd.DataFrame):
        """Store comprehensive learning data from DataFrame for future retraining."""
        for timestamp, row in df.iterrows():
            features = {
                # Basic indicators
                'sma_5': row.get('sma_5', 0),
                'sma_20': row.get('sma_20', 0),
                'sma_50': row.get('sma_50', 0),
                'rsi': row.get('rsi', 50),
                'price_change': row.get('price_change', 0),
                'volatility': row.get('volatility', 0),
                # MACD
                'macd': row.get('macd', 0),
                'macd_signal': row.get('macd_signal', 0),
                'macd_histogram': row.get('macd_histogram', 0),
                # Bollinger Bands
                'bb_upper': row.get('bb_upper', 0),
                'bb_middle': row.get('bb_middle', 0),
                'bb_lower': row.get('bb_lower', 0),
                'bb_width': row.get('bb_width', 0),
                'bb_position': row.get('bb_position', 0.5),
                # Stochastic
                'stoch_k': row.get('stoch_k', 50),
                'stoch_d': row.get('stoch_d', 50),
                # Williams %R
                'williams_r': row.get('williams_r', -50),
                # Volume indicators
                'volume_sma': row.get('volume_sma', 0),
                'volume_ratio': row.get('volume_ratio', 1),
                'obv': row.get('obv', 0),
                'volume_price_trend': row.get('volume_price_trend', 0),
                # Momentum
                'momentum': row.get('momentum', 1),
                'rate_of_change': row.get('rate_of_change', 0)
            }
            
            # Determine actual outcome (1 for positive change, 0 for negative)
            actual_outcome = 1 if row.get('price_change', 0) > 0 else 0
            
            self.storage.store_learning_data(pair, timestamp, row['price'], features, actual_outcome)
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        width = (upper - lower) / sma
        position = (prices - lower) / (upper - lower)
        return upper, sma, lower, width, position
    
    def calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> tuple:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_window).min()
        high_max = df['high'].rolling(window=k_window).max()
        k_percent = 100 * ((df['price'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df['high'].rolling(window=window).max()
        low_min = df['low'].rolling(window=window).min()
        williams_r = -100 * ((high_max - df['price']) / (high_max - low_min))
        return williams_r
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = np.zeros(len(df))
        obv[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['price'].iloc[i] > df['price'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['price'].iloc[i] < df['price'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index)
    
    def calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        vpt = np.zeros(len(df))
        vpt[0] = 0
        
        for i in range(1, len(df)):
            price_change = (df['price'].iloc[i] - df['price'].iloc[i-1]) / df['price'].iloc[i-1]
            vpt[i] = vpt[i-1] + (df['volume'].iloc[i] * price_change)
        
        return pd.Series(vpt, index=df.index)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare comprehensive features for XGBoost model."""
        features = [
            # Basic indicators
            'sma_5', 'sma_20', 'sma_50', 'rsi', 'price_change', 'volatility',
            # MACD
            'macd', 'macd_signal', 'macd_histogram',
            # Bollinger Bands
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            # Stochastic
            'stoch_k', 'stoch_d',
            # Williams %R
            'williams_r',
            # Volume indicators
            'volume_sma', 'volume_ratio', 'obv', 'volume_price_trend',
            # Momentum
            'momentum', 'rate_of_change'
        ]
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].fillna(0).values
        return X
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Create labels for training."""
        future_prices = df['price'].shift(-1)
        price_change = (future_prices - df['price']) / df['price']
        labels = np.where(price_change > 0.01, 1, 0)  # Buy if price increases > 1%
        return labels[:-1]
    
    def train_model(self, pair: str = "BTC/GBP") -> Dict[str, Any]:
        """Train separate XGBoost model for specific cryptocurrency pair."""
        try:
            if pair not in self.available_pairs:
                return {"success": False, "error": f"Pair {pair} not supported"}
            
            print(f"🔄 Training enhanced model for {pair}...")
            
            # Get fresh market data
            df = self.get_market_data(pair)
            print(f"📊 Got market data with {len(df)} rows")
            
            if len(df) < 10:
                return {"success": False, "error": f"Insufficient data: {len(df)} rows"}
            
            # Prepare features and labels
            X = self.prepare_features(df)
            y = self.create_labels(df)
            
            # Ensure same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            
            # Scale features using pair-specific scaler
            X_scaled = self.scalers[pair].fit_transform(X)
            
            # Train pair-specific model
            self.models[pair] = xgb.XGBClassifier(
                n_estimators=150,  # Increased for better performance
                max_depth=8,       # Increased for more complex patterns
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.models[pair].fit(X_scaled, y)
            self.is_trained[pair] = True
            self.last_retrain[pair] = datetime.now()
            
            # Evaluate performance
            predictions = self.models[pair].predict(X_scaled)
            accuracy = accuracy_score(y, predictions)
            
            print(f"✅ Model training completed for {pair} - Accuracy: {accuracy:.3f}")
            
            return {
                "success": True,
                "accuracy": float(accuracy),
                "samples": len(X),
                "model_version": self.model_version,
                "pair": pair,
                "features": self.get_feature_names()
            }
            
        except Exception as e:
            print(f"❌ Training error for {pair}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names used in the model."""
        return [
            'sma_5', 'sma_20', 'sma_50', 'rsi', 'price_change', 'volatility',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'stoch_k', 'stoch_d', 'williams_r',
            'volume_sma', 'volume_ratio', 'obv', 'volume_price_trend',
            'momentum', 'rate_of_change'
        ]
    
    def retrain_with_learning_data(self):
        """Retrain model using stored learning data."""
        try:
            print("🔄 Retraining with learning data...")
            
            # Check if model is already being trained
            if not self.is_trained and self.model is None:
                print("⚠️ No existing model to retrain, skipping...")
                return
            
            # Get learning data for all pairs
            pairs = ['BTC/GBP', 'ETH/GBP', 'SOL/GBP', 'XRP/GBP', 'LTC/GBP', 'DOT/GBP', 'LINK/GBP', 'AVAX/GBP', 'ADA/GBP']
            
            # Collect all learning data
            all_learning_data = []
            for pair in pairs:
                learning_df = self.storage.get_learning_data(pair, days=30)
                if len(learning_df) >= 10:  # Need minimum data
                    learning_df['pair'] = pair
                    all_learning_data.append(learning_df)
            
            if not all_learning_data:
                print("⚠️ No learning data available for retraining")
                return
            
            # Combine all data
            combined_df = pd.concat(all_learning_data, ignore_index=True)
            
            if len(combined_df) < 20:
                print(f"⚠️ Insufficient combined data: {len(combined_df)} records")
                return
            
            # Prepare features and labels
            feature_cols = ['sma_5', 'sma_20', 'rsi', 'price_change', 'volatility']
            X = combined_df[feature_cols].fillna(0).values
            y = combined_df['actual_outcome'].fillna(0).values
            
            if len(X) < 10:
                print(f"⚠️ Insufficient features: {len(X)} records")
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize new model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self.last_retrain = datetime.now()
            
            # Evaluate performance (only if model is trained)
            if self.model is not None and hasattr(self.model, 'predict'):
                predictions = self.model.predict(X_scaled)
                accuracy = accuracy_score(y, predictions)
            else:
                print("⚠️ Model not available for evaluation")
                accuracy = 0.0
            
            print(f"✅ Retrained model with {len(combined_df)} records - Accuracy: {accuracy:.3f}")
            
            # Update model performance for each pair
            for pair in pairs:
                pair_data = combined_df[combined_df['pair'] == pair]
                if len(pair_data) > 0:
                    pair_X = pair_data[feature_cols].fillna(0).values
                    pair_X_scaled = self.scaler.transform(pair_X)
                    pair_predictions = self.model.predict(pair_X_scaled)
                    pair_accuracy = accuracy_score(pair_data['actual_outcome'].fillna(0).values, pair_predictions)
                    
                    self.storage.update_model_performance(
                        pair, datetime.now().strftime('%Y-%m-%d'),
                        len(pair_data), int(pair_accuracy * len(pair_data)), 0.8, self.model_version
                    )
            
        except Exception as e:
            print(f"❌ Retraining error: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_signal(self, pair: str) -> Dict[str, Any]:
        """Make prediction with enhanced learning capabilities using pair-specific model."""
        try:
            print(f"🔮 Predicting signal for {pair}")
            
            if pair not in self.available_pairs:
                return {"error": f"Pair {pair} not supported"}
            
            if not self.is_trained.get(pair, False):
                print(f"⚠️ Model for {pair} not trained, training now...")
                train_result = self.train_model(pair)
                if not train_result["success"]:
                    return {"error": f"Failed to train model for {pair}: {train_result['error']}"}
            
            # Get market data
            df = self.get_market_data(pair)
            if len(df) < 5:
                return {"error": "Insufficient data for prediction"}
            
            # Prepare features
            X = self.prepare_features(df)
            X_scaled = self.scalers[pair].transform(X)
            
            # Make prediction using pair-specific model
            prediction_proba = self.models[pair].predict_proba(X_scaled[-1:])[0]
            prediction = self.models[pair].predict(X_scaled[-1:])[0]
            
            # Get current price
            current_price = float(df['price'].iloc[-1])
            
            # Calculate predicted price with enhanced algorithm
            volatility = df['volatility'].iloc[-1] if not pd.isna(df['volatility'].iloc[-1]) else 0.02
            rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
            
            # Enhanced price prediction based on signal and market conditions
            if prediction == 1:  # Buy signal
                base_change = 0.02 + (volatility * 2)  # More aggressive for buy
                if rsi < 30:  # Oversold
                    base_change += 0.01
            else:  # Hold/Sell signal
                base_change = -0.01 + (volatility * 0.5)  # Conservative for hold/sell
                if rsi > 70:  # Overbought
                    base_change -= 0.01
            
            predicted_price = current_price * (1 + base_change)
            price_change_pct = (predicted_price - current_price) / current_price * 100
            
            # Generate 6-hour predictions
            hourly_predictions = self.generate_6hour_predictions(pair, current_price, base_change)
            
            # Create prediction data
            prediction_data = {
                'pair': pair,
                'signal': 'buy' if prediction == 1 else 'hold',
                'confidence': float(prediction_proba.max()),
                'current_price': current_price,  # Fixed: was 'price'
                'predicted_price': predicted_price,
                'predicted_change_pct': price_change_pct,  # Fixed: was 'price_change_pct'
                'timestamp': datetime.now().isoformat(),
                'reasoning': f'Enhanced XGBoost prediction (volatility: {volatility:.4f}, RSI: {rsi:.1f})',
                'hourly_predictions': hourly_predictions,
                'features': {
                    'sma_5': float(df['sma_5'].iloc[-1]) if not pd.isna(df['sma_5'].iloc[-1]) else 0,
                    'sma_20': float(df['sma_20'].iloc[-1]) if not pd.isna(df['sma_20'].iloc[-1]) else 0,
                    'rsi': float(rsi),
                    'price_change': float(df['price_change'].iloc[-1]) if not pd.isna(df['price_change'].iloc[-1]) else 0,
                    'volatility': float(volatility)
                },
                'model_version': self.model_version
            }
            
            # Store prediction for future evaluation
            prediction_id = self.storage.store_prediction(prediction_data)
            prediction_data['prediction_id'] = prediction_id
            
            print(f"✅ Prediction stored with ID: {prediction_id}")
            
            return prediction_data
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {"error": str(e)}
    
    def generate_6hour_predictions(self, pair: str, current_price: float, base_change: float) -> List[Dict]:
        """Generate 6-hour price predictions with 15-minute intervals."""
        predictions = []
        current_time = datetime.now()
        
        for i in range(24):  # 24 intervals of 15 minutes = 6 hours
            # Add some variation to the base change
            time_factor = i / 24
            volatility_factor = np.random.normal(0, 0.005)  # Small random variation
            
            # Gradual change over time
            interval_change = base_change * (1 - time_factor * 0.5) + volatility_factor
            price = current_price * (1 + interval_change * (i + 1) / 24)
            
            prediction_time = current_time + timedelta(minutes=15 * (i + 1))
            
            predictions.append({
                'timestamp': prediction_time.isoformat(),
                'price': round(price, 6),
                'change_pct': round((price - current_price) / current_price * 100, 2),
                'interval': f"{i + 1}/24"
            })
        
        return predictions
    
    def evaluate_pending_outcomes(self):
        """Evaluate outcomes of pending predictions."""
        try:
            pending_predictions = self.storage.get_pending_outcomes(hours=1)
            
            for prediction_id, pair, current_price, predicted_price, created_at in pending_predictions:
                # Get current price to compare
                try:
                    current_market_price = self.get_current_price_for_pair(pair)
                    if current_market_price:
                        self.storage.store_outcome(prediction_id, current_market_price, 60)
                        print(f"✅ Evaluated prediction {prediction_id} for {pair}")
                except Exception as e:
                    print(f"❌ Error evaluating prediction {prediction_id}: {e}")
                    
        except Exception as e:
            print(f"❌ Error in outcome evaluation: {e}")
    
    def get_current_price_for_pair(self, pair: str) -> Optional[float]:
        """Get current price for a specific pair."""
        try:
            pair_mapping = {
                'BTC/GBP': 'XXBTZGBP', 'ETH/GBP': 'XETHZGBP', 'SOL/GBP': 'SOLGBP',
                'XRP/GBP': 'XRPGBP', 'DOGE/GBP': 'XXDGZGBP', 'LTC/GBP': 'LTCGBP',
                'DOT/GBP': 'DOTGBP', 'LINK/GBP': 'LINKGBP', 'AVAX/GBP': 'AVAXGBP', 'ADA/GBP': 'ADAGBP'
            }
            
            kraken_pair = pair_mapping.get(pair)
            if kraken_pair:
                return self.get_current_price(kraken_pair)
            return None
        except Exception as e:
            print(f"❌ Error getting current price for {pair}: {e}")
            return None
    
    def get_performance_metrics(self, pair: str = None) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            if pair:
                pairs = [pair]
            else:
                pairs = self.available_pairs
            
            metrics = {}
            for p in pairs:
                performance = self.storage.get_prediction_accuracy(p, days=7)
                metrics[p] = performance
            
            # Calculate overall training status
            trained_pairs = sum(1 for p in pairs if self.is_trained.get(p, False))
            overall_trained = trained_pairs > 0
            
            return {
                'performance_by_pair': metrics,
                'model_version': self.model_version,
                'last_retrain': {p: self.last_retrain[p].isoformat() if self.last_retrain[p] else None for p in pairs},
                'is_trained': overall_trained,
                'trained_pairs': trained_pairs,
                'total_pairs': len(pairs)
            }
            
        except Exception as e:
            print(f"❌ Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self):
        """Clean up old data to maintain performance."""
        try:
            self.storage.cleanup_old_data(days=90)
            print("🧹 Cleaned up old data")
        except Exception as e:
            print(f"❌ Error cleaning up data: {e}")
