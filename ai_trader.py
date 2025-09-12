"""AI trading strategy and decision engine."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime, timedelta
from loguru import logger
from market_analyzer import MarketAnalyzer
from config import config

class AITrader:
    """AI-powered trading decision engine."""
    
    def __init__(self, market_analyzer: MarketAnalyzer):
        self.analyzer = market_analyzer
        self.models = {}
        self.scalers = {}
        self.model_type = config.model_type
        self.feature_window = config.feature_window
        self.retrain_interval = config.retrain_interval
        self.last_retrain = None
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create trading labels based on future price movements."""
        if df.empty:
            return pd.Series()
            
        try:
            # Calculate future returns
            future_returns = df['close'].shift(-horizon) / df['close'] - 1
            
            # Create labels: 1 for buy, 0 for hold, -1 for sell
            labels = pd.Series(0, index=df.index)  # Default to hold
            
            # Buy signal: strong upward movement
            labels[future_returns > 0.02] = 1
            
            # Sell signal: strong downward movement
            labels[future_returns < -0.02] = -1
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series()
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for ML models."""
        try:
            # Prepare features
            features_df = self.analyzer.prepare_features(df)
            
            if features_df.empty:
                return pd.DataFrame(), pd.Series()
            
            # Create labels
            labels = self.create_labels(features_df)
            
            # Align features and labels
            common_index = features_df.index.intersection(labels.index)
            features_aligned = features_df.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            # Remove rows with NaN labels
            valid_mask = ~labels_aligned.isna()
            features_clean = features_aligned[valid_mask]
            labels_clean = labels_aligned[valid_mask]
            
            logger.info(f"Prepared training data: {len(features_clean)} samples, {len(features_clean.columns)} features")
            return features_clean, labels_clean
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, pair: str) -> Dict[str, float]:
        """Train ML models on historical data."""
        try:
            # Collect more data for training
            df = self.analyzer.collect_market_data(pair, limit=self.feature_window * 2)
            
            if df.empty:
                logger.warning(f"No data available for training on {pair}")
                return {}
            
            # Prepare training data
            X, y = self.prepare_training_data(df)
            
            if X.empty or y.empty:
                logger.warning(f"Insufficient data for training on {pair}")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            accuracies = {}
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    accuracies[model_name] = accuracy
                    
                    logger.info(f"Trained {model_name} with accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    accuracies[model_name] = 0.0
            
            # Save models
            self._save_models(pair)
            
            self.last_retrain = datetime.now()
            logger.info(f"Model training completed for {pair}")
            return accuracies
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def _save_models(self, pair: str):
        """Save trained models to disk."""
        try:
            model_dir = f"models/{pair.replace('/', '_')}"
            os.makedirs(model_dir, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_path = f"{model_dir}/{model_name}.joblib"
                scaler_path = f"{model_dir}/{model_name}_scaler.joblib"
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
            
            logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self, pair: str) -> bool:
        """Load trained models from disk."""
        try:
            model_dir = f"models/{pair.replace('/', '_')}"
            
            if not os.path.exists(model_dir):
                return False
            
            for model_name in self.models.keys():
                model_path = f"{model_dir}/{model_name}.joblib"
                scaler_path = f"{model_dir}/{model_name}_scaler.joblib"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                else:
                    return False
            
            logger.info(f"Models loaded from {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_signal(self, pair: str) -> Dict[str, Any]:
        """Generate trading signal using AI models."""
        try:
            # Check if models need retraining
            if (self.last_retrain is None or 
                datetime.now() - self.last_retrain > timedelta(hours=self.retrain_interval)):
                logger.info("Retraining models...")
                self.train_models(pair)
            
            # Load models if not already loaded
            if not self._load_models(pair):
                logger.info("No saved models found, training new models...")
                self.train_models(pair)
            
            # Get current market data
            df = self.analyzer.collect_market_data(pair, limit=self.feature_window)
            
            if df.empty:
                return {'signal': 'hold', 'confidence': 0.0, 'reason': 'No market data'}
            
            # Prepare features
            features_df = self.analyzer.prepare_features(df)
            
            if features_df.empty:
                return {'signal': 'hold', 'confidence': 0.0, 'reason': 'No features available'}
            
            # Get latest features
            latest_features = features_df.iloc[-1:].values
            
            predictions = {}
            confidences = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    features_scaled = self.scalers[model_name].transform(latest_features)
                    
                    # Predict
                    prediction = model.predict(features_scaled)[0]
                    confidence = model.predict_proba(features_scaled).max()
                    
                    predictions[model_name] = prediction
                    confidences[model_name] = confidence
                    
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    predictions[model_name] = 0
                    confidences[model_name] = 0.0
            
            # Ensemble prediction
            if self.model_type == 'ensemble':
                # Weighted average based on confidence
                total_weight = sum(confidences.values())
                if total_weight > 0:
                    weighted_prediction = sum(
                        predictions[model] * confidences[model] 
                        for model in predictions.keys()
                    ) / total_weight
                    
                    ensemble_confidence = np.mean(list(confidences.values()))
                else:
                    weighted_prediction = 0
                    ensemble_confidence = 0.0
            else:
                # Use single model
                model_name = self.model_type
                if model_name in predictions:
                    weighted_prediction = predictions[model_name]
                    ensemble_confidence = confidences[model_name]
                else:
                    weighted_prediction = 0
                    ensemble_confidence = 0.0
            
            # Convert to signal
            if weighted_prediction > 0.3:
                signal = 'buy'
            elif weighted_prediction < -0.3:
                signal = 'sell'
            else:
                signal = 'hold'
            
            # Get market context
            market_summary = self.analyzer.get_market_summary(pair)
            
            result = {
                'signal': signal,
                'confidence': ensemble_confidence,
                'prediction_score': weighted_prediction,
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'market_context': market_summary,
                'timestamp': datetime.now(),
                'pair': pair
            }
            
            logger.info(f"Generated signal for {pair}: {signal} (confidence: {ensemble_confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def get_trading_recommendation(self, pair: str) -> Dict[str, Any]:
        """Get comprehensive trading recommendation."""
        try:
            # Get AI signal
            ai_signal = self.predict_signal(pair)
            
            # Get market analysis
            market_summary = self.analyzer.get_market_summary(pair)
            
            # Combine AI signal with market analysis
            recommendation = {
                'pair': pair,
                'timestamp': datetime.now(),
                'ai_signal': ai_signal,
                'market_analysis': market_summary,
                'recommendation': self._combine_signals(ai_signal, market_summary)
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error getting trading recommendation: {e}")
            return {'recommendation': 'hold', 'reason': f'Error: {str(e)}'}
    
    def _combine_signals(self, ai_signal: Dict, market_analysis: Dict) -> Dict[str, Any]:
        """Combine AI signal with market analysis for final recommendation."""
        try:
            ai_signal_type = ai_signal.get('signal', 'hold')
            ai_confidence = ai_signal.get('confidence', 0.0)
            
            # Get market sentiment
            sentiment = market_analysis.get('sentiment', {})
            trend_sentiment = sentiment.get('trend_sentiment', 0)
            momentum_sentiment = sentiment.get('momentum_sentiment', 0)
            
            # Combine signals
            final_signal = ai_signal_type
            final_confidence = ai_confidence
            
            # Adjust based on market sentiment
            if trend_sentiment > 0.5 and ai_signal_type == 'buy':
                final_confidence += 0.1
            elif trend_sentiment < -0.5 and ai_signal_type == 'sell':
                final_confidence += 0.1
            elif trend_sentiment < -0.5 and ai_signal_type == 'buy':
                final_confidence -= 0.1
            elif trend_sentiment > 0.5 and ai_signal_type == 'sell':
                final_confidence -= 0.1
            
            # Adjust based on momentum
            if momentum_sentiment > 0.5 and ai_signal_type == 'buy':
                final_confidence += 0.05
            elif momentum_sentiment < -0.5 and ai_signal_type == 'sell':
                final_confidence += 0.05
            
            # Cap confidence
            final_confidence = min(max(final_confidence, 0.0), 1.0)
            
            # Determine position size based on confidence
            if final_confidence > 0.8:
                position_size = 'large'
            elif final_confidence > 0.6:
                position_size = 'medium'
            elif final_confidence > 0.4:
                position_size = 'small'
            else:
                position_size = 'minimal'
                final_signal = 'hold'
            
            return {
                'action': final_signal,
                'confidence': final_confidence,
                'position_size': position_size,
                'reasoning': {
                    'ai_signal': ai_signal_type,
                    'ai_confidence': ai_confidence,
                    'trend_sentiment': trend_sentiment,
                    'momentum_sentiment': momentum_sentiment
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}