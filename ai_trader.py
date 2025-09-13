"""AI trading strategy and decision engine."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from datetime import datetime, timedelta
from loguru import logger
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
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
        """Initialize comprehensive ML models."""
        self.models = {
            # Traditional ML models
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                C=0.1
            ),
            
            # Advanced gradient boosting
            'xgboost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                class_weight='balanced'
            ),
            
            # Ensemble model
            'ensemble': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
                    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)),
                    ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, class_weight='balanced'))
                ],
                voting='soft'
            )
        }
        
        # Initialize scalers
        for model_name in self.models.keys():
            if model_name != 'lstm':  # LSTM doesn't need scaler
                self.scalers[model_name] = StandardScaler()
        
        # Initialize LSTM model (will be created dynamically)
        self.lstm_model = None
        self.lstm_scaler = StandardScaler()
        
        # Initialize hyperparameter optimization
        self.hyperopt_results = {}
    
    def create_lstm_model(self, input_shape: Tuple[int, int], num_classes: int = 3) -> tf.keras.Model:
        """Create LSTM model for time series prediction."""
        try:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                BatchNormalization(),
                
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                BatchNormalization(),
                
                Dense(64, activation='relu'),
                Dropout(0.3),
                BatchNormalization(),
                
                Dense(32, activation='relu'),
                Dropout(0.2),
                
                Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None
    
    def prepare_lstm_data(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        try:
            # Scale features
            X_scaled = self.lstm_scaler.fit_transform(X)
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-sequence_length:i])
                y_sequences.append(y[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Convert labels to categorical
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_sequences)
            y_categorical = tf.keras.utils.to_categorical(y_encoded)
            
            return X_sequences, y_categorical
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {e}")
            return np.array([]), np.array([])
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        try:
            def objective(trial):
                if model_name == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                    }
                    model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)
                
                elif model_name == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                    }
                    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                
                elif model_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                    }
                    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                
                else:
                    return 0.0
                
                # Use time series split for validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict and calculate F1 score
                    y_pred = model.predict(X_val_scaled)
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    scores.append(f1)
                
                return np.mean(scores)
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            best_params = study.best_params
            best_score = study.best_value
            
            logger.info(f"Best parameters for {model_name}: {best_params}")
            logger.info(f"Best F1 score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study
            }
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters for {model_name}: {e}")
            return {}
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create sophisticated trading labels with dynamic thresholds."""
        if df.empty:
            return pd.Series()
            
        try:
            # Calculate future returns for multiple horizons
            future_returns_1 = df['close'].shift(-1) / df['close'] - 1
            future_returns_3 = df['close'].shift(-3) / df['close'] - 1
            future_returns_5 = df['close'].shift(-5) / df['close'] - 1
            future_returns_10 = df['close'].shift(-10) / df['close'] - 1
            
            # Calculate rolling volatility for dynamic thresholds
            volatility = df['close'].pct_change().rolling(window=20).std()
            volatility_median = volatility.median()
            
            # Dynamic thresholds based on volatility
            high_vol_threshold = 0.03 + (volatility / volatility_median) * 0.01
            low_vol_threshold = 0.01 + (volatility / volatility_median) * 0.005
            
            # Create multi-horizon labels
            labels = pd.Series(0, index=df.index)  # Default to hold
            
            # Short-term signals (1-3 periods)
            short_term_buy = (future_returns_1 > low_vol_threshold) | (future_returns_3 > high_vol_threshold)
            short_term_sell = (future_returns_1 < -low_vol_threshold) | (future_returns_3 < -high_vol_threshold)
            
            # Medium-term signals (5-10 periods)
            medium_term_buy = (future_returns_5 > high_vol_threshold) | (future_returns_10 > high_vol_threshold * 1.5)
            medium_term_sell = (future_returns_5 < -high_vol_threshold) | (future_returns_10 < -high_vol_threshold * 1.5)
            
            # Combine signals with different weights
            buy_signals = short_term_buy.astype(int) * 0.6 + medium_term_buy.astype(int) * 0.4
            sell_signals = short_term_sell.astype(int) * 0.6 + medium_term_sell.astype(int) * 0.4
            
            # Create final labels with confidence levels
            labels[buy_signals > 0.5] = 1  # Buy
            labels[sell_signals > 0.5] = -1  # Sell
            
            # Add confidence-based labels (stronger signals)
            strong_buy = (buy_signals > 0.8) & (future_returns_5 > high_vol_threshold * 1.2)
            strong_sell = (sell_signals > 0.8) & (future_returns_5 < -high_vol_threshold * 1.2)
            
            labels[strong_buy] = 2  # Strong buy
            labels[strong_sell] = -2  # Strong sell
            
            # Add trend-following labels
            sma_20 = df['close'].rolling(window=20).mean()
            sma_50 = df['close'].rolling(window=50).mean()
            trend_up = (sma_20 > sma_50) & (df['close'] > sma_20)
            trend_down = (sma_20 < sma_50) & (df['close'] < sma_20)
            
            # Adjust labels based on trend
            labels[(labels == 1) & trend_up] = 1.5  # Trend-confirmed buy
            labels[(labels == -1) & trend_down] = -1.5  # Trend-confirmed sell
            
            # Add momentum-based labels
            rsi = df['close'].rolling(window=14).apply(lambda x: self._calculate_rsi(x))
            momentum_buy = (rsi < 30) & (future_returns_3 > 0.01)  # Oversold bounce
            momentum_sell = (rsi > 70) & (future_returns_3 < -0.01)  # Overbought drop
            
            labels[momentum_buy] = 1.2  # Momentum buy
            labels[momentum_sell] = -1.2  # Momentum sell
            
            # Add volatility-based adjustments
            high_vol = volatility > volatility.quantile(0.8)
            low_vol = volatility < volatility.quantile(0.2)
            
            # Reduce signal strength in high volatility
            labels[(labels > 0) & high_vol] = labels[(labels > 0) & high_vol] * 0.8
            labels[(labels < 0) & high_vol] = labels[(labels < 0) & high_vol] * 0.8
            
            # Increase signal strength in low volatility with clear trends
            clear_trend = abs(sma_20 - sma_50) / sma_50 > 0.02
            labels[(labels > 0) & low_vol & clear_trend] = labels[(labels > 0) & low_vol & clear_trend] * 1.2
            labels[(labels < 0) & low_vol & clear_trend] = labels[(labels < 0) & low_vol & clear_trend] * 1.2
            
            logger.info(f"Created labels: {labels.value_counts().to_dict()}")
            return labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series()
    
    def _calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI for a price series."""
        try:
            if len(prices) < 14:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
            
        except Exception:
            return 50.0
    
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
        """Train comprehensive ML models on historical data."""
        try:
            # Collect more data for training
            df = self.analyzer.collect_market_data(pair, limit=self.feature_window * 3)
            
            if df.empty:
                logger.warning(f"No data available for training on {pair}")
                return {}
            
            # Prepare training data
            X, y = self.prepare_training_data(df)
            
            if X.empty or y.empty:
                logger.warning(f"Insufficient data for training on {pair}")
                return {}
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=3)
            accuracies = {}
            f1_scores = {}
            
            # Train traditional models
            for model_name, model in self.models.items():
                if model_name == 'ensemble':
                    continue  # Skip ensemble for now
                    
                try:
                    # Use time series cross-validation
                    cv_scores = []
                    cv_f1_scores = []
                    
                    for train_idx, val_idx in tscv.split(X):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Handle class imbalance
                        if model_name in ['xgboost', 'lightgbm']:
                            # These models handle class imbalance internally
                            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                            X_val_scaled = self.scalers[model_name].transform(X_val)
                        else:
                            # Use SMOTE for other models
                            smote = SMOTE(random_state=42)
                            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                            X_val_scaled = self.scalers[model_name].transform(X_val)
                            
                            try:
                                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                            except:
                                X_train_balanced, y_train_balanced = X_train_scaled, y_train
                        
                        # Train model
                        if model_name in ['xgboost', 'lightgbm']:
                            model.fit(X_train_scaled, y_train)
                        else:
                            model.fit(X_train_balanced, y_train_balanced)
                        
                        # Evaluate
                        y_pred = model.predict(X_val_scaled)
                        accuracy = accuracy_score(y_val, y_pred)
                        f1 = f1_score(y_val, y_pred, average='weighted')
                        
                        cv_scores.append(accuracy)
                        cv_f1_scores.append(f1)
                    
                    accuracies[model_name] = np.mean(cv_scores)
                    f1_scores[model_name] = np.mean(cv_f1_scores)
                    
                    logger.info(f"Trained {model_name} - Accuracy: {accuracies[model_name]:.3f}, F1: {f1_scores[model_name]:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    accuracies[model_name] = 0.0
                    f1_scores[model_name] = 0.0
            
            # Train LSTM model
            try:
                X_lstm, y_lstm = self.prepare_lstm_data(X.values, y.values)
                
                if len(X_lstm) > 0:
                    # Split for LSTM
                    split_idx = int(len(X_lstm) * 0.8)
                    X_train_lstm, X_val_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
                    y_train_lstm, y_val_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
                    
                    # Create LSTM model
                    self.lstm_model = self.create_lstm_model(
                        input_shape=(X_lstm.shape[1], X_lstm.shape[2]),
                        num_classes=y_lstm.shape[1]
                    )
                    
                    if self.lstm_model:
                        # Train LSTM
                        callbacks = [
                            EarlyStopping(patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(factor=0.5, patience=5)
                        ]
                        
                        history = self.lstm_model.fit(
                            X_train_lstm, y_train_lstm,
                            validation_data=(X_val_lstm, y_val_lstm),
                            epochs=100,
                            batch_size=32,
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        # Evaluate LSTM
                        lstm_pred = self.lstm_model.predict(X_val_lstm)
                        lstm_pred_classes = np.argmax(lstm_pred, axis=1)
                        y_val_classes = np.argmax(y_val_lstm, axis=1)
                        
                        lstm_accuracy = accuracy_score(y_val_classes, lstm_pred_classes)
                        lstm_f1 = f1_score(y_val_classes, lstm_pred_classes, average='weighted')
                        
                        accuracies['lstm'] = lstm_accuracy
                        f1_scores['lstm'] = lstm_f1
                        
                        logger.info(f"Trained LSTM - Accuracy: {lstm_accuracy:.3f}, F1: {lstm_f1:.3f}")
                
            except Exception as e:
                logger.error(f"Error training LSTM: {e}")
                accuracies['lstm'] = 0.0
                f1_scores['lstm'] = 0.0
            
            # Train ensemble model
            try:
                # Get best performing models for ensemble
                best_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[:3]
                
                if len(best_models) >= 2:
                    ensemble_estimators = []
                    for model_name, _ in best_models:
                        if model_name in self.models and model_name != 'ensemble':
                            ensemble_estimators.append((model_name, self.models[model_name]))
                    
                    if ensemble_estimators:
                        self.models['ensemble'] = VotingClassifier(
                            estimators=ensemble_estimators,
                            voting='soft'
                        )
                        
                        # Train ensemble
                        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        
                        X_train_scaled = self.scalers['random_forest'].fit_transform(X_train_final)
                        X_test_scaled = self.scalers['random_forest'].transform(X_test_final)
                        
                        self.models['ensemble'].fit(X_train_scaled, y_train_final)
                        
                        y_pred_ensemble = self.models['ensemble'].predict(X_test_scaled)
                        ensemble_accuracy = accuracy_score(y_test_final, y_pred_ensemble)
                        ensemble_f1 = f1_score(y_test_final, y_pred_ensemble, average='weighted')
                        
                        accuracies['ensemble'] = ensemble_accuracy
                        f1_scores['ensemble'] = ensemble_f1
                        
                        logger.info(f"Trained ensemble - Accuracy: {ensemble_accuracy:.3f}, F1: {ensemble_f1:.3f}")
                
            except Exception as e:
                logger.error(f"Error training ensemble: {e}")
                accuracies['ensemble'] = 0.0
                f1_scores['ensemble'] = 0.0
            
            # Save models
            self._save_models(pair)
            
            self.last_retrain = datetime.now()
            logger.info(f"Model training completed for {pair}")
            logger.info(f"Final accuracies: {accuracies}")
            logger.info(f"Final F1 scores: {f1_scores}")
            
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