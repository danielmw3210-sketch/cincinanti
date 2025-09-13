"""Hyperparameter optimization module using Optuna."""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import joblib
import os

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, ai_trader, market_analyzer):
        self.ai_trader = ai_trader
        self.market_analyzer = market_analyzer
        self.optimization_results = {}
        
    def optimize_all_models(self, pair: str, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters for all models."""
        try:
            logger.info(f"Starting hyperparameter optimization for {pair}")
            
            # Get training data
            df = self.market_analyzer.collect_market_data(pair, limit=2000)
            if df.empty:
                logger.error("No data available for optimization")
                return {}
            
            X, y = self.ai_trader.prepare_training_data(df)
            if X.empty or y.empty:
                logger.error("Insufficient data for optimization")
                return {}
            
            # Optimize each model
            models_to_optimize = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
            
            for model_name in models_to_optimize:
                logger.info(f"Optimizing {model_name}...")
                result = self._optimize_single_model(model_name, X, y, n_trials)
                self.optimization_results[model_name] = result
                
                # Update the model with best parameters
                if result and 'best_params' in result:
                    self._update_model_with_best_params(model_name, result['best_params'])
            
            # Save optimization results
            self._save_optimization_results(pair)
            
            logger.info("Hyperparameter optimization completed")
            return self.optimization_results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {}
    
    def _optimize_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, n_trials: int) -> Dict[str, Any]:
        """Optimize hyperparameters for a single model."""
        try:
            def objective(trial):
                # Get parameter suggestions based on model type
                params = self._get_parameter_suggestions(trial, model_name)
                
                # Create model with suggested parameters
                model = self._create_model_with_params(model_name, params)
                
                # Use time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    scaler = self.ai_trader.scalers.get(model_name, self.ai_trader.scalers['random_forest'])
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict and calculate score
                    y_pred = model.predict(X_val_scaled)
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    scores.append(f1)
                
                return np.mean(scores)
            
            # Create study and optimize
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'study': study
            }
            
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}")
            return {}
    
    def _get_parameter_suggestions(self, trial, model_name: str) -> Dict[str, Any]:
        """Get parameter suggestions for a specific model."""
        if model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300)
            }
        
        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.suggest_categorical('use_max_samples', [True, False]) else None
            }
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        
        else:
            return {}
    
    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Create a model instance with given parameters."""
        try:
            if model_name == 'xgboost':
                import xgboost as xgb
                return xgb.XGBClassifier(
                    **params,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
            
            elif model_name == 'lightgbm':
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    **params,
                    random_state=42,
                    verbose=-1,
                    class_weight='balanced'
                )
            
            elif model_name == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            
            elif model_name == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    **params,
                    random_state=42
                )
            
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            return None
    
    def _update_model_with_best_params(self, model_name: str, best_params: Dict[str, Any]):
        """Update the AI trader's model with best parameters."""
        try:
            if model_name in self.ai_trader.models:
                # Create new model with best parameters
                new_model = self._create_model_with_params(model_name, best_params)
                if new_model:
                    self.ai_trader.models[model_name] = new_model
                    logger.info(f"Updated {model_name} with optimized parameters")
            
        except Exception as e:
            logger.error(f"Error updating model {model_name}: {e}")
    
    def _save_optimization_results(self, pair: str):
        """Save optimization results to disk."""
        try:
            results_dir = f"optimization_results/{pair.replace('/', '_')}"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{results_dir}/optimization_{timestamp}.json"
            
            # Convert results to serializable format
            serializable_results = {}
            for model_name, result in self.optimization_results.items():
                if result and 'study' in result:
                    # Remove study object as it's not serializable
                    serializable_result = {k: v for k, v in result.items() if k != 'study'}
                    serializable_results[model_name] = serializable_result
                else:
                    serializable_results[model_name] = result
            
            import json
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
    
    def get_optimization_summary(self) -> str:
        """Get a summary of optimization results."""
        try:
            if not self.optimization_results:
                return "No optimization results available"
            
            summary = []
            summary.append("=" * 60)
            summary.append("HYPERPARAMETER OPTIMIZATION SUMMARY")
            summary.append("=" * 60)
            
            for model_name, result in self.optimization_results.items():
                if result and 'best_score' in result:
                    summary.append(f"\n{model_name.upper()}:")
                    summary.append(f"  Best Score: {result['best_score']:.4f}")
                    summary.append(f"  Trials: {result.get('n_trials', 'N/A')}")
                    summary.append(f"  Best Params: {result.get('best_params', {})}")
                else:
                    summary.append(f"\n{model_name.upper()}: Optimization failed")
            
            return "\n".join(summary)
            
        except Exception as e:
            logger.error(f"Error generating optimization summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def optimize_feature_selection(self, pair: str, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize feature selection using Optuna."""
        try:
            logger.info(f"Starting feature selection optimization for {pair}")
            
            # Get training data
            df = self.market_analyzer.collect_market_data(pair, limit=2000)
            if df.empty:
                logger.error("No data available for feature selection")
                return {}
            
            X, y = self.ai_trader.prepare_training_data(df)
            if X.empty or y.empty:
                logger.error("Insufficient data for feature selection")
                return {}
            
            def objective(trial):
                # Select features
                feature_mask = []
                for col in X.columns:
                    feature_mask.append(trial.suggest_categorical(f"feature_{col}", [True, False]))
                
                if not any(feature_mask):
                    return 0.0  # At least one feature must be selected
                
                X_selected = X.iloc[:, feature_mask]
                
                # Use a simple model for feature selection
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_selected):
                    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    scores.append(f1)
                
                return np.mean(scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Extract selected features
            selected_features = []
            for i, col in enumerate(X.columns):
                if study.best_params.get(f"feature_{col}", False):
                    selected_features.append(col)
            
            result = {
                'selected_features': selected_features,
                'best_score': study.best_value,
                'n_trials': len(study.trials)
            }
            
            logger.info(f"Feature selection completed. Selected {len(selected_features)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error in feature selection optimization: {e}")
            return {}