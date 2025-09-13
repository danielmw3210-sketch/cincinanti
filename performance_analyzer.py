"""Comprehensive performance analysis and monitoring module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import json

class PerformanceAnalyzer:
    """Comprehensive performance analysis and monitoring."""
    
    def __init__(self):
        self.trade_history = []
        self.portfolio_history = []
        self.performance_metrics = {}
        
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add a trade to the history."""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'pair': trade_data.get('pair', ''),
                'signal': trade_data.get('signal', ''),
                'size': trade_data.get('size', 0),
                'entry_price': trade_data.get('entry_price', 0),
                'exit_price': trade_data.get('exit_price', 0),
                'pnl': trade_data.get('pnl', 0),
                'duration': trade_data.get('duration', timedelta(0)),
                'confidence': trade_data.get('confidence', 0),
                'stop_loss': trade_data.get('stop_loss', 0),
                'take_profit': trade_data.get('take_profit', 0),
                'exit_reason': trade_data.get('exit_reason', ''),
                'success': trade_data.get('success', False)
            }
            
            self.trade_history.append(trade_record)
            logger.debug(f"Added trade record: {trade_record}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
    
    def add_portfolio_snapshot(self, portfolio_data: Dict[str, Any]):
        """Add a portfolio snapshot to the history."""
        try:
            snapshot = {
                'timestamp': datetime.now(),
                'total_balance': portfolio_data.get('total_balance', 0),
                'available_balance': portfolio_data.get('available_balance', 0),
                'unrealized_pnl': portfolio_data.get('unrealized_pnl', 0),
                'open_positions': len(portfolio_data.get('positions', {})),
                'daily_pnl': portfolio_data.get('daily_pnl', 0),
                'max_drawdown': portfolio_data.get('max_drawdown', 0),
                'sharpe_ratio': portfolio_data.get('sharpe_ratio', 0)
            }
            
            self.portfolio_history.append(snapshot)
            logger.debug(f"Added portfolio snapshot: {snapshot}")
            
        except Exception as e:
            logger.error(f"Error adding portfolio snapshot: {e}")
    
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if not self.trade_history:
                return {'error': 'No trade history available'}
            
            df_trades = pd.DataFrame(self.trade_history)
            df_portfolio = pd.DataFrame(self.portfolio_history)
            
            metrics = {}
            
            # Basic trade metrics
            metrics.update(self._calculate_trade_metrics(df_trades))
            
            # Portfolio metrics
            metrics.update(self._calculate_portfolio_metrics(df_portfolio))
            
            # Risk metrics
            metrics.update(self._calculate_risk_metrics(df_trades, df_portfolio))
            
            # Model performance metrics
            metrics.update(self._calculate_model_metrics(df_trades))
            
            # Market regime analysis
            metrics.update(self._calculate_regime_metrics(df_trades))
            
            # Time-based analysis
            metrics.update(self._calculate_time_metrics(df_trades))
            
            self.performance_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_trade_metrics(self, df_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-level performance metrics."""
        try:
            if df_trades.empty:
                return {}
            
            # Basic statistics
            total_trades = len(df_trades)
            winning_trades = len(df_trades[df_trades['pnl'] > 0])
            losing_trades = len(df_trades[df_trades['pnl'] < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L statistics
            total_pnl = df_trades['pnl'].sum()
            avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Profit factor
            gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk-reward ratio
            risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Consecutive wins/losses
            df_trades['win'] = df_trades['pnl'] > 0
            df_trades['consecutive'] = (df_trades['win'] != df_trades['win'].shift()).cumsum()
            
            consecutive_wins = df_trades.groupby('consecutive')['win'].sum().max()
            consecutive_losses = df_trades.groupby('consecutive')['win'].apply(lambda x: (x == False).sum()).max()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'risk_reward_ratio': risk_reward_ratio,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {}
    
    def _calculate_portfolio_metrics(self, df_portfolio: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio-level performance metrics."""
        try:
            if df_portfolio.empty:
                return {}
            
            # Portfolio value progression
            initial_balance = df_portfolio['total_balance'].iloc[0]
            final_balance = df_portfolio['total_balance'].iloc[-1]
            total_return = (final_balance - initial_balance) / initial_balance
            
            # Calculate daily returns
            df_portfolio['daily_return'] = df_portfolio['total_balance'].pct_change()
            daily_returns = df_portfolio['daily_return'].dropna()
            
            # Sharpe ratio
            if len(daily_returns) > 1:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1:
                sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = 0
            
            # Value at Risk (VaR)
            var_95 = np.percentile(daily_returns, 5)
            var_99 = np.percentile(daily_returns, 1)
            
            # Expected Shortfall (CVaR)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            cvar_99 = daily_returns[daily_returns <= var_99].mean()
            
            return {
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annual_return': annual_return,
                'annual_return_pct': annual_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'volatility': daily_returns.std() * np.sqrt(252),
                'skewness': daily_returns.skew(),
                'kurtosis': daily_returns.kurtosis()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, df_trades: pd.DataFrame, df_portfolio: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-related metrics."""
        try:
            if df_trades.empty or df_portfolio.empty:
                return {}
            
            # Position sizing analysis
            position_sizes = df_trades['size'].abs()
            avg_position_size = position_sizes.mean()
            max_position_size = position_sizes.max()
            position_size_std = position_sizes.std()
            
            # Risk per trade analysis
            df_trades['risk_per_trade'] = abs(df_trades['pnl']) / df_trades['size'].abs()
            avg_risk_per_trade = df_trades['risk_per_trade'].mean()
            
            # Drawdown analysis
            df_portfolio['cumulative_return'] = (1 + df_portfolio['daily_return']).cumprod()
            running_max = df_portfolio['cumulative_return'].expanding().max()
            df_portfolio['drawdown'] = (df_portfolio['cumulative_return'] - running_max) / running_max
            
            # Drawdown duration
            drawdown_periods = []
            in_drawdown = False
            start_drawdown = None
            
            for i, row in df_portfolio.iterrows():
                if row['drawdown'] < 0 and not in_drawdown:
                    in_drawdown = True
                    start_drawdown = i
                elif row['drawdown'] >= 0 and in_drawdown:
                    in_drawdown = False
                    if start_drawdown is not None:
                        drawdown_periods.append(i - start_drawdown)
            
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            
            # Recovery factor
            max_dd = df_portfolio['drawdown'].min()
            recovery_factor = abs(df_portfolio['cumulative_return'].iloc[-1] - 1) / abs(max_dd) if max_dd != 0 else 0
            
            return {
                'avg_position_size': avg_position_size,
                'max_position_size': max_position_size,
                'position_size_std': position_size_std,
                'avg_risk_per_trade': avg_risk_per_trade,
                'avg_drawdown_duration': avg_drawdown_duration,
                'max_drawdown_duration': max_drawdown_duration,
                'recovery_factor': recovery_factor,
                'drawdown_periods_count': len(drawdown_periods)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_model_metrics(self, df_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        try:
            if df_trades.empty:
                return {}
            
            # Signal accuracy
            correct_signals = 0
            total_signals = len(df_trades)
            
            for _, trade in df_trades.iterrows():
                if trade['signal'] == 'buy' and trade['pnl'] > 0:
                    correct_signals += 1
                elif trade['signal'] == 'sell' and trade['pnl'] > 0:
                    correct_signals += 1
            
            signal_accuracy = correct_signals / total_signals if total_signals > 0 else 0
            
            # Confidence analysis
            if 'confidence' in df_trades.columns:
                high_confidence_trades = df_trades[df_trades['confidence'] > 0.7]
                high_conf_accuracy = len(high_confidence_trades[high_confidence_trades['pnl'] > 0]) / len(high_confidence_trades) if len(high_confidence_trades) > 0 else 0
                
                confidence_correlation = df_trades['confidence'].corr(df_trades['pnl'])
            else:
                high_conf_accuracy = 0
                confidence_correlation = 0
            
            # Pair performance
            pair_performance = df_trades.groupby('pair').agg({
                'pnl': ['sum', 'count', 'mean'],
                'confidence': 'mean'
            }).round(4)
            
            return {
                'signal_accuracy': signal_accuracy,
                'signal_accuracy_pct': signal_accuracy * 100,
                'high_confidence_accuracy': high_conf_accuracy,
                'high_confidence_accuracy_pct': high_conf_accuracy * 100,
                'confidence_correlation': confidence_correlation,
                'pair_performance': pair_performance.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error calculating model metrics: {e}")
            return {}
    
    def _calculate_regime_metrics(self, df_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market regime analysis."""
        try:
            if df_trades.empty:
                return {}
            
            # Add time-based features
            df_trades['hour'] = pd.to_datetime(df_trades['timestamp']).dt.hour
            df_trades['day_of_week'] = pd.to_datetime(df_trades['timestamp']).dt.dayofweek
            df_trades['month'] = pd.to_datetime(df_trades['timestamp']).dt.month
            
            # Session analysis
            df_trades['session'] = 'other'
            df_trades.loc[df_trades['hour'].between(0, 8), 'session'] = 'asian'
            df_trades.loc[df_trades['hour'].between(8, 16), 'session'] = 'european'
            df_trades.loc[df_trades['hour'].between(16, 24), 'session'] = 'us'
            
            session_performance = df_trades.groupby('session')['pnl'].agg(['sum', 'count', 'mean']).round(4)
            
            # Day of week analysis
            dow_performance = df_trades.groupby('day_of_week')['pnl'].agg(['sum', 'count', 'mean']).round(4)
            
            # Monthly analysis
            monthly_performance = df_trades.groupby('month')['pnl'].agg(['sum', 'count', 'mean']).round(4)
            
            return {
                'session_performance': session_performance.to_dict(),
                'day_of_week_performance': dow_performance.to_dict(),
                'monthly_performance': monthly_performance.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime metrics: {e}")
            return {}
    
    def _calculate_time_metrics(self, df_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-based performance metrics."""
        try:
            if df_trades.empty:
                return {}
            
            # Trade duration analysis
            if 'duration' in df_trades.columns:
                durations = pd.to_timedelta(df_trades['duration']).dt.total_seconds() / 3600  # Convert to hours
                
                avg_duration = durations.mean()
                median_duration = durations.median()
                max_duration = durations.max()
                min_duration = durations.min()
            else:
                avg_duration = median_duration = max_duration = min_duration = 0
            
            # Trading frequency
            if len(df_trades) > 1:
                time_span = (pd.to_datetime(df_trades['timestamp'].iloc[-1]) - 
                           pd.to_datetime(df_trades['timestamp'].iloc[0])).total_seconds() / 3600
                trades_per_hour = len(df_trades) / time_span if time_span > 0 else 0
            else:
                trades_per_hour = 0
            
            return {
                'avg_trade_duration_hours': avg_duration,
                'median_trade_duration_hours': median_duration,
                'max_trade_duration_hours': max_duration,
                'min_trade_duration_hours': min_duration,
                'trades_per_hour': trades_per_hour
            }
            
        except Exception as e:
            logger.error(f"Error calculating time metrics: {e}")
            return {}
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        try:
            metrics = self.calculate_comprehensive_metrics()
            
            if 'error' in metrics:
                return f"Error generating report: {metrics['error']}"
            
            report = []
            report.append("=" * 80)
            report.append("AI TRADING SYSTEM PERFORMANCE REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Trade Performance
            report.append("TRADE PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
            report.append(f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
            report.append(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
            report.append(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
            report.append(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
            report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report.append(f"Risk-Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}")
            report.append("")
            
            # Portfolio Performance
            report.append("PORTFOLIO PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
            report.append(f"Annual Return: {metrics.get('annual_return_pct', 0):.2f}%")
            report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            report.append(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            report.append("")
            
            # Risk Metrics
            report.append("RISK METRICS")
            report.append("-" * 40)
            report.append(f"VaR (95%): {metrics.get('var_95', 0):.4f}")
            report.append(f"VaR (99%): {metrics.get('var_99', 0):.4f}")
            report.append(f"CVaR (95%): {metrics.get('cvar_95', 0):.4f}")
            report.append(f"CVaR (99%): {metrics.get('cvar_99', 0):.4f}")
            report.append(f"Volatility: {metrics.get('volatility', 0):.4f}")
            report.append("")
            
            # Model Performance
            report.append("MODEL PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Signal Accuracy: {metrics.get('signal_accuracy_pct', 0):.2f}%")
            report.append(f"High Confidence Accuracy: {metrics.get('high_confidence_accuracy_pct', 0):.2f}%")
            report.append(f"Confidence Correlation: {metrics.get('confidence_correlation', 0):.4f}")
            report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {str(e)}"
    
    def save_metrics_to_file(self, filename: str = None):
        """Save performance metrics to a JSON file."""
        try:
            if filename is None:
                filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            metrics = self.calculate_comprehensive_metrics()
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return obj
            
            # Recursively convert all numpy types
            def recursive_convert(d):
                if isinstance(d, dict):
                    return {k: recursive_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [recursive_convert(item) for item in d]
                else:
                    return convert_numpy(d)
            
            metrics_serializable = recursive_convert(metrics)
            
            with open(filename, 'w') as f:
                json.dump(metrics_serializable, f, indent=2, default=str)
            
            logger.info(f"Performance metrics saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")