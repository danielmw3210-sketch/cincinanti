"""Backtesting script for the AI Crypto Trader."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json
from loguru import logger

from kraken_client import KrakenClient
from market_analyzer import MarketAnalyzer
from ai_trader import AITrader
from risk_manager import RiskManager
from config import config

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self):
        self.client = KrakenClient()
        self.analyzer = MarketAnalyzer(self.client)
        self.ai_trader = AITrader(self.analyzer)
        self.risk_manager = RiskManager()
        
        # Backtest results
        self.trades = []
        self.portfolio_history = []
        self.initial_balance = config.initial_balance
        self.current_balance = self.initial_balance
        self.positions = {}
        
    def run_backtest(self, 
                    pair: str, 
                    start_date: str, 
                    end_date: str,
                    initial_balance: float = None) -> Dict[str, any]:
        """Run backtest for specified period."""
        try:
            logger.info(f"Starting backtest for {pair} from {start_date} to {end_date}")
            
            if initial_balance:
                self.initial_balance = initial_balance
                self.current_balance = initial_balance
            
            # Get historical data
            df = self._get_historical_data(pair, start_date, end_date)
            
            if df.empty:
                logger.error("No historical data available")
                return {'error': 'No historical data'}
            
            # Train models on first portion of data
            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]
            
            logger.info(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
            
            # Train AI models
            self.ai_trader.train_models(pair)
            
            # Run backtest
            self._run_backtest_loop(test_data, pair)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics()
            
            logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def _get_historical_data(self, pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data for backtesting."""
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # Get OHLC data
            ohlc_data = self.client.get_ohlc_data(pair, interval=60, since=start_ts)  # 1-hour intervals
            
            if not ohlc_data:
                logger.warning("No OHLC data received, creating sample data")
                return self._create_sample_data(start_date, end_date)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"Retrieved {len(df)} data points for backtesting")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return self._create_sample_data(start_date, end_date)
    
    def _create_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create sample data for testing."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate hourly data
            dates = pd.date_range(start, end, freq='1H')
            
            # Generate realistic price data
            np.random.seed(42)
            base_price = 50000
            returns = np.random.normal(0, 0.02, len(dates))  # 2% volatility
            prices = base_price * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'open': prices,
                'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
                'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            logger.info(f"Created {len(df)} sample data points")
            return df
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            return pd.DataFrame()
    
    def _run_backtest_loop(self, df: pd.DataFrame, pair: str):
        """Run the main backtesting loop."""
        try:
            for i in range(100, len(df)):  # Start after enough data for indicators
                current_data = df.iloc[:i+1]
                current_price = df.iloc[i]['close']
                current_time = df.index[i]
                
                # Get AI signal
                signal_data = self._get_ai_signal(current_data, pair)
                
                if signal_data and signal_data.get('signal') in ['buy', 'sell']:
                    signal = signal_data['signal']
                    confidence = signal_data.get('confidence', 0.5)
                    
                    # Calculate position size
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        current_price, signal, 0.02
                    )
                    
                    position_calc = self.risk_manager.calculate_position_size(
                        confidence, current_price, stop_loss, 
                        self.current_balance, pair
                    )
                    
                    position_size = position_calc['size']
                    
                    if position_size > 0:
                        # Execute trade
                        trade_result = self._execute_backtest_trade(
                            pair, signal, position_size, current_price, current_time
                        )
                        
                        if trade_result['success']:
                            self.trades.append(trade_result)
                
                # Update portfolio history
                self._update_portfolio_history(current_time, current_price)
                
        except Exception as e:
            logger.error(f"Error in backtest loop: {e}")
    
    def _get_ai_signal(self, df: pd.DataFrame, pair: str) -> Dict[str, any]:
        """Get AI signal for given data."""
        try:
            # Prepare features
            features_df = self.analyzer.prepare_features(df)
            
            if features_df.empty or len(features_df) < 10:
                return None
            
            # Get latest features
            latest_features = features_df.iloc[-1:].values
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model in self.ai_trader.models.items():
                try:
                    # Scale features
                    features_scaled = self.ai_trader.scalers[model_name].transform(latest_features)
                    
                    # Predict
                    prediction = model.predict(features_scaled)[0]
                    confidence = model.predict_proba(features_scaled).max()
                    
                    predictions[model_name] = prediction
                    confidences[model_name] = confidence
                    
                except Exception as e:
                    logger.debug(f"Error getting prediction from {model_name}: {e}")
                    predictions[model_name] = 0
                    confidences[model_name] = 0.0
            
            # Ensemble prediction
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
            
            # Convert to signal
            if weighted_prediction > 0.3:
                signal = 'buy'
            elif weighted_prediction < -0.3:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'signal': signal,
                'confidence': ensemble_confidence,
                'prediction_score': weighted_prediction
            }
            
        except Exception as e:
            logger.debug(f"Error getting AI signal: {e}")
            return None
    
    def _execute_backtest_trade(self, 
                               pair: str, 
                               signal: str, 
                               size: float, 
                               price: float, 
                               timestamp: datetime) -> Dict[str, any]:
        """Execute a trade in backtest."""
        try:
            trade_value = size * price
            
            # Check if we have enough balance
            if signal == 'buy' and trade_value > self.current_balance:
                return {'success': False, 'reason': 'Insufficient balance'}
            
            # Execute trade
            if signal == 'buy':
                self.current_balance -= trade_value
                if pair not in self.positions:
                    self.positions[pair] = {'size': 0, 'avg_price': 0}
                
                # Update position
                current_pos = self.positions[pair]
                new_size = current_pos['size'] + size
                new_value = current_pos['size'] * current_pos['avg_price'] + trade_value
                new_avg_price = new_value / new_size if new_size > 0 else price
                
                self.positions[pair] = {
                    'size': new_size,
                    'avg_price': new_avg_price
                }
                
            elif signal == 'sell':
                if pair not in self.positions or self.positions[pair]['size'] < size:
                    return {'success': False, 'reason': 'Insufficient position'}
                
                # Calculate P&L
                avg_price = self.positions[pair]['avg_price']
                pnl = (price - avg_price) * size
                
                self.current_balance += trade_value
                self.positions[pair]['size'] -= size
                
                if self.positions[pair]['size'] <= 0:
                    del self.positions[pair]
            
            return {
                'success': True,
                'pair': pair,
                'signal': signal,
                'size': size,
                'price': price,
                'timestamp': timestamp,
                'balance': self.current_balance
            }
            
        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _update_portfolio_history(self, timestamp: datetime, current_price: float):
        """Update portfolio history."""
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_balance
            
            for pair, position in self.positions.items():
                portfolio_value += position['size'] * current_price
            
            self.portfolio_history.append({
                'timestamp': timestamp,
                'balance': self.current_balance,
                'portfolio_value': portfolio_value,
                'positions': dict(self.positions)
            })
            
        except Exception as e:
            logger.error(f"Error updating portfolio history: {e}")
    
    def _calculate_performance_metrics(self) -> Dict[str, any]:
        """Calculate backtest performance metrics."""
        try:
            if not self.portfolio_history:
                return {'error': 'No portfolio history'}
            
            # Calculate returns
            initial_value = self.portfolio_history[0]['portfolio_value']
            final_value = self.portfolio_history[-1]['portfolio_value']
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate daily returns
            portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Calculate metrics
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Calculate win rate
            winning_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            
            # Calculate profit factor
            gross_profit = sum(self._calculate_trade_pnl(t) for t in winning_trades)
            gross_loss = sum(abs(self._calculate_trade_pnl(t)) for t in self.trades if self._calculate_trade_pnl(t) < 0)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'profit_factor': profit_factor,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(self.trades) - len(winning_trades),
                'initial_balance': self.initial_balance,
                'final_balance': final_value,
                'portfolio_history': self.portfolio_history,
                'trades': self.trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_trade_pnl(self, trade: Dict[str, any]) -> float:
        """Calculate P&L for a trade."""
        try:
            # This is simplified - in reality you'd need to track entry/exit prices
            # For now, return 0 as placeholder
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trade P&L: {e}")
            return 0.0

def main():
    """Main backtesting function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest AI Crypto Trader")
    parser.add_argument("--pair", "-p", default="BTC/USD", help="Trading pair")
    parser.add_argument("--start", "-s", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", "-b", type=float, default=10000, help="Initial balance")
    parser.add_argument("--output", "-o", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(
        pair=args.pair,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance
    )
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Backtest results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()