"""Trading execution and portfolio management module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
from kraken_client import KrakenClient
from risk_manager import RiskManager
from ai_trader import AITrader
from market_analyzer import MarketAnalyzer
from config import config

class TradingExecutor:
    """Handles trade execution and portfolio management."""
    
    def __init__(self):
        self.client = KrakenClient()
        self.risk_manager = RiskManager()
        self.market_analyzer = MarketAnalyzer(self.client)
        self.ai_trader = AITrader(self.market_analyzer)
        
        # Portfolio tracking
        self.portfolio = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        # Trading state
        self.is_trading_active = True
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=5)  # Minimum time between trades
        
    def initialize_portfolio(self):
        """Initialize portfolio with current account balance."""
        try:
            balance = self.client.get_account_balance()
            
            if not balance:
                logger.error("Failed to get account balance")
                return False
            
            # Initialize portfolio
            self.portfolio = {
                'total_balance': sum(float(v) for v in balance.values()),
                'available_balance': sum(float(v) for v in balance.values()),
                'positions': {},
                'cash': balance.get('ZUSD', 0.0) or balance.get('USD', 0.0),
                'last_updated': datetime.now()
            }
            
            logger.info(f"Portfolio initialized with balance: {self.portfolio['total_balance']:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
            return False
    
    def update_portfolio(self):
        """Update portfolio with current account state."""
        try:
            balance = self.client.get_account_balance()
            
            if not balance:
                logger.warning("Failed to get account balance")
                return
            
            # Update portfolio
            self.portfolio['total_balance'] = sum(float(v) for v in balance.values())
            self.portfolio['cash'] = balance.get('ZUSD', 0.0) or balance.get('USD', 0.0)
            self.portfolio['last_updated'] = datetime.now()
            
            # Update available balance (total - locked in positions)
            locked_amount = sum(pos.get('value', 0) for pos in self.portfolio['positions'].values())
            self.portfolio['available_balance'] = self.portfolio['total_balance'] - locked_amount
            
            logger.debug(f"Portfolio updated: Total={self.portfolio['total_balance']:.2f}, Available={self.portfolio['available_balance']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    def execute_trade(self, 
                     pair: str, 
                     signal: str, 
                     confidence: float,
                     position_size: Optional[float] = None) -> Dict[str, Any]:
        """Execute a trade based on AI signal."""
        try:
            # Check if trading is active
            if not self.is_trading_active:
                return {'success': False, 'reason': 'Trading is disabled'}
            
            # Check emergency stop
            if self.risk_manager.emergency_stop():
                logger.critical("Emergency stop activated, halting trading")
                self.is_trading_active = False
                return {'success': False, 'reason': 'Emergency stop activated'}
            
            # Check minimum trade interval
            if (self.last_trade_time and 
                datetime.now() - self.last_trade_time < self.min_trade_interval):
                return {'success': False, 'reason': 'Minimum trade interval not met'}
            
            # Update portfolio
            self.update_portfolio()
            
            # Get current market data
            ticker = self.client.get_ticker(pair)
            if not ticker:
                return {'success': False, 'reason': 'Failed to get market data'}
            
            # Extract current price
            ticker_data = list(ticker.values())[0] if ticker else {}
            current_price = float(ticker_data.get('c', [0])[0]) if ticker_data.get('c') else 0
            
            if current_price == 0:
                return {'success': False, 'reason': 'Invalid current price'}
            
            # Get market analysis for risk calculation
            market_summary = self.market_analyzer.get_market_summary(pair)
            volatility = market_summary.get('sentiment', {}).get('volatility_sentiment', 0.02)
            
            # Calculate position size if not provided
            if position_size is None:
                stop_loss = self.risk_manager.calculate_stop_loss(
                    current_price, signal, abs(volatility)
                )
                
                position_calc = self.risk_manager.calculate_position_size(
                    confidence, current_price, stop_loss, 
                    self.portfolio['available_balance'], pair
                )
                
                position_size = position_calc['size']
            
            # Validate trade
            validation = self.risk_manager.validate_trade(
                pair, signal, position_size, current_price, 
                self.portfolio['available_balance']
            )
            
            if not validation['valid']:
                return {
                    'success': False, 
                    'reason': f"Trade validation failed: {', '.join(validation['errors'])}"
                }
            
            # Execute the trade
            if signal == 'buy':
                result = self._execute_buy_order(pair, position_size, current_price)
            elif signal == 'sell':
                result = self._execute_sell_order(pair, position_size, current_price)
            else:
                return {'success': False, 'reason': 'Invalid signal type'}
            
            if result['success']:
                # Update risk manager
                stop_loss = self.risk_manager.calculate_stop_loss(
                    current_price, signal, abs(volatility)
                )
                take_profit = self.risk_manager.calculate_take_profit(
                    current_price, signal, stop_loss_price=stop_loss
                )
                
                self.risk_manager.update_position(
                    pair, signal, position_size, current_price, stop_loss, take_profit
                )
                
                # Update portfolio
                self._update_portfolio_position(pair, signal, position_size, current_price)
                
                # Record trade
                self._record_trade(pair, signal, position_size, current_price, result)
                
                self.last_trade_time = datetime.now()
                
                logger.info(f"Trade executed: {signal} {position_size} {pair} @ {current_price}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'success': False, 'reason': f'Execution error: {str(e)}'}
    
    def _execute_buy_order(self, pair: str, size: float, price: float) -> Dict[str, Any]:
        """Execute a buy order."""
        try:
            # Place market order
            result = self.client.place_market_order(pair, 'buy', size)
            
            if result and 'txid' in result:
                return {
                    'success': True,
                    'order_id': result['txid'],
                    'type': 'buy',
                    'size': size,
                    'price': price,
                    'timestamp': datetime.now()
                }
            else:
                return {'success': False, 'reason': 'Order placement failed'}
                
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            return {'success': False, 'reason': f'Buy order error: {str(e)}'}
    
    def _execute_sell_order(self, pair: str, size: float, price: float) -> Dict[str, Any]:
        """Execute a sell order."""
        try:
            # Place market order
            result = self.client.place_market_order(pair, 'sell', size)
            
            if result and 'txid' in result:
                return {
                    'success': True,
                    'order_id': result['txid'],
                    'type': 'sell',
                    'size': size,
                    'price': price,
                    'timestamp': datetime.now()
                }
            else:
                return {'success': False, 'reason': 'Order placement failed'}
                
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            return {'success': False, 'reason': f'Sell order error: {str(e)}'}
    
    def _update_portfolio_position(self, pair: str, signal: str, size: float, price: float):
        """Update portfolio with new position."""
        try:
            if pair not in self.portfolio['positions']:
                self.portfolio['positions'][pair] = {
                    'size': 0.0,
                    'avg_price': 0.0,
                    'value': 0.0,
                    'unrealized_pnl': 0.0
                }
            
            position = self.portfolio['positions'][pair]
            
            if signal == 'buy':
                # Add to position
                new_size = position['size'] + size
                new_value = position['value'] + (size * price)
                new_avg_price = new_value / new_size if new_size > 0 else 0
                
                position['size'] = new_size
                position['avg_price'] = new_avg_price
                position['value'] = new_value
                
            elif signal == 'sell':
                # Reduce position
                position['size'] = max(0, position['size'] - size)
                position['value'] = position['size'] * position['avg_price']
                
                # If position is closed, remove it
                if position['size'] == 0:
                    del self.portfolio['positions'][pair]
            
            logger.debug(f"Portfolio position updated for {pair}: {position}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio position: {e}")
    
    def _record_trade(self, pair: str, signal: str, size: float, price: float, result: Dict):
        """Record trade in history."""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'pair': pair,
                'signal': signal,
                'size': size,
                'price': price,
                'order_id': result.get('order_id'),
                'success': result['success']
            }
            
            self.trade_history.append(trade_record)
            
            # Keep only last 1000 trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def close_position(self, pair: str, reason: str = 'manual') -> Dict[str, Any]:
        """Close an existing position."""
        try:
            if pair not in self.portfolio['positions']:
                return {'success': False, 'reason': 'No position found'}
            
            position = self.portfolio['positions'][pair]
            size = position['size']
            
            if size == 0:
                return {'success': False, 'reason': 'Position already closed'}
            
            # Get current price
            ticker = self.client.get_ticker(pair)
            ticker_data = list(ticker.values())[0] if ticker else {}
            current_price = float(ticker_data.get('c', [0])[0]) if ticker_data.get('c') else 0
            
            if current_price == 0:
                return {'success': False, 'reason': 'Invalid current price'}
            
            # Execute sell order
            result = self._execute_sell_order(pair, size, current_price)
            
            if result['success']:
                # Calculate P&L
                pnl_result = self.risk_manager.close_position(pair, current_price)
                
                # Update portfolio
                del self.portfolio['positions'][pair]
                
                logger.info(f"Position closed for {pair}: P&L = {pnl_result.get('pnl', 0):.2f}")
                
                return {
                    'success': True,
                    'pnl': pnl_result.get('pnl', 0),
                    'reason': reason,
                    'order_id': result.get('order_id')
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'reason': f'Error: {str(e)}'}
    
    def check_stop_losses_and_take_profits(self):
        """Check and execute stop losses and take profits."""
        try:
            # Get current prices for all open positions
            current_prices = {}
            for pair in self.portfolio['positions'].keys():
                ticker = self.client.get_ticker(pair)
                if ticker:
                    ticker_data = list(ticker.values())[0]
                    current_price = float(ticker_data.get('c', [0])[0]) if ticker_data.get('c') else 0
                    if current_price > 0:
                        current_prices[pair] = current_price
            
            # Check stop losses
            stop_loss_hits = self.risk_manager.check_stop_losses(current_prices)
            for hit in stop_loss_hits:
                logger.warning(f"Stop loss hit for {hit['pair']}: {hit['current_price']} <= {hit['stop_loss']}")
                self.close_position(hit['pair'], 'stop_loss')
            
            # Check take profits
            take_profit_hits = self.risk_manager.check_take_profits(current_prices)
            for hit in take_profit_hits:
                logger.info(f"Take profit hit for {hit['pair']}: {hit['current_price']} >= {hit['take_profit']}")
                self.close_position(hit['pair'], 'take_profit')
            
        except Exception as e:
            logger.error(f"Error checking stop losses and take profits: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            self.update_portfolio()
            
            # Calculate performance metrics
            total_pnl = 0.0
            unrealized_pnl = 0.0
            
            for pair, position in self.portfolio['positions'].items():
                # Get current price
                ticker = self.client.get_ticker(pair)
                if ticker:
                    ticker_data = list(ticker.values())[0]
                    current_price = float(ticker_data.get('c', [0])[0]) if ticker_data.get('c') else 0
                    
                    if current_price > 0:
                        position_value = position['size'] * current_price
                        unrealized_pnl += (current_price - position['avg_price']) * position['size']
                        position['current_value'] = position_value
                        position['unrealized_pnl'] = (current_price - position['avg_price']) * position['size']
            
            # Get risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            summary = {
                'timestamp': datetime.now(),
                'portfolio': self.portfolio,
                'positions': self.portfolio['positions'],
                'unrealized_pnl': unrealized_pnl,
                'total_trades': len(self.trade_history),
                'recent_trades': self.trade_history[-10:] if self.trade_history else [],
                'risk_metrics': risk_metrics,
                'trading_active': self.is_trading_active,
                'performance': {
                    'total_return': (self.portfolio['total_balance'] - config.initial_balance) / config.initial_balance,
                    'daily_pnl': risk_metrics.get('daily_pnl', 0),
                    'max_drawdown': risk_metrics.get('max_drawdown', 0)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def start_trading_session(self, pair: str) -> Dict[str, Any]:
        """Start an automated trading session."""
        try:
            logger.info(f"Starting trading session for {pair}")
            
            # Initialize portfolio
            if not self.initialize_portfolio():
                return {'success': False, 'reason': 'Failed to initialize portfolio'}
            
            # Get initial AI recommendation
            recommendation = self.ai_trader.get_trading_recommendation(pair)
            
            logger.info(f"Initial recommendation: {recommendation['recommendation']['action']}")
            
            return {
                'success': True,
                'pair': pair,
                'initial_recommendation': recommendation,
                'portfolio_initialized': True
            }
            
        except Exception as e:
            logger.error(f"Error starting trading session: {e}")
            return {'success': False, 'reason': f'Error: {str(e)}'}
    
    def stop_trading(self):
        """Stop trading and close all positions."""
        try:
            logger.info("Stopping trading session")
            self.is_trading_active = False
            
            # Close all open positions
            for pair in list(self.portfolio['positions'].keys()):
                self.close_position(pair, 'session_stop')
            
            logger.info("Trading session stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")