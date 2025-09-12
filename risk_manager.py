"""Risk management and position sizing module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from config import config

class RiskManager:
    """Handles risk management and position sizing."""
    
    def __init__(self):
        self.max_position_size = config.max_position_size
        self.risk_per_trade = config.risk_per_trade
        self.max_daily_loss = config.max_daily_loss
        self.initial_balance = config.initial_balance
        
        # Track daily performance
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Track open positions
        self.open_positions = {}
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
    def reset_daily_tracking(self):
        """Reset daily tracking metrics."""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            logger.info("Daily risk tracking reset")
    
    def calculate_position_size(self, 
                              signal_confidence: float, 
                              current_price: float, 
                              stop_loss_price: float,
                              account_balance: float,
                              pair: str) -> Dict[str, float]:
        """Calculate optimal position size based on risk parameters."""
        try:
            self.reset_daily_tracking()
            
            # Check if we've hit daily loss limit
            if self.daily_pnl <= -self.max_daily_loss * account_balance:
                logger.warning("Daily loss limit reached, reducing position size")
                return {'size': 0.0, 'reason': 'Daily loss limit reached'}
            
            # Calculate risk amount
            risk_amount = account_balance * self.risk_per_trade
            
            # Adjust risk based on signal confidence
            confidence_multiplier = min(signal_confidence * 2, 1.0)  # Cap at 1.0
            adjusted_risk = risk_amount * confidence_multiplier
            
            # Calculate position size based on stop loss distance
            price_risk = abs(current_price - stop_loss_price) / current_price
            
            if price_risk == 0:
                logger.warning("Stop loss price equals current price, cannot calculate position size")
                return {'size': 0.0, 'reason': 'Invalid stop loss price'}
            
            # Position size = Risk amount / Price risk
            position_value = adjusted_risk / price_risk
            position_size = position_value / current_price
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / current_price
            
            if position_size > max_position_size:
                position_size = max_position_size
                logger.info(f"Position size capped at maximum: {max_position_size}")
            
            # Check if we have enough balance
            required_balance = position_size * current_price
            if required_balance > account_balance:
                position_size = account_balance / current_price
                logger.warning("Insufficient balance, reducing position size")
            
            return {
                'size': position_size,
                'value': position_size * current_price,
                'risk_amount': adjusted_risk,
                'confidence_multiplier': confidence_multiplier,
                'price_risk': price_risk
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'size': 0.0, 'reason': f'Error: {str(e)}'}
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          signal_type: str, 
                          volatility: float,
                          atr: Optional[float] = None) -> float:
        """Calculate stop loss price."""
        try:
            if signal_type == 'buy':
                # For long positions, stop loss below entry
                if atr:
                    stop_loss = entry_price - (2 * atr)  # 2x ATR stop loss
                else:
                    stop_loss = entry_price * (1 - volatility * 2)  # 2x volatility stop loss
            else:  # sell
                # For short positions, stop loss above entry
                if atr:
                    stop_loss = entry_price + (2 * atr)  # 2x ATR stop loss
                else:
                    stop_loss = entry_price * (1 + volatility * 2)  # 2x volatility stop loss
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price  # Return entry price as fallback
    
    def calculate_take_profit(self, 
                            entry_price: float, 
                            signal_type: str, 
                            risk_reward_ratio: float = 2.0,
                            stop_loss_price: float = None) -> float:
        """Calculate take profit price."""
        try:
            if stop_loss_price:
                # Calculate based on risk-reward ratio
                risk_distance = abs(entry_price - stop_loss_price)
                
                if signal_type == 'buy':
                    take_profit = entry_price + (risk_distance * risk_reward_ratio)
                else:  # sell
                    take_profit = entry_price - (risk_distance * risk_reward_ratio)
            else:
                # Calculate based on percentage
                profit_percentage = 0.02 * risk_reward_ratio  # 2% base profit
                
                if signal_type == 'buy':
                    take_profit = entry_price * (1 + profit_percentage)
                else:  # sell
                    take_profit = entry_price * (1 - profit_percentage)
            
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return entry_price  # Return entry price as fallback
    
    def validate_trade(self, 
                     pair: str, 
                     signal_type: str, 
                     position_size: float, 
                     entry_price: float,
                     account_balance: float) -> Dict[str, any]:
        """Validate if a trade meets risk management criteria."""
        try:
            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': []
            }
            
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss * account_balance:
                validation_result['valid'] = False
                validation_result['errors'].append('Daily loss limit exceeded')
            
            # Check position size limits
            position_value = position_size * entry_price
            max_position_value = account_balance * self.max_position_size
            
            if position_value > max_position_value:
                validation_result['valid'] = False
                validation_result['errors'].append('Position size exceeds maximum allowed')
            
            # Check account balance
            if position_value > account_balance:
                validation_result['valid'] = False
                validation_result['errors'].append('Insufficient account balance')
            
            # Check for existing position in same pair
            if pair in self.open_positions:
                validation_result['warnings'].append(f'Existing position in {pair}')
            
            # Check daily trade limit (optional)
            if self.daily_trades > 50:  # Arbitrary limit
                validation_result['warnings'].append('High number of daily trades')
            
            # Check drawdown
            if self.current_drawdown > 0.1:  # 10% drawdown
                validation_result['warnings'].append('High current drawdown')
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}']
            }
    
    def update_position(self, 
                       pair: str, 
                       signal_type: str, 
                       size: float, 
                       entry_price: float,
                       stop_loss: float,
                       take_profit: float):
        """Update position tracking."""
        try:
            self.open_positions[pair] = {
                'type': signal_type,
                'size': size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'unrealized_pnl': 0.0
            }
            
            self.daily_trades += 1
            logger.info(f"Position updated for {pair}: {signal_type} {size} @ {entry_price}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def close_position(self, pair: str, exit_price: float) -> Dict[str, any]:
        """Close a position and calculate P&L."""
        try:
            if pair not in self.open_positions:
                logger.warning(f"No open position found for {pair}")
                return {'pnl': 0.0, 'reason': 'No open position'}
            
            position = self.open_positions[pair]
            
            # Calculate P&L
            if position['type'] == 'buy':
                pnl = (exit_price - position['entry_price']) * position['size']
            else:  # sell
                pnl = (position['entry_price'] - exit_price) * position['size']
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Update drawdown tracking
            self._update_drawdown(pnl)
            
            # Remove position
            del self.open_positions[pair]
            
            logger.info(f"Position closed for {pair}: P&L = {pnl:.2f}")
            
            return {
                'pnl': pnl,
                'position_type': position['type'],
                'size': position['size'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'duration': datetime.now() - position['entry_time']
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'pnl': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _update_drawdown(self, pnl: float):
        """Update drawdown metrics."""
        try:
            # Update peak balance
            if pnl > 0:
                self.peak_balance += pnl
            
            # Calculate current drawdown
            current_balance = self.peak_balance + self.daily_pnl
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            
            # Update maximum drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
        except Exception as e:
            logger.error(f"Error updating drawdown: {e}")
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict[str, any]]:
        """Check if any positions should be stopped out."""
        try:
            stop_loss_hits = []
            
            for pair, position in self.open_positions.items():
                if pair not in current_prices:
                    continue
                
                current_price = current_prices[pair]
                stop_loss = position['stop_loss']
                
                # Check stop loss
                if position['type'] == 'buy' and current_price <= stop_loss:
                    stop_loss_hits.append({
                        'pair': pair,
                        'reason': 'stop_loss',
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'position': position
                    })
                elif position['type'] == 'sell' and current_price >= stop_loss:
                    stop_loss_hits.append({
                        'pair': pair,
                        'reason': 'stop_loss',
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'position': position
                    })
            
            return stop_loss_hits
            
        except Exception as e:
            logger.error(f"Error checking stop losses: {e}")
            return []
    
    def check_take_profits(self, current_prices: Dict[str, float]) -> List[Dict[str, any]]:
        """Check if any positions should take profit."""
        try:
            take_profit_hits = []
            
            for pair, position in self.open_positions.items():
                if pair not in current_prices:
                    continue
                
                current_price = current_prices[pair]
                take_profit = position['take_profit']
                
                # Check take profit
                if position['type'] == 'buy' and current_price >= take_profit:
                    take_profit_hits.append({
                        'pair': pair,
                        'reason': 'take_profit',
                        'current_price': current_price,
                        'take_profit': take_profit,
                        'position': position
                    })
                elif position['type'] == 'sell' and current_price <= take_profit:
                    take_profit_hits.append({
                        'pair': pair,
                        'reason': 'take_profit',
                        'current_price': current_price,
                        'take_profit': take_profit,
                        'position': position
                    })
            
            return take_profit_hits
            
        except Exception as e:
            logger.error(f"Error checking take profits: {e}")
            return []
    
    def get_risk_metrics(self) -> Dict[str, any]:
        """Get current risk metrics."""
        try:
            total_unrealized_pnl = 0.0
            for position in self.open_positions.values():
                # This would need current prices to calculate accurately
                total_unrealized_pnl += position.get('unrealized_pnl', 0.0)
            
            return {
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'open_positions': len(self.open_positions),
                'total_unrealized_pnl': total_unrealized_pnl,
                'risk_per_trade': self.risk_per_trade,
                'max_position_size': self.max_position_size,
                'max_daily_loss': self.max_daily_loss
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    def emergency_stop(self) -> bool:
        """Emergency stop all trading."""
        try:
            if config.emergency_stop:
                logger.critical("EMERGENCY STOP ACTIVATED")
                return True
            
            # Check if we've exceeded maximum drawdown
            if self.current_drawdown > 0.2:  # 20% drawdown
                logger.critical(f"Maximum drawdown exceeded: {self.current_drawdown:.2%}")
                return True
            
            # Check if daily loss is too high
            if self.daily_pnl < -self.max_daily_loss * self.initial_balance:
                logger.critical(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in emergency stop check: {e}")
            return True  # Default to stopping on error