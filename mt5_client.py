"""MetaTrader5 client for trading operations."""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import time
import threading
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    """Order types for MT5."""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"

class OrderStatus(Enum):
    """Order status for MT5."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order data structure."""
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: str = ""
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """Position data structure."""
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    price_open: float
    price_current: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    profit: float = 0.0
    swap: float = 0.0
    comment: str = ""
    time: datetime = None
    
    def __post_init__(self):
        if self.time is None:
            self.time = datetime.now()

class MT5Client:
    """MetaTrader5 client for trading operations."""
    
    def __init__(self, 
                 login: int = None, 
                 password: str = None, 
                 server: str = None,
                 timeout: int = 60000,
                 portable: bool = False):
        """Initialize MT5 client."""
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.portable = portable
        self.connected = False
        self.account_info = None
        self.symbols_info = {}
        self.kill_switch_active = False
        self.kill_switch_lock = threading.Lock()
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        
        logger.info("MT5 initialized successfully")
    
    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            if self.connected:
                return True
            
            # Login to account
            if self.login and self.password and self.server:
                if not mt5.login(login=self.login, password=self.password, server=self.server, timeout=self.timeout):
                    logger.error(f"Failed to login to MT5 account {self.login}")
                    return False
                logger.info(f"Successfully logged in to MT5 account {self.login}")
            else:
                logger.info("Using existing MT5 connection")
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Account info: {self.account_info}")
            
            # Get available symbols
            symbols = mt5.symbols_get()
            if symbols:
                for symbol in symbols:
                    self.symbols_info[symbol.name] = symbol
                logger.info(f"Loaded {len(self.symbols_info)} symbols")
            
            self.connected = True
            logger.info("Successfully connected to MT5")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5."""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
    
    def activate_kill_switch(self):
        """Activate the kill switch to close all positions and stop trading."""
        with self.kill_switch_lock:
            if self.kill_switch_active:
                logger.warning("Kill switch already active")
                return
            
            self.kill_switch_active = True
            logger.critical("KILL SWITCH ACTIVATED - Closing all positions and stopping trading")
            
            # Close all open positions
            self.close_all_positions()
            
            # Cancel all pending orders
            self.cancel_all_orders()
    
    def deactivate_kill_switch(self):
        """Deactivate the kill switch."""
        with self.kill_switch_lock:
            self.kill_switch_active = False
            logger.info("Kill switch deactivated")
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        with self.kill_switch_lock:
            return self.kill_switch_active
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        if not self.connected:
            return None
        
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            return {
                'login': account.login,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'currency': account.currency,
                'leverage': account.leverage,
                'profit': account.profit,
                'server': account.server,
                'name': account.name
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol {symbol} not found")
                return None
            
            return {
                'name': symbol_info.name,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'currency_margin': symbol_info.currency_margin,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_mode': symbol_info.trade_mode,
                'trade_stops_level': symbol_info.trade_stops_level,
                'trade_freeze_level': symbol_info.trade_freeze_level,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'price_min': symbol_info.price_min,
                'price_max': symbol_info.price_max,
                'trade_calc_mode': symbol_info.trade_calc_mode,
                'trade_mode_flags': symbol_info.trade_mode_flags
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for symbol."""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}")
                return None
            
            return {
                'symbol': symbol,
                'time': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'flags': tick.flags
            }
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: int, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical data for symbol."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"No historical data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def place_order(self, 
                   symbol: str, 
                   order_type: OrderType, 
                   volume: float, 
                   price: Optional[float] = None,
                   sl: Optional[float] = None, 
                   tp: Optional[float] = None,
                   comment: str = "",
                   magic: int = 0) -> Optional[Order]:
        """Place an order."""
        if self.is_kill_switch_active():
            logger.warning("Cannot place order - kill switch is active")
            return None
        
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Normalize volume
            volume = self._normalize_volume(symbol, volume)
            if volume is None:
                return None
            
            # Get current price if not provided
            if price is None:
                tick = self.get_current_price(symbol)
                if not tick:
                    return None
                price = tick['ask'] if order_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP] else tick['bid']
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": self._get_mt5_order_type(order_type),
                "price": price,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL and TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            if result is None:
                logger.error("Order send failed - no result")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None
            
            # Create order object
            order = Order(
                ticket=result.order,
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=price,
                sl=sl,
                tp=tp,
                comment=comment,
                status=OrderStatus.FILLED
            )
            
            logger.info(f"Order placed successfully: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket."""
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.warning(f"Position {ticket} not found")
                return False
            
            position = positions[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "deviation": 20,
                "magic": position.magic,
                "comment": f"Close position {ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(request)
            if result is None:
                logger.error("Close position failed - no result")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close position failed: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"Position {ticket} closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def close_all_positions(self) -> int:
        """Close all open positions."""
        try:
            positions = mt5.positions_get()
            if not positions:
                logger.info("No open positions to close")
                return 0
            
            closed_count = 0
            for position in positions:
                if self.close_position(position.ticket):
                    closed_count += 1
            
            logger.info(f"Closed {closed_count} positions")
            return closed_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return 0
    
    def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        try:
            orders = mt5.orders_get()
            if not orders:
                logger.info("No pending orders to cancel")
                return 0
            
            cancelled_count = 0
            for order in orders:
                if self.cancel_order(order.ticket):
                    cancelled_count += 1
            
            logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    def cancel_order(self, ticket: int) -> bool:
        """Cancel a pending order."""
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            if result is None:
                logger.error("Cancel order failed - no result")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Cancel order failed: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"Order {ticket} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {ticket}: {e}")
            return False
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            positions = mt5.positions_get()
            if not positions:
                return []
            
            result = []
            for pos in positions:
                position = Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    order_type=self._get_order_type_from_mt5(pos.type),
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    sl=pos.sl,
                    tp=pos.tp,
                    profit=pos.profit,
                    swap=pos.swap,
                    comment=pos.comment,
                    time=datetime.fromtimestamp(pos.time)
                )
                result.append(position)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_orders(self) -> List[Order]:
        """Get all pending orders."""
        try:
            orders = mt5.orders_get()
            if not orders:
                return []
            
            result = []
            for order in orders:
                order_obj = Order(
                    ticket=order.ticket,
                    symbol=order.symbol,
                    order_type=self._get_order_type_from_mt5(order.type),
                    volume=order.volume_initial,
                    price=order.price_open,
                    sl=order.sl,
                    tp=order.tp,
                    comment=order.comment,
                    status=OrderStatus.PENDING,
                    timestamp=datetime.fromtimestamp(order.time_setup)
                )
                result.append(order_obj)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def _normalize_volume(self, symbol: str, volume: float) -> Optional[float]:
        """Normalize volume according to symbol requirements."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Round to volume step
            volume_step = symbol_info['volume_step']
            normalized_volume = round(volume / volume_step) * volume_step
            
            # Check min/max limits
            if normalized_volume < symbol_info['volume_min']:
                logger.warning(f"Volume {normalized_volume} below minimum {symbol_info['volume_min']}")
                return None
            
            if normalized_volume > symbol_info['volume_max']:
                logger.warning(f"Volume {normalized_volume} above maximum {symbol_info['volume_max']}")
                return None
            
            return normalized_volume
            
        except Exception as e:
            logger.error(f"Error normalizing volume: {e}")
            return None
    
    def _get_mt5_order_type(self, order_type: OrderType) -> int:
        """Convert OrderType to MT5 order type."""
        mapping = {
            OrderType.BUY: mt5.ORDER_TYPE_BUY,
            OrderType.SELL: mt5.ORDER_TYPE_SELL,
            OrderType.BUY_LIMIT: mt5.ORDER_TYPE_BUY_LIMIT,
            OrderType.SELL_LIMIT: mt5.ORDER_TYPE_SELL_LIMIT,
            OrderType.BUY_STOP: mt5.ORDER_TYPE_BUY_STOP,
            OrderType.SELL_STOP: mt5.ORDER_TYPE_SELL_STOP,
        }
        return mapping.get(order_type, mt5.ORDER_TYPE_BUY)
    
    def _get_order_type_from_mt5(self, mt5_type: int) -> OrderType:
        """Convert MT5 order type to OrderType."""
        mapping = {
            mt5.ORDER_TYPE_BUY: OrderType.BUY,
            mt5.ORDER_TYPE_SELL: OrderType.SELL,
            mt5.ORDER_TYPE_BUY_LIMIT: OrderType.BUY_LIMIT,
            mt5.ORDER_TYPE_SELL_LIMIT: OrderType.SELL_LIMIT,
            mt5.ORDER_TYPE_BUY_STOP: OrderType.BUY_STOP,
            mt5.ORDER_TYPE_SELL_STOP: OrderType.SELL_STOP,
        }
        return mapping.get(mt5_type, OrderType.BUY)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()