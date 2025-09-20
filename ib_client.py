"""Interactive Brokers client using ib_insync."""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from loguru import logger
import threading
from dataclasses import dataclass
from enum import Enum

from ib_insync import IB, Stock, Option, Forex, Contract, Order, MarketOrder, LimitOrder, StopOrder
from ib_insync import util, PortfolioItem, Position, Trade, OrderStatus as IBOrderStatus

class IBOrderType(Enum):
    """Interactive Brokers order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAIL = "trail"
    TRAIL_LIMIT = "trail_limit"

@dataclass
class IBOrder:
    """Interactive Brokers order data structure."""
    order_id: int
    contract: Contract
    order_type: IBOrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    time_in_force: str = "DAY"
    status: str = "pending"
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class IBPosition:
    """Interactive Brokers position data structure."""
    contract: Contract
    position: float
    market_price: float
    market_value: float
    average_cost: float
    unrealized_pnl: float
    realized_pnl: float
    account: str = ""

class IBClient:
    """Interactive Brokers client using ib_insync."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, 
                 client_id: int = 1, timeout: float = 10.0):
        """Initialize IB client."""
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.ib = IB()
        self.connected = False
        self.account = None
        self.positions = {}
        self.orders = {}
        self.kill_switch_active = False
        self.kill_switch_lock = threading.Lock()
        
        # Event handlers
        self.order_filled_handler = None
        self.position_update_handler = None
        self.error_handler = None
        
        logger.info(f"IB Client initialized: {host}:{port} (client_id: {client_id})")
    
    async def connect(self) -> bool:
        """Connect to Interactive Brokers."""
        try:
            if self.connected:
                return True
            
            # Connect to IB Gateway/TWS
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
            
            # Get account info
            account_summary = await self.ib.accountSummaryAsync()
            if account_summary:
                # Extract account number
                for item in account_summary:
                    if item.tag == "AccountID":
                        self.account = item.value
                        break
                
                logger.info(f"Connected to IB account: {self.account}")
            else:
                logger.warning("No account summary received")
            
            # Set up event handlers
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.positionEvent += self._on_position_update
            self.ib.errorEvent += self._on_error
            
            # Get initial positions
            await self._update_positions()
            
            self.connected = True
            logger.info("Successfully connected to Interactive Brokers")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to IB: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Interactive Brokers."""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from Interactive Brokers")
        except Exception as e:
            logger.error(f"Error disconnecting from IB: {e}")
    
    def activate_kill_switch(self):
        """Activate the kill switch to close all positions and stop trading."""
        with self.kill_switch_lock:
            if self.kill_switch_active:
                logger.warning("Kill switch already active")
                return
            
            self.kill_switch_active = True
            logger.critical("KILL SWITCH ACTIVATED - Closing all positions and stopping trading")
            
            # Close all positions asynchronously
            asyncio.create_task(self._close_all_positions())
    
    def deactivate_kill_switch(self):
        """Deactivate the kill switch."""
        with self.kill_switch_lock:
            self.kill_switch_active = False
            logger.info("Kill switch deactivated")
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        with self.kill_switch_lock:
            return self.kill_switch_active
    
    async def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        if not self.connected:
            return None
        
        try:
            account_summary = await self.ib.accountSummaryAsync()
            if not account_summary:
                return None
            
            # Parse account summary
            account_info = {}
            for item in account_summary:
                account_info[item.tag] = item.value
            
            # Get portfolio
            portfolio = await self.ib.portfolioAsync()
            positions = []
            for item in portfolio:
                positions.append({
                    'contract': self._contract_to_dict(item.contract),
                    'position': item.position,
                    'market_price': item.marketPrice,
                    'market_value': item.marketValue,
                    'average_cost': item.averageCost,
                    'unrealized_pnl': item.unrealizedPNL,
                    'realized_pnl': item.realizedPNL
                })
            
            account_info['positions'] = positions
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    async def get_positions(self) -> List[IBPosition]:
        """Get all positions."""
        try:
            portfolio = await self.ib.portfolioAsync()
            positions = []
            
            for item in portfolio:
                if item.position != 0:  # Only include non-zero positions
                    position = IBPosition(
                        contract=item.contract,
                        position=item.position,
                        market_price=item.marketPrice,
                        market_value=item.marketValue,
                        average_cost=item.averageCost,
                        unrealized_pnl=item.unrealizedPNL,
                        realized_pnl=item.realizedPNL,
                        account=self.account or ""
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def place_order(self, contract: Contract, order_type: IBOrderType, 
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         trail_amount: Optional[float] = None,
                         time_in_force: str = "DAY") -> Optional[IBOrder]:
        """Place an order."""
        if self.is_kill_switch_active():
            logger.warning("Cannot place order - kill switch is active")
            return None
        
        if not self.connected:
            logger.error("Not connected to IB")
            return None
        
        try:
            # Create order based on type
            if order_type == IBOrderType.MARKET:
                order = MarketOrder("BUY" if quantity > 0 else "SELL", abs(quantity))
            elif order_type == IBOrderType.LIMIT:
                if price is None:
                    logger.error("Price required for limit order")
                    return None
                order = LimitOrder("BUY" if quantity > 0 else "SELL", abs(quantity), price)
            elif order_type == IBOrderType.STOP:
                if stop_price is None:
                    logger.error("Stop price required for stop order")
                    return None
                order = StopOrder("BUY" if quantity > 0 else "SELL", abs(quantity), stop_price)
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return None
            
            # Set time in force
            order.tif = time_in_force
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Create order object
            ib_order = IBOrder(
                order_id=trade.order.orderId,
                contract=contract,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                trail_amount=trail_amount,
                time_in_force=time_in_force,
                status="pending"
            )
            
            # Store order
            self.orders[trade.order.orderId] = ib_order
            
            logger.info(f"Order placed: {ib_order}")
            return ib_order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an order."""
        try:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            # Cancel order
            self.ib.cancelOrder(self.orders[order_id].contract, self.orders[order_id].order_id)
            
            # Update status
            self.orders[order_id].status = "cancelled"
            
            logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        try:
            cancelled_count = 0
            for order_id in list(self.orders.keys()):
                if self.orders[order_id].status == "pending":
                    if await self.cancel_order(order_id):
                        cancelled_count += 1
            
            logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    async def close_position(self, contract: Contract) -> bool:
        """Close a position by selling/buying to zero."""
        try:
            # Get current position
            positions = await self.get_positions()
            position = None
            for pos in positions:
                if self._contracts_equal(pos.contract, contract):
                    position = pos
                    break
            
            if not position:
                logger.warning(f"No position found for contract: {contract}")
                return False
            
            # Create closing order (opposite direction)
            closing_quantity = -position.position
            order_type = IBOrderType.MARKET
            
            closing_order = await self.place_order(
                contract=contract,
                order_type=order_type,
                quantity=closing_quantity
            )
            
            if closing_order:
                logger.info(f"Closing order placed for position: {contract}")
                return True
            else:
                logger.error(f"Failed to place closing order for position: {contract}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def close_all_positions(self) -> int:
        """Close all open positions."""
        try:
            positions = await self.get_positions()
            closed_count = 0
            
            for position in positions:
                if await self.close_position(position.contract):
                    closed_count += 1
            
            logger.info(f"Closed {closed_count} positions")
            return closed_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return 0
    
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", 
                            currency: str = "USD") -> Stock:
        """Create a stock contract."""
        return Stock(symbol, exchange, currency)
    
    def create_forex_contract(self, symbol: str, exchange: str = "IDEALPRO") -> Forex:
        """Create a forex contract."""
        return Forex(symbol, exchange)
    
    def create_option_contract(self, symbol: str, expiry: str, strike: float, 
                             right: str, exchange: str = "SMART") -> Option:
        """Create an option contract."""
        return Option(symbol, expiry, strike, right, exchange)
    
    def set_order_filled_handler(self, handler: Callable):
        """Set order filled event handler."""
        self.order_filled_handler = handler
    
    def set_position_update_handler(self, handler: Callable):
        """Set position update event handler."""
        self.position_update_handler = handler
    
    def set_error_handler(self, handler: Callable):
        """Set error event handler."""
        self.error_handler = handler
    
    async def _update_positions(self):
        """Update positions from IB."""
        try:
            portfolio = await self.ib.portfolioAsync()
            self.positions = {}
            
            for item in portfolio:
                if item.position != 0:
                    key = self._contract_key(item.contract)
                    self.positions[key] = item
                    
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _on_order_status(self, trade: Trade):
        """Handle order status updates."""
        try:
            order_id = trade.order.orderId
            status = trade.orderStatus.status
            
            if order_id in self.orders:
                self.orders[order_id].status = status.lower()
                self.orders[order_id].filled_quantity = trade.orderStatus.filled
                self.orders[order_id].average_price = trade.orderStatus.avgFillPrice
                
                logger.info(f"Order {order_id} status: {status}")
                
                # Call handler if order is filled
                if status == "Filled" and self.order_filled_handler:
                    self.order_filled_handler(trade)
                    
        except Exception as e:
            logger.error(f"Error handling order status: {e}")
    
    def _on_position_update(self, position: Position):
        """Handle position updates."""
        try:
            key = self._contract_key(position.contract)
            self.positions[key] = position
            
            logger.debug(f"Position updated: {key} = {position.position}")
            
            # Call handler
            if self.position_update_handler:
                self.position_update_handler(position)
                
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str):
        """Handle error events."""
        try:
            logger.error(f"IB Error {errorCode}: {errorString}")
            
            # Call handler
            if self.error_handler:
                self.error_handler(reqId, errorCode, errorString)
                
        except Exception as e:
            logger.error(f"Error handling IB error: {e}")
    
    async def _close_all_positions(self):
        """Close all positions (internal method)."""
        try:
            await self.close_all_positions()
        except Exception as e:
            logger.error(f"Error in kill switch position closing: {e}")
    
    def _contract_to_dict(self, contract: Contract) -> Dict:
        """Convert contract to dictionary."""
        return {
            'symbol': contract.symbol,
            'secType': contract.secType,
            'exchange': contract.exchange,
            'currency': contract.currency,
            'localSymbol': contract.localSymbol,
            'tradingClass': contract.tradingClass
        }
    
    def _contract_key(self, contract: Contract) -> str:
        """Generate unique key for contract."""
        return f"{contract.symbol}_{contract.secType}_{contract.exchange}_{contract.currency}"
    
    def _contracts_equal(self, contract1: Contract, contract2: Contract) -> bool:
        """Check if two contracts are equal."""
        return (contract1.symbol == contract2.symbol and
                contract1.secType == contract2.secType and
                contract1.exchange == contract2.exchange and
                contract1.currency == contract2.currency)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()