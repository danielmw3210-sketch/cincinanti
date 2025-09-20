"""Enhanced trading executor with MT5, IB, and CCXT integration."""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from dataclasses import dataclass
from enum import Enum

# Import our modules
from mt5_client import MT5Client, OrderType as MT5OrderType, OrderStatus as MT5OrderStatus
from ib_client import IBClient, IBOrderType
from database import DatabaseManager, TradingPlatform, OrderStatus, PositionStatus
import ccxt

class TradingMode(Enum):
    """Trading mode enumeration."""
    MT5_ONLY = "mt5_only"
    IB_ONLY = "ib_only"
    CCXT_ONLY = "ccxt_only"
    MULTI_PLATFORM = "multi_platform"

@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: Optional[float] = None
    platform: Optional[TradingPlatform] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EnhancedTradingExecutor:
    """Enhanced trading executor with multi-platform support."""
    
    def __init__(self, 
                 mt5_config: Dict = None,
                 ib_config: Dict = None,
                 ccxt_config: Dict = None,
                 trading_mode: TradingMode = TradingMode.MT5_ONLY,
                 micro_lot_size: float = 0.01):
        """Initialize enhanced trading executor."""
        self.mt5_config = mt5_config or {}
        self.ib_config = ib_config or {}
        self.ccxt_config = ccxt_config or {}
        self.trading_mode = trading_mode
        self.micro_lot_size = micro_lot_size
        
        # Initialize clients
        self.mt5_client = None
        self.ib_client = None
        self.ccxt_exchanges = {}
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Trading state
        self.is_trading_active = True
        self.positions = {}
        self.orders = {}
        self.kill_switch_active = False
        self.kill_switch_lock = threading.Lock()
        
        # Risk management
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_position_size = 0.1  # 10% max position size
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info("Enhanced Trading Executor initialized")
    
    async def initialize(self) -> bool:
        """Initialize all trading platforms."""
        try:
            logger.info("Initializing enhanced trading executor...")
            
            # Initialize MT5 if configured
            if self.trading_mode in [TradingMode.MT5_ONLY, TradingMode.MULTI_PLATFORM]:
                await self._initialize_mt5()
            
            # Initialize IB if configured
            if self.trading_mode in [TradingMode.IB_ONLY, TradingMode.MULTI_PLATFORM]:
                await self._initialize_ib()
            
            # Initialize CCXT if configured
            if self.trading_mode in [TradingMode.CCXT_ONLY, TradingMode.MULTI_PLATFORM]:
                await self._initialize_ccxt()
            
            logger.info("Enhanced trading executor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading executor: {e}")
            return False
    
    async def _initialize_mt5(self):
        """Initialize MT5 client."""
        try:
            self.mt5_client = MT5Client(
                login=self.mt5_config.get('login'),
                password=self.mt5_config.get('password'),
                server=self.mt5_config.get('server'),
                timeout=self.mt5_config.get('timeout', 60000),
                portable=self.mt5_config.get('portable', False)
            )
            
            if not self.mt5_client.connect():
                raise Exception("Failed to connect to MT5")
            
            logger.info("MT5 client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            raise
    
    async def _initialize_ib(self):
        """Initialize Interactive Brokers client."""
        try:
            self.ib_client = IBClient(
                host=self.ib_config.get('host', '127.0.0.1'),
                port=self.ib_config.get('port', 7497),
                client_id=self.ib_config.get('client_id', 1),
                timeout=self.ib_config.get('timeout', 10.0)
            )
            
            if not await self.ib_client.connect():
                raise Exception("Failed to connect to Interactive Brokers")
            
            logger.info("Interactive Brokers client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing IB: {e}")
            raise
    
    async def _initialize_ccxt(self):
        """Initialize CCXT exchanges."""
        try:
            for exchange_name, config in self.ccxt_config.items():
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'apiKey': config.get('api_key'),
                    'secret': config.get('secret'),
                    'password': config.get('password'),
                    'sandbox': config.get('sandbox', True),
                    'enableRateLimit': True,
                })
                
                # Test connection
                await exchange.load_markets()
                
                self.ccxt_exchanges[exchange_name] = exchange
                logger.info(f"CCXT exchange {exchange_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing CCXT: {e}")
            raise
    
    def activate_kill_switch(self):
        """Activate kill switch across all platforms."""
        with self.kill_switch_lock:
            if self.kill_switch_active:
                logger.warning("Kill switch already active")
                return
            
            self.kill_switch_active = True
            self.is_trading_active = False
            
            logger.critical("KILL SWITCH ACTIVATED - Stopping all trading")
            
            # Activate kill switch on all platforms
            if self.mt5_client:
                self.mt5_client.activate_kill_switch()
            
            if self.ib_client:
                self.ib_client.activate_kill_switch()
            
            # Log kill switch event
            self.db_manager.log_risk_event({
                'event_type': 'kill_switch_activated',
                'severity': 'critical',
                'message': 'Kill switch activated - all trading stopped',
                'metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'platforms': self._get_active_platforms()
                }
            })
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch."""
        with self.kill_switch_lock:
            self.kill_switch_active = False
            self.is_trading_active = True
            
            logger.info("Kill switch deactivated")
            
            # Deactivate kill switch on all platforms
            if self.mt5_client:
                self.mt5_client.deactivate_kill_switch()
            
            if self.ib_client:
                self.ib_client.deactivate_kill_switch()
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        with self.kill_switch_lock:
            return self.kill_switch_active
    
    async def execute_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute trade based on signal."""
        if self.is_kill_switch_active():
            return {
                'success': False,
                'error': 'Kill switch is active',
                'platform': None
            }
        
        if not self.is_trading_active:
            return {
                'success': False,
                'error': 'Trading is not active',
                'platform': None
            }
        
        try:
            # Determine platform for execution
            platform = self._select_platform(signal)
            if not platform:
                return {
                    'success': False,
                    'error': 'No suitable platform available',
                    'platform': None
                }
            
            # Execute trade on selected platform
            if platform == TradingPlatform.MT5:
                return await self._execute_mt5_trade(signal)
            elif platform == TradingPlatform.IB:
                return await self._execute_ib_trade(signal)
            elif platform == TradingPlatform.CCXT:
                return await self._execute_ccxt_trade(signal)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported platform: {platform}',
                    'platform': platform
                }
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': None
            }
    
    def _select_platform(self, signal: TradingSignal) -> Optional[TradingPlatform]:
        """Select the best platform for executing the trade."""
        # If signal specifies a platform, use it
        if signal.platform:
            return signal.platform
        
        # Select based on trading mode and symbol
        if self.trading_mode == TradingMode.MT5_ONLY and self.mt5_client:
            return TradingPlatform.MT5
        elif self.trading_mode == TradingMode.IB_ONLY and self.ib_client:
            return TradingPlatform.IB
        elif self.trading_mode == TradingMode.CCXT_ONLY and self.ccxt_exchanges:
            return TradingPlatform.CCXT
        elif self.trading_mode == TradingMode.MULTI_PLATFORM:
            # Select based on symbol type
            if self._is_forex_symbol(signal.symbol) and self.mt5_client:
                return TradingPlatform.MT5
            elif self._is_stock_symbol(signal.symbol) and self.ib_client:
                return TradingPlatform.IB
            elif self._is_crypto_symbol(signal.symbol) and self.ccxt_exchanges:
                return TradingPlatform.CCXT
            else:
                # Fallback to first available platform
                if self.mt5_client:
                    return TradingPlatform.MT5
                elif self.ib_client:
                    return TradingPlatform.IB
                elif self.ccxt_exchanges:
                    return TradingPlatform.CCXT
        
        return None
    
    async def _execute_mt5_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute trade on MT5."""
        try:
            # Convert signal to MT5 order type
            order_type = MT5OrderType.BUY if signal.action == 'buy' else MT5OrderType.SELL
            
            # Use micro lot size for testing
            volume = signal.volume or self.micro_lot_size
            
            # Place order
            order = self.mt5_client.place_order(
                symbol=signal.symbol,
                order_type=order_type,
                volume=volume,
                price=signal.price,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"AI Signal: {signal.confidence:.3f}"
            )
            
            if not order:
                return {
                    'success': False,
                    'error': 'Failed to place MT5 order',
                    'platform': TradingPlatform.MT5
                }
            
            # Log order to database
            order_data = {
                'ticket': order.ticket,
                'platform': 'mt5',
                'symbol': order.symbol,
                'order_type': order.order_type.value,
                'volume': order.volume,
                'price': order.price,
                'sl': order.sl,
                'tp': order.tp,
                'status': order.status.value,
                'comment': order.comment,
                'metadata': {
                    'signal_confidence': signal.confidence,
                    'signal_action': signal.action,
                    'signal_metadata': signal.metadata
                }
            }
            
            order_id = self.db_manager.log_order(order_data)
            
            return {
                'success': True,
                'order_id': order_id,
                'ticket': order.ticket,
                'platform': TradingPlatform.MT5,
                'symbol': order.symbol,
                'volume': order.volume,
                'price': order.price,
                'order_type': order.order_type.value
            }
            
        except Exception as e:
            logger.error(f"Error executing MT5 trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': TradingPlatform.MT5
            }
    
    async def _execute_ib_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute trade on Interactive Brokers."""
        try:
            # Create contract based on symbol
            contract = self._create_ib_contract(signal.symbol)
            if not contract:
                return {
                    'success': False,
                    'error': f'Cannot create contract for symbol: {signal.symbol}',
                    'platform': TradingPlatform.IB
                }
            
            # Convert signal to IB order type
            order_type = IBOrderType.MARKET  # Use market orders for simplicity
            
            # Use micro lot size for testing
            quantity = signal.volume or self.micro_lot_size
            if signal.action == 'sell':
                quantity = -quantity  # Negative quantity for sell
            
            # Place order
            order = await self.ib_client.place_order(
                contract=contract,
                order_type=order_type,
                quantity=quantity,
                price=signal.price
            )
            
            if not order:
                return {
                    'success': False,
                    'error': 'Failed to place IB order',
                    'platform': TradingPlatform.IB
                }
            
            # Log order to database
            order_data = {
                'ticket': order.order_id,
                'platform': 'interactive_brokers',
                'symbol': signal.symbol,
                'order_type': order.order_type.value,
                'volume': abs(order.quantity),
                'price': order.price,
                'status': order.status,
                'comment': f"AI Signal: {signal.confidence:.3f}",
                'metadata': {
                    'signal_confidence': signal.confidence,
                    'signal_action': signal.action,
                    'signal_metadata': signal.metadata,
                    'contract': self.ib_client._contract_to_dict(contract)
                }
            }
            
            order_id = self.db_manager.log_order(order_data)
            
            return {
                'success': True,
                'order_id': order_id,
                'order_id_ib': order.order_id,
                'platform': TradingPlatform.IB,
                'symbol': signal.symbol,
                'volume': abs(order.quantity),
                'price': order.price,
                'order_type': order.order_type.value
            }
            
        except Exception as e:
            logger.error(f"Error executing IB trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': TradingPlatform.IB
            }
    
    async def _execute_ccxt_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute trade on CCXT exchange."""
        try:
            # Select exchange (use first available for now)
            exchange_name = list(self.ccxt_exchanges.keys())[0]
            exchange = self.ccxt_exchanges[exchange_name]
            
            # Use micro lot size for testing
            amount = signal.volume or self.micro_lot_size
            
            # Place order
            if signal.action == 'buy':
                order = await exchange.create_market_buy_order(signal.symbol, amount)
            else:
                order = await exchange.create_market_sell_order(signal.symbol, amount)
            
            # Log order to database
            order_data = {
                'ticket': order.get('id'),
                'platform': 'ccxt',
                'symbol': order.get('symbol'),
                'order_type': signal.action,
                'volume': order.get('amount'),
                'price': order.get('price'),
                'status': order.get('status', 'pending'),
                'comment': f"AI Signal: {signal.confidence:.3f}",
                'metadata': {
                    'signal_confidence': signal.confidence,
                    'signal_action': signal.action,
                    'signal_metadata': signal.metadata,
                    'exchange': exchange_name,
                    'ccxt_order': order
                }
            }
            
            order_id = self.db_manager.log_order(order_data)
            
            return {
                'success': True,
                'order_id': order_id,
                'ccxt_order_id': order.get('id'),
                'platform': TradingPlatform.CCXT,
                'symbol': order.get('symbol'),
                'volume': order.get('amount'),
                'price': order.get('price'),
                'order_type': signal.action,
                'exchange': exchange_name
            }
            
        except Exception as e:
            logger.error(f"Error executing CCXT trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': TradingPlatform.CCXT
            }
    
    def _create_ib_contract(self, symbol: str):
        """Create IB contract from symbol."""
        try:
            # Simple symbol parsing - extend as needed
            if '/' in symbol:
                # Forex pair
                base, quote = symbol.split('/')
                return self.ib_client.create_forex_contract(f"{base}{quote}")
            else:
                # Stock
                return self.ib_client.create_stock_contract(symbol)
        except Exception as e:
            logger.error(f"Error creating IB contract for {symbol}: {e}")
            return None
    
    def _is_forex_symbol(self, symbol: str) -> bool:
        """Check if symbol is forex."""
        return '/' in symbol and len(symbol.split('/')) == 2
    
    def _is_stock_symbol(self, symbol: str) -> bool:
        """Check if symbol is stock."""
        return not '/' in symbol and not symbol.endswith('USD')
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is cryptocurrency."""
        return '/' in symbol and any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA'])
    
    def _get_active_platforms(self) -> List[str]:
        """Get list of active platforms."""
        platforms = []
        if self.mt5_client and self.mt5_client.connected:
            platforms.append('mt5')
        if self.ib_client and self.ib_client.connected:
            platforms.append('ib')
        if self.ccxt_exchanges:
            platforms.append('ccxt')
        return platforms
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary from all platforms."""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'platforms': {},
            'total_equity': 0.0,
            'total_pnl': 0.0,
            'total_positions': 0,
            'kill_switch_active': self.is_kill_switch_active(),
            'trading_active': self.is_trading_active
        }
        
        # Get MT5 portfolio
        if self.mt5_client and self.mt5_client.connected:
            try:
                account_info = self.mt5_client.get_account_info()
                positions = self.mt5_client.get_positions()
                
                summary['platforms']['mt5'] = {
                    'account_info': account_info,
                    'positions': len(positions),
                    'equity': account_info.get('equity', 0) if account_info else 0,
                    'pnl': account_info.get('profit', 0) if account_info else 0
                }
                
                summary['total_equity'] += summary['platforms']['mt5']['equity']
                summary['total_pnl'] += summary['platforms']['mt5']['pnl']
                summary['total_positions'] += summary['platforms']['mt5']['positions']
                
            except Exception as e:
                logger.error(f"Error getting MT5 portfolio: {e}")
        
        # Get IB portfolio
        if self.ib_client and self.ib_client.connected:
            try:
                account_info = await self.ib_client.get_account_info()
                positions = await self.ib_client.get_positions()
                
                summary['platforms']['ib'] = {
                    'account_info': account_info,
                    'positions': len(positions),
                    'equity': 0,  # Calculate from account info
                    'pnl': 0  # Calculate from positions
                }
                
            except Exception as e:
                logger.error(f"Error getting IB portfolio: {e}")
        
        return summary
    
    async def close_all_positions(self) -> Dict[str, int]:
        """Close all positions across all platforms."""
        results = {}
        
        if self.mt5_client and self.mt5_client.connected:
            try:
                closed = self.mt5_client.close_all_positions()
                results['mt5'] = closed
            except Exception as e:
                logger.error(f"Error closing MT5 positions: {e}")
                results['mt5'] = 0
        
        if self.ib_client and self.ib_client.connected:
            try:
                closed = await self.ib_client.close_all_positions()
                results['ib'] = closed
            except Exception as e:
                logger.error(f"Error closing IB positions: {e}")
                results['ib'] = 0
        
        return results
    
    async def stop_trading(self):
        """Stop trading on all platforms."""
        self.is_trading_active = False
        
        # Close all positions
        await self.close_all_positions()
        
        logger.info("Trading stopped on all platforms")
    
    async def cleanup(self):
        """Cleanup all connections."""
        try:
            if self.mt5_client:
                self.mt5_client.disconnect()
            
            if self.ib_client:
                await self.ib_client.disconnect()
            
            for exchange in self.ccxt_exchanges.values():
                await exchange.close()
            
            logger.info("All connections cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.cleanup())