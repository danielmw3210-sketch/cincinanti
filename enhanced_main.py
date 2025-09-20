"""Enhanced main entry point for multi-platform trading system."""

import asyncio
import signal
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
import json

# Import our modules
from enhanced_config import config
from enhanced_trading_executor import EnhancedTradingExecutor, TradingSignal, TradingMode
from database import DatabaseManager, TradingPlatform
from dreamfactory_api import app
import uvicorn

class EnhancedTradingSystem:
    """Enhanced trading system with multi-platform support."""
    
    def __init__(self):
        """Initialize enhanced trading system."""
        self.trading_executor = None
        self.db_manager = None
        self.is_running = False
        self.api_server = None
        self.api_thread = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Enhanced Trading System initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    async def initialize(self) -> bool:
        """Initialize the trading system."""
        try:
            logger.info("Initializing Enhanced Trading System...")
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            logger.info("Database manager initialized")
            
            # Initialize trading executor
            self.trading_executor = EnhancedTradingExecutor(
                mt5_config=config.get_mt5_config() if config.is_platform_enabled('mt5') else None,
                ib_config=config.get_ib_config() if config.is_platform_enabled('ib') else None,
                ccxt_config=config.get_ccxt_config() if config.is_platform_enabled('ccxt') else None,
                trading_mode=config.get_trading_mode(),
                micro_lot_size=config.trading.micro_lot_size
            )
            
            # Initialize trading executor
            if not await self.trading_executor.initialize():
                logger.error("Failed to initialize trading executor")
                return False
            
            logger.info("Trading executor initialized successfully")
            
            # Start API server if enabled
            if config.api.enabled:
                await self._start_api_server()
            
            logger.info("Enhanced Trading System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            return False
    
    async def _start_api_server(self):
        """Start the API server."""
        try:
            def run_api():
                uvicorn.run(
                    app,
                    host=config.api.host,
                    port=config.api.port,
                    log_level=config.logging.level.lower(),
                    access_log=False
                )
            
            self.api_thread = threading.Thread(target=run_api, daemon=True)
            self.api_thread.start()
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            logger.info(f"API server started on {config.api.host}:{config.api.port}")
            
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
    
    async def run_trading_loop(self):
        """Main trading loop."""
        try:
            logger.info("Starting enhanced trading loop...")
            
            while self.is_running:
                try:
                    # Check if trading is active
                    if not self.trading_executor.is_trading_active:
                        logger.warning("Trading is disabled, waiting...")
                        await asyncio.sleep(60)
                        continue
                    
                    # Check kill switch
                    if self.trading_executor.is_kill_switch_active():
                        logger.warning("Kill switch is active, waiting...")
                        await asyncio.sleep(60)
                        continue
                    
                    # Monitor positions and risk
                    await self._monitor_positions()
                    await self._check_risk_limits()
                    
                    # Process trading signals (placeholder for AI integration)
                    await self._process_trading_signals()
                    
                    # Wait before next iteration
                    await asyncio.sleep(config.risk.position_monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Critical error in trading loop: {e}")
    
    async def _monitor_positions(self):
        """Monitor positions across all platforms."""
        try:
            # Get portfolio summary
            portfolio_summary = await self.trading_executor.get_portfolio_summary()
            
            # Log portfolio status
            logger.info(f"Portfolio Status: {portfolio_summary['total_positions']} positions, "
                       f"Total PnL: {portfolio_summary['total_pnl']:.2f}")
            
            # Check for position updates
            for platform, platform_data in portfolio_summary.get('platforms', {}).items():
                positions = platform_data.get('positions', 0)
                pnl = platform_data.get('pnl', 0)
                
                if positions > 0:
                    logger.debug(f"{platform.upper()}: {positions} positions, PnL: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    async def _check_risk_limits(self):
        """Check risk management limits."""
        try:
            # Get current portfolio
            portfolio_summary = await self.trading_executor.get_portfolio_summary()
            total_pnl = portfolio_summary.get('total_pnl', 0)
            
            # Check daily loss limit
            if total_pnl < -config.risk.max_daily_loss * 1000:  # Assuming $1000 account
                logger.critical(f"Daily loss limit exceeded: {total_pnl:.2f}")
                
                # Activate kill switch
                self.trading_executor.activate_kill_switch()
                
                # Log risk event
                self.db_manager.log_risk_event({
                    'event_type': 'daily_loss_limit_exceeded',
                    'severity': 'critical',
                    'message': f'Daily loss limit exceeded: {total_pnl:.2f}',
                    'metadata': {
                        'total_pnl': total_pnl,
                        'limit': config.risk.max_daily_loss,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                })
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    async def _process_trading_signals(self):
        """Process trading signals (placeholder for AI integration)."""
        try:
            # This is where you would integrate with your AI trading system
            # For now, we'll just log that we're ready to process signals
            
            if config.trading.enabled:
                logger.debug("Ready to process trading signals")
                
                # Example: Process signals for default symbols
                for symbol in config.trading.default_symbols:
                    # Here you would get AI signals and execute trades
                    # signal = await get_ai_signal(symbol)
                    # if signal:
                    #     await self.trading_executor.execute_trade(signal)
                    pass
            
        except Exception as e:
            logger.error(f"Error processing trading signals: {e}")
    
    async def execute_manual_trade(self, symbol: str, action: str, 
                                 confidence: float = 1.0, **kwargs) -> Dict[str, Any]:
        """Execute a manual trade."""
        try:
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                **kwargs
            )
            
            result = await self.trading_executor.execute_trade(signal)
            
            # Log trade execution
            if result.get('success'):
                logger.info(f"Manual trade executed: {action} {symbol}")
            else:
                logger.warning(f"Manual trade failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing manual trade: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_running': self.is_running,
                'trading_active': self.trading_executor.is_trading_active if self.trading_executor else False,
                'kill_switch_active': self.trading_executor.is_kill_switch_active() if self.trading_executor else False,
                'api_server_running': self.api_thread.is_alive() if self.api_thread else False,
                'enabled_platforms': config.get_enabled_platforms(),
                'trading_mode': config.trading.mode,
                'micro_lot_size': config.trading.micro_lot_size,
                'risk_limits': {
                    'max_daily_loss': config.risk.max_daily_loss,
                    'max_position_size': config.risk.max_position_size,
                    'risk_per_trade': config.risk.risk_per_trade
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def start(self):
        """Start the trading system."""
        try:
            if self.is_running:
                logger.warning("Trading system is already running")
                return
            
            logger.info("Starting Enhanced Trading System...")
            
            # Initialize system
            if not await self.initialize():
                logger.error("Failed to initialize trading system")
                return
            
            self.is_running = True
            
            # Start trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the trading system."""
        try:
            if not self.is_running:
                return
            
            logger.info("Stopping Enhanced Trading System...")
            
            self.is_running = False
            
            # Stop trading
            if self.trading_executor:
                await self.trading_executor.stop_trading()
                await self.trading_executor.cleanup()
            
            logger.info("Enhanced Trading System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")

async def main():
    """Main entry point."""
    try:
        # Create trading system
        trading_system = EnhancedTradingSystem()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'status':
                # Get system status
                status = trading_system.get_status()
                print(json.dumps(status, indent=2))
                return
            
            elif command == 'trade':
                # Execute manual trade
                if len(sys.argv) < 4:
                    print("Usage: python enhanced_main.py trade <symbol> <action> [confidence]")
                    return
                
                symbol = sys.argv[2]
                action = sys.argv[3]
                confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
                
                # Initialize system
                if not await trading_system.initialize():
                    print("Failed to initialize trading system")
                    return
                
                result = await trading_system.execute_manual_trade(symbol, action, confidence)
                print(json.dumps(result, indent=2))
                return
            
            elif command == 'kill-switch':
                # Toggle kill switch
                if not await trading_system.initialize():
                    print("Failed to initialize trading system")
                    return
                
                if trading_system.trading_executor.is_kill_switch_active():
                    trading_system.trading_executor.deactivate_kill_switch()
                    print("Kill switch deactivated")
                else:
                    trading_system.trading_executor.activate_kill_switch()
                    print("Kill switch activated")
                return
            
            elif command == 'start':
                # Start automated trading
                await trading_system.start()
                return
            
            else:
                print(f"Unknown command: {command}")
                print("Available commands: status, trade, kill-switch, start")
                return
        
        # Default: start automated trading
        await trading_system.start()
        
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        level=config.logging.level,
        format=config.logging.format
    )
    
    if config.logging.file:
        logger.add(
            config.logging.file,
            level=config.logging.level,
            format=config.logging.format,
            rotation=config.logging.max_size,
            retention=config.logging.backup_count
        )
    
    # Run main
    asyncio.run(main())