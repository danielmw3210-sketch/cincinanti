"""Main application file for the AI Crypto Trader."""

import asyncio
import signal
import sys
import time
from typing import Dict, Any
from datetime import datetime
from loguru import logger
from config import config
from trading_executor import TradingExecutor
from monitor import Monitor
from ai_trader import AITrader
from market_analyzer import MarketAnalyzer
from kraken_client import KrakenClient

class AICryptoTrader:
    """Main AI Crypto Trader application."""
    
    def __init__(self):
        self.trading_executor = TradingExecutor()
        self.monitor = Monitor()
        self.is_running = False
        self.trading_pairs = [config.default_pair]
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def initialize(self) -> bool:
        """Initialize the trading system."""
        try:
            logger.info("Initializing AI Crypto Trader...")
            
            # Test API connection
            server_time = self.trading_executor.client.get_server_time()
            if not server_time:
                logger.error("Failed to connect to Kraken API")
                return False
            
            logger.info("Successfully connected to Kraken API")
            
            # Initialize portfolio
            if not self.trading_executor.initialize_portfolio():
                logger.error("Failed to initialize portfolio")
                return False
            
            logger.info("Portfolio initialized successfully")
            
            # Start monitoring
            self.monitor.start_monitoring(self.trading_executor)
            
            logger.info("AI Crypto Trader initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            return False
    
    def run_trading_loop(self):
        """Main trading loop."""
        try:
            logger.info("Starting trading loop...")
            
            while self.is_running:
                try:
                    # Check if trading is active
                    if not self.trading_executor.is_trading_active:
                        logger.warning("Trading is disabled, waiting...")
                        time.sleep(60)
                        continue
                    
                    # Check stop losses and take profits
                    self.trading_executor.check_stop_losses_and_take_profits()
                    
                    # Process each trading pair
                    for pair in self.trading_pairs:
                        try:
                            # Get AI trading recommendation
                            recommendation = self.trading_executor.ai_trader.get_trading_recommendation(pair)
                            
                            # Log the recommendation
                            self.monitor.log_ai_signal(recommendation)
                            
                            # Execute trade if recommendation is strong enough
                            action = recommendation.get('recommendation', {}).get('action', 'hold')
                            confidence = recommendation.get('recommendation', {}).get('confidence', 0.0)
                            
                            if action in ['buy', 'sell'] and confidence > 0.6:
                                logger.info(f"Executing {action} signal for {pair} (confidence: {confidence:.3f})")
                                
                                trade_result = self.trading_executor.execute_trade(
                                    pair=pair,
                                    signal=action,
                                    confidence=confidence
                                )
                                
                                # Log trade execution
                                self.monitor.log_trade_execution(trade_result)
                                
                                if trade_result['success']:
                                    logger.info(f"Trade executed successfully: {trade_result}")
                                else:
                                    logger.warning(f"Trade execution failed: {trade_result}")
                            
                            # Wait between trades on same pair
                            time.sleep(30)
                            
                        except Exception as e:
                            logger.error(f"Error processing pair {pair}: {e}")
                            self.monitor.log_error(e, f"Processing pair {pair}")
                    
                    # Wait before next iteration
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    self.monitor.log_error(e, "Main trading loop")
                    time.sleep(60)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Critical error in trading loop: {e}")
            self.monitor.log_error(e, "Critical trading loop error")
    
    def run_single_analysis(self, pair: str) -> Dict[str, Any]:
        """Run a single analysis and return recommendation."""
        try:
            logger.info(f"Running single analysis for {pair}")
            
            # Get market analysis
            market_summary = self.trading_executor.market_analyzer.get_market_summary(pair)
            
            # Get AI recommendation
            recommendation = self.trading_executor.ai_trader.get_trading_recommendation(pair)
            
            # Get portfolio status
            portfolio_summary = self.trading_executor.get_portfolio_summary()
            
            result = {
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'market_analysis': market_summary,
                'ai_recommendation': recommendation,
                'portfolio_status': portfolio_summary,
                'risk_metrics': self.trading_executor.risk_manager.get_risk_metrics()
            }
            
            logger.info(f"Analysis completed for {pair}")
            return result
            
        except Exception as e:
            logger.error(f"Error running single analysis: {e}")
            return {'error': str(e)}
    
    def execute_manual_trade(self, pair: str, action: str, size: Optional[float] = None) -> Dict[str, Any]:
        """Execute a manual trade."""
        try:
            logger.info(f"Executing manual trade: {action} {size} {pair}")
            
            trade_result = self.trading_executor.execute_trade(
                pair=pair,
                signal=action,
                confidence=1.0,  # Manual trades have full confidence
                position_size=size
            )
            
            # Log trade execution
            self.monitor.log_trade_execution(trade_result)
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing manual trade: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            portfolio_summary = self.trading_executor.get_portfolio_summary()
            monitoring_status = self.monitor.get_monitoring_status()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_running': self.is_running,
                'trading_active': self.trading_executor.is_trading_active,
                'portfolio': portfolio_summary,
                'monitoring': monitoring_status,
                'trading_pairs': self.trading_pairs,
                'config': {
                    'default_pair': config.default_pair,
                    'risk_per_trade': config.risk_per_trade,
                    'max_position_size': config.max_position_size,
                    'max_daily_loss': config.max_daily_loss
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def start(self):
        """Start the trading system."""
        try:
            if self.is_running:
                logger.warning("Trading system is already running")
                return
            
            logger.info("Starting AI Crypto Trader...")
            
            # Initialize system
            if not self.initialize():
                logger.error("Failed to initialize trading system")
                return
            
            self.is_running = True
            
            # Start trading loop in a separate thread
            trading_thread = threading.Thread(target=self.run_trading_loop, daemon=True)
            trading_thread.start()
            
            logger.info("AI Crypto Trader started successfully")
            
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            self.stop()
    
    def stop(self):
        """Stop the trading system."""
        try:
            if not self.is_running:
                return
            
            logger.info("Stopping AI Crypto Trader...")
            
            self.is_running = False
            
            # Stop trading
            self.trading_executor.stop_trading()
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            logger.info("AI Crypto Trader stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")

def main():
    """Main entry point."""
    try:
        # Create trader instance
        trader = AICryptoTrader()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'analyze':
                # Run single analysis
                pair = sys.argv[2] if len(sys.argv) > 2 else config.default_pair
                result = trader.run_single_analysis(pair)
                print(json.dumps(result, indent=2))
                return
            
            elif command == 'status':
                # Get system status
                status = trader.get_status()
                print(json.dumps(status, indent=2))
                return
            
            elif command == 'trade':
                # Execute manual trade
                if len(sys.argv) < 4:
                    print("Usage: python main.py trade <pair> <action> [size]")
                    return
                
                pair = sys.argv[2]
                action = sys.argv[3]
                size = float(sys.argv[4]) if len(sys.argv) > 4 else None
                
                result = trader.execute_manual_trade(pair, action, size)
                print(json.dumps(result, indent=2))
                return
            
            elif command == 'start':
                # Start automated trading
                trader.start()
                
                # Keep running until interrupted
                try:
                    while trader.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    trader.stop()
                return
            
            else:
                print(f"Unknown command: {command}")
                print("Available commands: analyze, status, trade, start")
                return
        
        # Default: start automated trading
        trader.start()
        
        # Keep running until interrupted
        try:
            while trader.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            trader.stop()
    
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import json
    import threading
    main()