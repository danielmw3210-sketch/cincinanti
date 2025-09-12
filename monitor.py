"""Monitoring, logging, and alerting system."""

import json
import requests
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import schedule
import time
import threading
from loguru import logger
from config import config

class Monitor:
    """Handles monitoring, logging, and alerting."""
    
    def __init__(self):
        self.webhook_url = config.webhook_url
        self.alert_email = config.alert_email
        self.log_level = config.log_level
        
        # Alert thresholds
        self.alert_thresholds = {
            'daily_loss_percent': 0.03,  # 3% daily loss
            'drawdown_percent': 0.10,    # 10% drawdown
            'consecutive_losses': 5,     # 5 consecutive losses
            'api_errors': 10,           # 10 API errors
            'model_accuracy': 0.4       # Model accuracy below 40%
        }
        
        # Monitoring state
        self.alerts_sent = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            # Remove default handler
            logger.remove()
            
            # Add console handler
            logger.add(
                "logs/trading_{time}.log",
                rotation="1 day",
                retention="30 days",
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
            )
            
            # Add console handler for important messages
            logger.add(
                lambda msg: print(msg, end=""),
                level="INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
            )
            
            logger.info("Logging system initialized")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def start_monitoring(self, trading_executor):
        """Start monitoring thread."""
        try:
            self.trading_executor = trading_executor
            self.monitoring_active = True
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            # Schedule periodic checks
            schedule.every(1).minutes.do(self._check_portfolio_health)
            schedule.every(5).minutes.do(self._check_api_health)
            schedule.every(15).minutes.do(self._check_model_performance)
            schedule.every(1).hours.do(self._generate_performance_report)
            
            logger.info("Monitoring system started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring system."""
        try:
            self.monitoring_active = False
            schedule.clear()
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            logger.info("Monitoring system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _check_portfolio_health(self):
        """Check portfolio health and send alerts if needed."""
        try:
            if not hasattr(self, 'trading_executor'):
                return
            
            portfolio_summary = self.trading_executor.get_portfolio_summary()
            risk_metrics = portfolio_summary.get('risk_metrics', {})
            performance = portfolio_summary.get('performance', {})
            
            # Check daily loss
            daily_loss = risk_metrics.get('daily_pnl', 0)
            daily_loss_percent = abs(daily_loss) / config.initial_balance
            
            if daily_loss_percent > self.alert_thresholds['daily_loss_percent']:
                self._send_alert(
                    'HIGH_DAILY_LOSS',
                    f"Daily loss of {daily_loss_percent:.2%} exceeds threshold",
                    {'daily_loss': daily_loss, 'daily_loss_percent': daily_loss_percent}
                )
            
            # Check drawdown
            current_drawdown = risk_metrics.get('current_drawdown', 0)
            if current_drawdown > self.alert_thresholds['drawdown_percent']:
                self._send_alert(
                    'HIGH_DRAWDOWN',
                    f"Current drawdown of {current_drawdown:.2%} exceeds threshold",
                    {'current_drawdown': current_drawdown}
                )
            
            # Check consecutive losses
            recent_trades = portfolio_summary.get('recent_trades', [])
            consecutive_losses = self._count_consecutive_losses(recent_trades)
            
            if consecutive_losses >= self.alert_thresholds['consecutive_losses']:
                self._send_alert(
                    'CONSECUTIVE_LOSSES',
                    f"{consecutive_losses} consecutive losses detected",
                    {'consecutive_losses': consecutive_losses}
                )
            
        except Exception as e:
            logger.error(f"Error checking portfolio health: {e}")
    
    def _check_api_health(self):
        """Check API health and connectivity."""
        try:
            if not hasattr(self, 'trading_executor'):
                return
            
            # Test API connection
            server_time = self.trading_executor.client.get_server_time()
            
            if not server_time:
                self._send_alert(
                    'API_ERROR',
                    "Failed to connect to Kraken API",
                    {'error': 'API connection failed'}
                )
            
        except Exception as e:
            logger.error(f"Error checking API health: {e}")
            self._send_alert(
                'API_ERROR',
                f"API health check failed: {str(e)}",
                {'error': str(e)}
            )
    
    def _check_model_performance(self):
        """Check AI model performance."""
        try:
            if not hasattr(self, 'trading_executor'):
                return
            
            # This would need to be implemented based on actual model performance tracking
            # For now, we'll just log that the check was performed
            logger.debug("Model performance check completed")
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    def _generate_performance_report(self):
        """Generate and log performance report."""
        try:
            if not hasattr(self, 'trading_executor'):
                return
            
            portfolio_summary = self.trading_executor.get_portfolio_summary()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_summary.get('portfolio', {}).get('total_balance', 0),
                'total_return': portfolio_summary.get('performance', {}).get('total_return', 0),
                'daily_pnl': portfolio_summary.get('performance', {}).get('daily_pnl', 0),
                'open_positions': len(portfolio_summary.get('positions', {})),
                'total_trades': portfolio_summary.get('total_trades', 0),
                'risk_metrics': portfolio_summary.get('risk_metrics', {})
            }
            
            logger.info(f"Performance Report: {json.dumps(report, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
    
    def _count_consecutive_losses(self, trades: List[Dict]) -> int:
        """Count consecutive losing trades."""
        try:
            consecutive_losses = 0
            
            # Go through trades in reverse order (most recent first)
            for trade in reversed(trades):
                # This would need to be implemented based on actual trade P&L tracking
                # For now, return 0
                break
            
            return consecutive_losses
            
        except Exception as e:
            logger.error(f"Error counting consecutive losses: {e}")
            return 0
    
    def _send_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """Send alert via webhook and/or email."""
        try:
            # Check if we've already sent this type of alert recently
            alert_key = f"{alert_type}_{datetime.now().strftime('%Y%m%d%H')}"
            if alert_key in self.alerts_sent:
                return  # Don't spam alerts
            
            self.alerts_sent[alert_key] = True
            
            # Clean up old alert keys
            current_hour = datetime.now().strftime('%Y%m%d%H')
            self.alerts_sent = {k: v for k, v in self.alerts_sent.items() if k.endswith(current_hour)}
            
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': alert_type,
                'message': message,
                'data': data,
                'severity': self._get_alert_severity(alert_type)
            }
            
            # Send webhook alert
            if self.webhook_url:
                self._send_webhook_alert(alert_data)
            
            # Send email alert
            if self.alert_email:
                self._send_email_alert(alert_data)
            
            # Log alert
            logger.warning(f"ALERT [{alert_type}]: {message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level."""
        high_severity = ['HIGH_DAILY_LOSS', 'HIGH_DRAWDOWN', 'API_ERROR', 'EMERGENCY_STOP']
        
        if alert_type in high_severity:
            return 'HIGH'
        elif alert_type in ['CONSECUTIVE_LOSSES', 'MODEL_PERFORMANCE']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send alert via webhook."""
        try:
            response = requests.post(
                self.webhook_url,
                json=alert_data,
                timeout=10
            )
            response.raise_for_status()
            logger.debug("Webhook alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email."""
        try:
            # This is a placeholder - you would need to configure SMTP settings
            logger.info(f"Email alert would be sent: {alert_data['message']}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def log_trade_execution(self, trade_result: Dict[str, Any]):
        """Log trade execution details."""
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'trade_result': trade_result,
                'portfolio_snapshot': self.trading_executor.get_portfolio_summary() if hasattr(self, 'trading_executor') else {}
            }
            
            logger.info(f"Trade Execution: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging trade execution: {e}")
    
    def log_ai_signal(self, signal_data: Dict[str, Any]):
        """Log AI signal generation."""
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'signal_data': signal_data
            }
            
            logger.info(f"AI Signal: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging AI signal: {e}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        try:
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context
            }
            
            logger.error(f"Error Log: {json.dumps(error_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'alerts_sent_today': len(self.alerts_sent),
                'alert_thresholds': self.alert_thresholds,
                'webhook_configured': bool(self.webhook_url),
                'email_configured': bool(self.alert_email),
                'log_level': self.log_level
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {}