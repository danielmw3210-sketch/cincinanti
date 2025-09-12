"""Test script for the AI Crypto Trader."""

import unittest
import os
import sys
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kraken_client import KrakenClient
from market_analyzer import MarketAnalyzer
from ai_trader import AITrader
from risk_manager import RiskManager
from trading_executor import TradingExecutor
from config import config

class TestKrakenClient(unittest.TestCase):
    """Test Kraken API client."""
    
    def setUp(self):
        self.client = KrakenClient()
    
    @patch('requests.get')
    def test_get_server_time(self, mock_get):
        """Test getting server time."""
        mock_response = Mock()
        mock_response.json.return_value = {'result': {'unixtime': 1234567890}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_server_time()
        self.assertIn('unixtime', result)
    
    @patch('requests.get')
    def test_get_ticker(self, mock_get):
        """Test getting ticker data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'result': {
                'XXBTZUSD': {
                    'c': ['50000.0', '0.1'],
                    'v': ['100.0', '200.0']
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_ticker('BTC/USD')
        self.assertIn('XXBTZUSD', result)

class TestMarketAnalyzer(unittest.TestCase):
    """Test market analysis functionality."""
    
    def setUp(self):
        self.client = Mock()
        self.analyzer = MarketAnalyzer(self.client)
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculations."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.rand(100) * 50,
            'low': prices - np.random.rand(100) * 50,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        result = self.analyzer.calculate_technical_indicators(df)
        
        # Check that indicators were added
        self.assertIn('sma_20', result.columns)
        self.assertIn('rsi', result.columns)
        self.assertIn('macd', result.columns)
        self.assertIn('bb_upper', result.columns)
    
    def test_detect_patterns(self):
        """Test pattern detection."""
        # Create sample data with a hammer pattern
        dates = pd.date_range('2023-01-01', periods=5, freq='1H')
        df = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50000],
            'high': [50100, 50200, 50300, 50400, 50100],
            'low': [49900, 50000, 50100, 50200, 49800],
            'close': [50050, 50150, 50250, 50350, 50050],
            'volume': [1000, 1200, 1100, 1300, 1500]
        }, index=dates)
        
        patterns = self.analyzer.detect_patterns(df)
        self.assertIsInstance(patterns, dict)

class TestRiskManager(unittest.TestCase):
    """Test risk management functionality."""
    
    def setUp(self):
        self.risk_manager = RiskManager()
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        result = self.risk_manager.calculate_position_size(
            signal_confidence=0.8,
            current_price=50000,
            stop_loss_price=49000,
            account_balance=10000,
            pair='BTC/USD'
        )
        
        self.assertIn('size', result)
        self.assertGreater(result['size'], 0)
    
    def test_calculate_stop_loss(self):
        """Test stop loss calculation."""
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=50000,
            signal_type='buy',
            volatility=0.02
        )
        
        self.assertLess(stop_loss, 50000)  # Should be below entry for buy
    
    def test_validate_trade(self):
        """Test trade validation."""
        validation = self.risk_manager.validate_trade(
            pair='BTC/USD',
            signal_type='buy',
            position_size=0.01,
            entry_price=50000,
            account_balance=10000
        )
        
        self.assertIn('valid', validation)
        self.assertIsInstance(validation['valid'], bool)

class TestAITrader(unittest.TestCase):
    """Test AI trading functionality."""
    
    def setUp(self):
        self.market_analyzer = Mock()
        self.ai_trader = AITrader(self.market_analyzer)
    
    def test_create_labels(self):
        """Test label creation for ML training."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'close': prices
        }, index=dates)
        
        labels = self.ai_trader.create_labels(df)
        
        self.assertEqual(len(labels), len(df))
        self.assertTrue(all(label in [-1, 0, 1] for label in labels.dropna()))
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        # Mock the analyzer to return sample data
        sample_data = pd.DataFrame({
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.randint(1000, 10000, 100),
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50000,
            'low': np.random.randn(100) * 100 + 50000
        })
        
        self.market_analyzer.prepare_features.return_value = sample_data
        
        X, y = self.ai_trader.prepare_training_data(sample_data)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

class TestTradingExecutor(unittest.TestCase):
    """Test trading execution functionality."""
    
    def setUp(self):
        self.executor = TradingExecutor()
    
    def test_initialize_portfolio(self):
        """Test portfolio initialization."""
        # Mock the client
        self.executor.client.get_account_balance = Mock(return_value={
            'ZUSD': '1000.0',
            'XXBT': '0.1'
        })
        
        result = self.executor.initialize_portfolio()
        self.assertTrue(result)
        self.assertIn('total_balance', self.executor.portfolio)
    
    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        # Initialize portfolio first
        self.executor.portfolio = {
            'total_balance': 10000,
            'available_balance': 9000,
            'positions': {},
            'cash': 10000
        }
        
        summary = self.executor.get_portfolio_summary()
        
        self.assertIn('portfolio', summary)
        self.assertIn('risk_metrics', summary)
        self.assertIn('performance', summary)

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test the complete trading workflow."""
        # This would test the full integration
        # For now, just verify components can be instantiated
        client = KrakenClient()
        analyzer = MarketAnalyzer(client)
        ai_trader = AITrader(analyzer)
        risk_manager = RiskManager()
        executor = TradingExecutor()
        
        # Verify all components are created
        self.assertIsNotNone(client)
        self.assertIsNotNone(analyzer)
        self.assertIsNotNone(ai_trader)
        self.assertIsNotNone(risk_manager)
        self.assertIsNotNone(executor)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestKrakenClient,
        TestMarketAnalyzer,
        TestRiskManager,
        TestAITrader,
        TestTradingExecutor,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🧪 Running AI Crypto Trader Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)