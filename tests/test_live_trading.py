import unittest
import sys
import os
import json
from datetime import datetime, timedelta
import logging
from io import StringIO
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.live_trader import LiveTradingBot, start_live_trading

def create_mock_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Create mock stock data for testing"""
    dates = pd.date_range(end=datetime(2024, 1, 1), periods=days)
    data = {
        'open': np.random.uniform(100, 150, days),
        'high': np.random.uniform(120, 170, days),
        'low': np.random.uniform(90, 130, days),
        'close': np.random.uniform(100, 160, days),
        'volume': np.random.uniform(1000000, 5000000, days)
    }
    return pd.DataFrame(data, index=dates)

class TestLiveTrading(unittest.TestCase):
    def setUp(self):
        # Capture stdout to test terminal output
        self.stdout = StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self.stdout
        
        # Create test bot instance
        self.test_symbols = ['AAPL', 'MSFT']
        self.bot = LiveTradingBot(
            symbols=self.test_symbols,
            paper_trading=True,
            testing_mode=True,
            max_positions=2,
            risk_per_trade=0.02,
            confidence_threshold=0.65
        )
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
    
    def tearDown(self):
        # Restore stdout
        sys.stdout = self._original_stdout
        
        # Clean up test database
        if os.path.exists('trading_bot.db'):
            try:
                os.remove('trading_bot.db')
            except:
                pass
    
    def test_initialization(self):
        """Test bot initialization"""
        self.assertEqual(self.bot.symbols, self.test_symbols)
        self.assertTrue(self.bot.paper_trading)
        self.assertTrue(self.bot.testing_mode)
        self.assertEqual(self.bot.max_positions, 2)
        self.assertEqual(self.bot.risk_per_trade, 0.02)
        self.assertEqual(self.bot.confidence_threshold, 0.65)
    
    def test_market_hours(self):
        """Test market hours checking in testing mode"""
        # Test market open (10:30 AM on a Monday)
        self.bot.current_test_time = datetime(2024, 1, 1, 10, 30)  # Monday
        self.assertTrue(self.bot.is_market_open())
        
        # Test market closed (Sunday)
        self.bot.current_test_time = datetime(2024, 1, 7, 10, 30)  # Sunday
        self.assertFalse(self.bot.is_market_open())
        
        # Test pre-market
        self.bot.current_test_time = datetime(2024, 1, 1, 8, 30)
        self.assertFalse(self.bot.is_market_open())
        
        # Test after-market
        self.bot.current_test_time = datetime(2024, 1, 1, 16, 30)
        self.assertFalse(self.bot.is_market_open())
    
    @patch('yfinance.Ticker')
    def test_data_fetching(self, mock_ticker):
        """Test historical data fetching"""
        # Setup mock
        for symbol in self.test_symbols:
            mock_instance = MagicMock()
            mock_instance.history.return_value = create_mock_data(symbol)
            mock_ticker.return_value = mock_instance
            
            # Test data fetching
            data = self.bot.get_live_data(symbol)
            self.assertIsNotNone(data)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            self.assertTrue(all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']))
    
    @patch('yfinance.Ticker')
    def test_trading_cycle_output(self, mock_ticker):
        """Test that trading cycle produces correct terminal output format"""
        # Setup mock with data that will generate signals
        for symbol in self.test_symbols:
            mock_instance = MagicMock()
            mock_data = create_mock_data(symbol)
            # Make the data look bullish to generate signals
            mock_data['close'] = mock_data['close'].sort_values()
            mock_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_instance
        
        # Run a trading cycle
        self.bot.current_test_time = datetime(2024, 1, 1, 10, 30)
        self.bot.run_trading_cycle()
        
        output = self.stdout.getvalue()
        
        # Check if confidence levels are printed
        for symbol in self.test_symbols:
            self.assertIn(f"{symbol} Confidence:", output)
        
        # Check if portfolio summary is printed in correct format
        try:
            # Find the last JSON object in the output
            json_start = output.rfind('{')
            json_end = output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                portfolio_json = json.loads(output[json_start:json_end])
                
                # Verify portfolio summary structure
                self.assertIn('portfolio_value', portfolio_json)
                self.assertIn('open_positions', portfolio_json)
                self.assertIn('unrealized_pnl', portfolio_json)
                self.assertIn('total_value', portfolio_json)
                self.assertIn('positions', portfolio_json)
                self.assertIsInstance(portfolio_json['positions'], dict)
        except json.JSONDecodeError:
            self.fail("Portfolio summary is not in valid JSON format")
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test invalid symbol
        self.bot.symbols = ['INVALID_SYMBOL'] + self.test_symbols
        self.bot.run_trading_cycle()
        # Should not crash and should continue with valid symbols
        
        # Test API error handling
        self.bot.api = None  # Simulate API connection failure
        self.bot.run_trading_cycle()
        # Should handle the error gracefully
        
        # Test database error handling
        self.bot.db_path = '/invalid/path/trading_bot.db'
        self.bot.run_trading_cycle()
        # Should handle the error gracefully
    
    def test_logging_configuration(self):
        """Test that logging is properly configured"""
        # Verify log file exists and is being written to
        log_file = 'logs/trading_bot.log'
        self.bot.run_trading_cycle()
        
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            log_content = f.read()
            
        # Check that detailed logs are in the file
        self.assertIn("LiveTradingBot initialized", log_content)
        self.assertIn("Testing Mode", log_content)
        
        # Check that detailed logs are not in stdout
        stdout_content = self.stdout.getvalue()
        self.assertNotIn("LiveTradingBot initialized", stdout_content)
        self.assertNotIn("Testing Mode", stdout_content)

if __name__ == '__main__':
    unittest.main() 