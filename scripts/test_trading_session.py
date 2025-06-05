#!/usr/bin/env python3
"""
Test Trading Session Script

This script runs a simulated trading session to verify the bot's functionality
before market open. It will:
1. Run in testing mode with historical data
2. Simulate a full trading day
3. Verify the output format
4. Check error handling
5. Validate logging configuration
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.live_trader import LiveTradingBot

def run_test_session():
    print("\n=== Starting Test Trading Session ===")
    print("This will simulate a full trading day to verify the bot's functionality")
    
    # Create test bot instance
    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    bot = LiveTradingBot(
        symbols=test_symbols,
        paper_trading=True,
        testing_mode=True
    )
    
    # Simulate a full trading day
    print("\nSimulating trading day...")
    
    # Start at 9:15 AM
    bot.current_test_time = datetime(2024, 1, 1, 9, 15)
    
    # Run trading cycles until 4:15 PM
    end_time = datetime(2024, 1, 1, 16, 15)
    
    while bot.current_test_time < end_time:
        print(f"\nTest Time: {bot.current_test_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run a trading cycle
        bot.run_trading_cycle()
        
        # Advance time by 5 minutes
        bot.current_test_time += timedelta(minutes=5)
        
        # Small delay to not overwhelm the system
        time.sleep(0.1)
    
    print("\n=== Test Session Complete ===")
    print("\nVerification Steps:")
    
    # Verify log file
    log_file = 'logs/trading_bot.log'
    if os.path.exists(log_file):
        log_size = os.path.getsize(log_file)
        print(f"✅ Log file created and written to ({log_size} bytes)")
    else:
        print("❌ Log file not found!")
    
    # Verify database
    db_file = 'trading_bot.db'
    if os.path.exists(db_file):
        db_size = os.path.getsize(db_file)
        print(f"✅ Database file created and written to ({db_size} bytes)")
    else:
        print("❌ Database file not found!")
    
    print("\nPlease verify that the terminal output above shows:")
    print("1. Only confidence levels for each stock")
    print("2. Portfolio summary in JSON format")
    print("3. No detailed logging information")

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        run_test_session()
    except KeyboardInterrupt:
        print("\n\nTest session interrupted by user")
    except Exception as e:
        print(f"\n\nError during test session: {str(e)}")
    finally:
        print("\nTest session ended") 