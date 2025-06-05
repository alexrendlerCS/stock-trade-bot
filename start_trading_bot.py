#!/usr/bin/env python3
"""
Live Trading Bot Starter Script

This script starts your AI-powered stock trading bot that combines:
- Machine Learning predictions (Random Forest with 61% accuracy)
- Real-time news sentiment analysis (FinBERT)
- Risk management and position sizing
- Automated trade execution

Usage:
    python start_trading_bot.py                    # Paper trading mode (safe)
    python start_trading_bot.py --paper            # Paper trading mode (explicit)
    python start_trading_bot.py --live             # LIVE trading mode (real money!)
    python start_trading_bot.py --symbols AAPL MSFT TSLA  # Custom symbols
"""

import argparse
import sys
from app.live_trader import start_live_trading
import logging

def main():
    parser = argparse.ArgumentParser(description='Start the AI Stock Trading Bot')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--testing', action='store_true', help='Run in testing mode with historical data')
    parser.add_argument('--symbols', nargs='+', help='List of stock symbols to trade',
                      default=['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'META', 'COIN', 'BTC'])
    
    args = parser.parse_args()
    
    print("\n============================================================")
    print("🤖 AI STOCK TRADING BOT")
    print("============================================================")
    if args.testing:
        print("Mode: 🔬 Testing Mode (Historical Data)")
    else:
        print("Mode: 📝 Paper Trading" if args.paper else "Mode: 💰 Live Trading")
    print(f"Symbols: {', '.join(args.symbols)}")
    print("Confidence Threshold: 65.0%")
    print("Max Positions: 8")
    print("Risk Per Trade: 2.0%")
    print("============================================================\n")
    
    print("🚀 Starting Trading Bot...")
    print("💡 Features active:")
    print("   ✅ ML Predictions (Random Forest - 61% accuracy)")
    print("   ✅ News Sentiment Analysis (FinBERT)")
    print("   ✅ Risk Management (2% max risk per trade)")
    print("   ✅ Automated Position Management")
    print("   ✅ Real-time Market Monitoring")
    
    start_live_trading(symbols=args.symbols, paper_trading=args.paper, testing_mode=args.testing)

if __name__ == "__main__":
    main() 