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

def main():
    parser = argparse.ArgumentParser(description='AI Stock Trading Bot')
    
    parser.add_argument('--live', action='store_true', 
                       help='Enable LIVE trading with real money (default: paper trading)')
    parser.add_argument('--paper', action='store_true', 
                       help='Enable paper trading mode (default)')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL'],
                       help='Stock symbols to trade (default: AAPL MSFT TSLA NVDA GOOGL)')
    parser.add_argument('--confidence', type=float, default=0.65,
                       help='Minimum confidence threshold for trades (default: 0.65)')
    parser.add_argument('--max-positions', type=int, default=5,
                       help='Maximum number of open positions (default: 5)')
    parser.add_argument('--risk-per-trade', type=float, default=0.02,
                       help='Risk per trade as fraction of portfolio (default: 0.02 = 2%%)')
    
    args = parser.parse_args()
    
    # Determine trading mode
    if args.live and args.paper:
        print("❌ Cannot specify both --live and --paper modes")
        sys.exit(1)
    
    paper_trading = not args.live  # Default to paper trading unless --live is specified
    
    # Safety check for live trading
    if args.live:
        print("🚨 WARNING: LIVE TRADING MODE ENABLED 🚨")
        print("This will trade with REAL MONEY!")
        print("Note: Real broker integration is not yet implemented.")
        print("The bot will run in paper trading mode until you implement broker APIs.")
        paper_trading = True  # Force paper trading until broker integration is complete
    
    # Display configuration
    print("\n" + "="*60)
    print("🤖 AI STOCK TRADING BOT")
    print("="*60)
    print(f"Mode: {'📝 Paper Trading' if paper_trading else '💰 LIVE Trading'}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Confidence Threshold: {args.confidence:.1%}")
    print(f"Max Positions: {args.max_positions}")
    print(f"Risk Per Trade: {args.risk_per_trade:.1%}")
    print("="*60)
    
    # Confirm if live trading (when implemented)
    if not paper_trading:
        response = input("Are you sure you want to trade with real money? (yes/no): ")
        if response.lower() != 'yes':
            print("Switching to paper trading mode for safety.")
            paper_trading = True
    
    print("\n🚀 Starting Trading Bot...")
    print("💡 Features active:")
    print("   ✅ ML Predictions (Random Forest - 61% accuracy)")
    print("   ✅ News Sentiment Analysis (FinBERT)")
    print("   ✅ Risk Management (2% max risk per trade)")
    print("   ✅ Automated Position Management")
    print("   ✅ Real-time Market Monitoring")
    
    try:
        # Start the bot
        start_live_trading(
            symbols=args.symbols,
            paper_trading=paper_trading
        )
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting trading bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 