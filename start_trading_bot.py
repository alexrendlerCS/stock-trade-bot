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
import logging
import os
from app.live_trader import start_live_trading

def setup_logging():
    """Set up logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point for the trading bot"""
    parser = argparse.ArgumentParser(description='Start the trading bot')
    
    # Add command line arguments
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--symbols', type=str, nargs='+', 
                       default=['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL'],
                       help='List of symbols to trade')
    parser.add_argument('--max-positions', type=int, default=5,
                       help='Maximum number of open positions')
    parser.add_argument('--risk-per-trade', type=float, default=0.02,
                       help='Risk per trade as decimal (e.g., 0.02 for 2%%)')
    parser.add_argument('--confidence-threshold', type=float, default=0.65,
                       help='Minimum confidence threshold for trades')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Log startup configuration
    logging.info("Starting trading bot with configuration:")
    logging.info(f"Paper Trading: {args.paper}")
    logging.info(f"Symbols: {args.symbols}")
    logging.info(f"Max Positions: {args.max_positions}")
    logging.info(f"Risk Per Trade: {args.risk_per_trade*100}%")
    logging.info(f"Confidence Threshold: {args.confidence_threshold}")
    
    try:
        # Start the trading bot
        start_live_trading(
            symbols=args.symbols,
            paper_trading=args.paper,
            max_positions=args.max_positions,
            risk_per_trade=args.risk_per_trade,
            confidence_threshold=args.confidence_threshold
        )
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Error running bot: {str(e)}")

if __name__ == "__main__":
    main() 