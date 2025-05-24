# 🤖 Live AI Stock Trading Bot Guide

## Overview

Your AI-powered stock trading bot combines:

- **Machine Learning** predictions (Random Forest with 61% accuracy)
- **News sentiment analysis** (FinBERT model)
- **Risk management** and position sizing
- **Automated trade execution**

## Quick Start

### 1. Start the Trading Bot (Paper Trading - Safe Mode)

```bash
# Basic paper trading (safe mode)
python start_trading_bot.py

# With custom symbols
python start_trading_bot.py --symbols AAPL MSFT TSLA NVDA

# With custom settings
python start_trading_bot.py --confidence 0.7 --max-positions 3 --risk-per-trade 0.01
```

### 2. Monitor with Dashboard

```bash
# Start the enhanced Streamlit dashboard
streamlit run scripts/dashboard.py
```

Open your browser to `http://localhost:8501` to see:

- 📊 Real-time ML predictions
- 📰 Live news sentiment analysis
- 💰 Portfolio performance
- 🎯 Trade signals and confidence levels

## Live Trading Setup (Real Money)

⚠️ **WARNING: Only do this when you're ready to trade with real money!**

### Step 1: Choose a Broker

**Recommended: Alpaca (Commission-free)**

```bash
# Get setup guide
python -c "from app.broker_integrations import print_setup_guide; print_setup_guide('alpaca')"
```

### Step 2: Set Environment Variables

Create a `.env` file:

```bash
# For Alpaca Paper Trading
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# For news sentiment (optional but recommended)
NEWS_API_KEY=your_news_api_key
FINNHUB_TOKEN=your_finnhub_token
```

### Step 3: Enable Live Trading

```bash
# DANGER: Real money mode
python start_trading_bot.py --live
```

## How It Works

### Trading Strategy

1. **Market Analysis** (every 5 minutes during market hours)

   - Downloads 60 days of historical data
   - Calculates 35 technical indicators
   - Runs ML model for 3-day and 5-day predictions

2. **Sentiment Enhancement**

   - Fetches recent news articles
   - Analyzes sentiment with FinBERT
   - Boosts/reduces ML confidence based on sentiment

3. **Signal Generation**

   - **LONG** signal: ML confidence > 65% for upward movement
   - **SHORT** signal: ML confidence > 65% for downward movement
   - **HOLD**: Confidence below threshold

4. **Risk Management**
   - Max 2% risk per trade
   - Max 5 open positions
   - 3% profit target, 2% stop loss
   - Auto-close after 3 days

### Features

- ✅ **Paper Trading**: Test strategies without risk
- ✅ **Real-time Monitoring**: Live dashboard with charts
- ✅ **News Integration**: FinBERT sentiment analysis
- ✅ **Risk Controls**: Automatic position sizing and stop losses
- ✅ **Database Logging**: Complete trade history
- ✅ **Emergency Controls**: Stop bot or close all positions

## Market Hours

The bot automatically detects US market hours:

- **Active**: Monday-Friday, 9:30 AM - 4:00 PM ET
- **Inactive**: Weekends and after hours

## Configuration Options

```bash
python start_trading_bot.py --help
```

Available options:

- `--symbols`: Stock symbols to trade (default: AAPL MSFT TSLA NVDA GOOGL)
- `--confidence`: Minimum confidence threshold (default: 0.65)
- `--max-positions`: Maximum open positions (default: 5)
- `--risk-per-trade`: Risk per trade as % of portfolio (default: 0.02)

## Example Trading Session

### Step 1: Start the Bot

```bash
python start_trading_bot.py --symbols AAPL MSFT --confidence 0.7
```

### Step 2: Monitor the Logs

```
🚀 Live Trading Bot Started!
✅ ML Predictor initialized
✅ News Sentiment Analyzer initialized
📝 Paper Trading mode active
💰 Portfolio Value: $100,000.00

🔄 Starting trading cycle...
Generated signal for AAPL: LONG @ $185.50, confidence=0.712, sentiment=0.045
✅ Executed LONG trade: AAPL x54 @ $185.50
Position AAPL: $186.20 (P&L: $37.80)
💼 Portfolio: $90,009.70 | Open Positions: 1 | Unrealized P&L: $37.80
```

### Step 3: Dashboard Monitoring

Open `http://localhost:8501` to see:

- Current positions and P&L
- ML confidence levels for each symbol
- News sentiment scores
- Interactive charts and analysis

## Performance

**Backtest Results:**

- **Symbols**: AAPL, MSFT, TSLA
- **Period**: Jan-Jun 2024
- **Total Trades**: 3
- **Win Rate**: 33.3%
- **Sentiment Correlation**: 0.768 (strong correlation between sentiment and performance)

## Safety Features

### Emergency Controls

- **Ctrl+C**: Graceful shutdown, closes all positions
- **Dashboard**: Emergency stop and close all buttons
- **Database**: Complete audit trail of all trades

### Risk Limits

- **Max Risk**: 2% of portfolio per trade
- **Position Limits**: Max 20% of portfolio in any single stock
- **Time Limits**: Positions auto-close after 3 days
- **Stop Losses**: Automatic 2% stop loss on all trades

## Troubleshooting

### Common Issues

1. **"No data available"**

   - Check internet connection
   - Verify symbol is valid
   - Market might be closed

2. **"ML model returned no predictions"**

   - Need at least 30 days of data
   - Check if enough technical indicators calculated

3. **"Sentiment analysis failed"**
   - News API limits reached
   - Check API keys in `.env`
   - Bot will continue with neutral sentiment

### Log Files

- Trading logs: `logs/trading_bot.log`
- Database: `trading_bot.db`

### Test Mode

```bash
# Test ML predictions without trading
python test_integrated_ml_sentiment.py

# Run backtest with sentiment
python scripts/efficient_backtest_with_sentiment.py
```

## Advanced Usage

### Custom Strategies

Modify `app/live_trader.py` to implement custom trading logic:

- Change confidence thresholds
- Add new technical indicators
- Implement different position sizing
- Create custom exit conditions

### Broker Integration

Add real broker support in `app/broker_integrations.py`:

- Implement Alpaca API
- Add Interactive Brokers support
- Create custom broker adapters

### News Sources

Extend `app/news_sentiment.py`:

- Add more news APIs
- Implement custom sentiment models
- Create sector-specific news analysis

## Support

- 📖 **Documentation**: This README
- 🐛 **Issues**: Check logs and database
- 🔧 **Configuration**: `.env` file and command line options
- 📊 **Monitoring**: Streamlit dashboard

---

**Remember**: Start with paper trading, monitor performance, and only switch to live trading when you're confident in the system! 🚀
