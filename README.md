# 🤖 AI Stock Trading Bot

An intelligent trading bot that combines machine learning predictions, news sentiment analysis, and automated trading strategies to make data-driven investment decisions.

## Features

- **Machine Learning Predictions**: Random Forest models with 61% accuracy for 3-day and 5-day price movements
- **News Sentiment Analysis**: Real-time sentiment analysis using FinBERT
- **Risk Management**: Automated position sizing and stop-loss implementation
- **Paper Trading**: Test strategies without financial risk
- **Real-time Monitoring**: Live dashboard with performance metrics
- **Database Logging**: Complete trade history and portfolio tracking

## 📁 Project Structure

```
stock-trade-bot/
├── app/                      # Core application code
│   ├── core/                # Core configurations and settings
│   ├── live_trader.py       # Main trading logic
│   ├── ml_predictor.py      # ML prediction engine
│   └── news_sentiment.py    # News sentiment analyzer
│
├── data/                    # Data directory
│   ├── features/           # Feature files for each stock
│   ├── models/            # ML models and configurations
│   └── archive/           # Archived/old files
│
├── tests/                   # Test suite
│   ├── integration/        # Integration tests
│   ├── unit/              # Unit tests
│   └── performance/       # Performance and backtest scripts
│
├── scripts/                 # Utility scripts
│   └── dashboard.py        # Streamlit dashboard
│
├── logs/                    # Log files
│   └── trading_bot.log     # Main log file
│
└── trading_bot.db          # SQLite database for trade history
```

## Quick Start

### Prerequisites

1. Python 3.8 or higher
2. pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-trade-bot.git
cd stock-trade-bot
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# Create .env file with your API keys
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
NEWS_API_KEY=your_news_api_key
```

### Running the Bot

1. Start in paper trading mode (safe):

```bash
python start_trading_bot.py
```

2. Launch the monitoring dashboard:

```bash
streamlit run scripts/dashboard.py
```

## Trading Strategy

The bot employs a multi-factor approach:

1. **Market Analysis** (5-minute intervals)

   - Technical indicators calculation
   - ML predictions for 3-day and 5-day movements
   - News sentiment analysis

2. **Signal Generation**

   - LONG: ML confidence > 65% for upward movement
   - SHORT: ML confidence > 65% for downward movement
   - Sentiment score adjustment

3. **Risk Management**
   - 2% maximum risk per trade
   - 5 maximum open positions
   - Automated stop-loss and take-profit

## Configuration

Command line options:

```bash
python start_trading_bot.py --help
```

Available parameters:

- `--symbols`: Stocks to trade (default: AAPL MSFT TSLA NVDA GOOGL)
- `--confidence`: Minimum confidence threshold (default: 0.65)
- `--max-positions`: Maximum open positions (default: 5)
- `--risk-per-trade`: Risk per trade as % of portfolio (default: 0.02)

## Performance Monitoring

1. **Dashboard** (`http://localhost:8501`):

   - Real-time portfolio value
   - Open positions and P&L
   - ML confidence levels
   - News sentiment scores

2. **Logging**:
   - Trading logs: `logs/trading_bot.log`
   - Trade history: `trading_bot.db`

## Safety Features

- Paper trading mode by default
- Automatic stop-loss implementation
- Position size limits
- Emergency shutdown (Ctrl+C)
- Complete trade audit trail

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/unit/
python -m pytest tests/integration/
```

### Adding New Features

1. **Custom Strategies**:

   - Modify `app/live_trader.py`
   - Add new technical indicators
   - Implement custom entry/exit conditions

2. **ML Enhancements**:
   - Update models in `app/ml_predictor.py`
   - Add new features to feature generation
   - Implement new prediction algorithms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Disclaimer

This software is for educational purposes only. Use at your own risk. The creators are not responsible for any financial losses incurred through the use of this trading bot.
