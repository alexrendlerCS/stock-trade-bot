import pandas as pd
import yfinance as yf
import time
import logging
from datetime import datetime, timedelta
from datetime import time as datetime_time
from typing import Dict, List, Optional
import schedule
import json
from dataclasses import dataclass
from .ml_predictor import MLPredictor
import sqlite3
import os
import pytz
from .core.config import settings
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST
import traceback
from sklearn.dummy import DummyClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log')  # Only log to file
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    sentiment_score: float
    ml_probability: float
    timestamp: datetime

@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    quantity: int
    entry_time: datetime
    target_price: float
    stop_loss: float
    current_price: float
    unrealized_pnl: float
    sentiment_score: float

class LiveTradingBot:
    """
    Live trading bot that uses ML predictions and news sentiment to make trades
    """
    
    def __init__(self, 
                 symbols: List[str] = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL'],
                 paper_trading: bool = True,
                 max_positions: int = 5,
                 risk_per_trade: float = 0.02,  # 2% risk per trade
                 confidence_threshold: float = 0.65,
                 testing_mode: bool = False):  # Added testing mode parameter
        
        self.symbols = symbols
        self.paper_trading = paper_trading
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.confidence_threshold = confidence_threshold
        self.testing_mode = testing_mode
        
        # Initialize Alpaca API
        self.api = REST(
            key_id=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL
        )
        
        # Get initial account info
        account = self.api.get_account()
        self.portfolio_value = float(account.portfolio_value)
        
        # Initialize components
        self.ml_predictor = MLPredictor()
        self.positions = {}
        
        # For testing mode
        self.current_test_time = datetime.now() if not testing_mode else datetime(2024, 1, 1, 10, 0)
        
        # Database for trade history
        self._init_database()
        
        logger.info(f"LiveTradingBot initialized")
        logger.info(f"Mode: {'Testing' if testing_mode else 'Paper Trading' if paper_trading else 'Live Trading'}")
        logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"Symbols: {symbols}")
    
    def _init_database(self):
        """Initialize SQLite database for trade tracking"""
        try:
            self.db_path = 'trading_bot.db'
            conn = sqlite3.connect(self.db_path)
            
            # Create tables if they don't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    pnl REAL,
                    ml_probability REAL,
                    sentiment_score REAL,
                    confidence REAL,
                    status TEXT DEFAULT 'OPEN'
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    num_positions INTEGER NOT NULL,
                    daily_pnl REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    def is_market_open(self) -> bool:
        """Check if US stock market is currently open"""
        if self.testing_mode:
            # In testing mode, we'll simulate market hours
            test_time = self.current_test_time.time()
            test_day = self.current_test_time.weekday()
            
            # Update test time (simulate time passing)
            self.current_test_time += timedelta(minutes=5)
            
            # Check if it's a weekday (Monday = 0, Sunday = 6)
            if test_day >= 5:  # Saturday or Sunday
                logger.info("Market is closed (Weekend)")
                return False
            
            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = datetime_time(9, 30)
            market_close = datetime_time(16, 0)
            
            if test_time < market_open:
                logger.info("Market is not open yet")
                return False
            elif test_time > market_close:
                logger.info("Market is closed for the day")
                return False
            
            logger.info(f"Market is open (Test Time: {self.current_test_time.strftime('%Y-%m-%d %H:%M:%S')})")
            return True
        
        # Real market hours check
        et_tz = pytz.timezone(settings.TIMEZONE)
        now = datetime.now(et_tz)
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if now.weekday() >= 5:  # Saturday or Sunday
            logger.info("Market is closed (Weekend)")
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open:
            logger.info("Market is not open yet")
            return False
        elif now > market_close:
            logger.info("Market is closed for the day")
            return False
        
        logger.info("Market is open")
        return True
    
    def get_live_data(self, symbol: str, period: str = '5d') -> Optional[pd.DataFrame]:
        """Get live market data for a symbol"""
        try:
            logger.info(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            
            if self.testing_mode:
                # In testing mode, fetch historical data up to current_test_time
                end_date = self.current_test_time
                start_date = end_date - timedelta(days=60)  # Get 60 days of data
                
                logger.info(f"Testing Mode: Fetching historical data from {start_date.date()} to {end_date.date()}")
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if df.empty:
                    logger.warning(f"No historical data available for {symbol}")
                    return None
                
                logger.info(f"Retrieved {len(df)} historical daily points for {symbol}")
            else:
                # Normal live trading mode
                logger.info(f"Attempting to fetch 15-minute data for {symbol}")
                df = ticker.history(period='30d', interval='15m')
                
                if df.empty:
                    logger.info(f"No intraday data available for {symbol}, falling back to daily data")
                    df = ticker.history(period='30d', interval='1d')
                    if df.empty:
                        logger.warning(f"No data available for {symbol}")
                        return None
                
                logger.info(f"Retrieved {len(df)} raw data points for {symbol}")
                
                # Resample to daily data if we got intraday
                if len(df) > 10:  # If we got intraday data
                    logger.info(f"Resampling intraday data to daily for {symbol}")
                    df = df.resample('D').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                    logger.info(f"Resampled to {len(df)} daily points for {symbol}")
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure we have enough data
            if len(df) < 10:
                logger.warning(f"Insufficient data points for {symbol}: only {len(df)} days available")
                return None
            
            # Fix for stock splits and adjustments
            if symbol == 'NVDA' and df['close'].mean() < 100:
                logger.info(f"Adjusting NVDA price data for split")
                df['open'] *= 10
                df['high'] *= 10
                df['low'] *= 10
                df['close'] *= 10
            
            logger.info(f"Final dataset for {symbol}: {len(df)} days from {df.index[0]} to {df.index[-1]}")
            logger.info(f"Latest price for {symbol}: ${df['close'].iloc[-1]:.2f}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Generate a trade signal using ML and sentiment analysis"""
        try:
            logger.info(f"\n{'='*20} Analyzing {symbol} {'='*20}")
            
            # Get market data
            df = self.get_live_data(symbol, period='60d')  # Need more data for technical indicators
            if df is None or len(df) < 10:  # Changed from 30 to 10 to match our resampled data
                logger.warning(f"Insufficient data for {symbol} (minimum 10 days required)")
                return None
            
            # Calculate price changes
            price_change_1d = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            price_change_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
            
            logger.info("\nPrice Analysis:")
            logger.info(f"Current Price: ${df['close'].iloc[-1]:.2f}")
            logger.info(f"1-day Change: {price_change_1d:+.2f}%")
            logger.info(f"5-day Change: {price_change_5d:+.2f}%")
            
            # Get ML prediction with sentiment
            prediction = self.ml_predictor.predict(df, symbol=symbol)
            if prediction is None:
                logger.warning("No prediction available")
                return None
            
            # Extract probabilities and sentiment (single API call)
            prob_3d_up = prediction['3d'][0]
            prob_3d_down = prediction['3d'][1]
            sentiment_info = prediction.get('sentiment', {})
            sentiment_score = sentiment_info.get('sentiment_score', 0.0)
            
            current_price = df['close'].iloc[-1]
            
            # Print confidence levels for all symbols
            direction = 'UP' if prob_3d_up > prob_3d_down else 'DOWN'
            confidence = max(prob_3d_up, prob_3d_down)
            print(f"{symbol} Confidence: {confidence:.3f} ({direction})")
            
            logger.info("\nSignal Analysis:")
            logger.info(f"3-day Up Probability: {prob_3d_up:.3f}")
            logger.info(f"3-day Down Probability: {prob_3d_down:.3f}")
            logger.info(f"Sentiment Score: {sentiment_score:.3f}")
            logger.info(f"Required Confidence: {self.confidence_threshold:.3f}")
            
            # Determine trade direction with adjusted confidence threshold
            base_confidence = max(prob_3d_up, prob_3d_down)
            sentiment_boost = abs(sentiment_score) * 0.1
            
            # Align sentiment with prediction
            if (prob_3d_up > prob_3d_down and sentiment_score > 0) or \
               (prob_3d_down > prob_3d_up and sentiment_score < 0):
                final_confidence = base_confidence + sentiment_boost
            else:
                final_confidence = base_confidence
            
            # Lower confidence threshold slightly for strong technical signals
            adjusted_threshold = self.confidence_threshold
            if base_confidence > 0.60:  # Strong technical signal
                adjusted_threshold = self.confidence_threshold * 0.95  # 5% lower threshold
            
            if final_confidence > adjusted_threshold:
                direction = 'LONG' if prob_3d_up > prob_3d_down else 'SHORT'
                target_price = current_price * 1.03 if direction == 'LONG' else current_price * 0.97
                stop_loss = current_price * 0.98 if direction == 'LONG' else current_price * 1.02
                
                logger.info(f"\n✅ Strong {direction} signal detected:")
                logger.info(f"  Base Confidence: {base_confidence:.3f}")
                logger.info(f"  Sentiment Boost: {sentiment_boost:.3f}")
                logger.info(f"  Final Confidence: {final_confidence:.3f}")
                logger.info(f"  Target Price: ${target_price:.2f}")
                logger.info(f"  Stop Loss: ${stop_loss:.2f}")
                
                return TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    confidence=final_confidence,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    sentiment_score=sentiment_score,
                    ml_probability=prob_3d_up if direction == 'LONG' else prob_3d_down,
                    timestamp=self.current_test_time if self.testing_mode else datetime.now()
                )
            else:
                logger.info("\n❌ No actionable signal:")
                logger.info(f"  Final Confidence: {final_confidence:.3f}")
                logger.info(f"  Required Threshold: {adjusted_threshold:.3f}")
                logger.info(f"  Gap to Threshold: {(adjusted_threshold - final_confidence):.3f}")
                return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def calculate_position_size(self, signal: TradeSignal) -> int:
        """Calculate position size based on risk management"""
        try:
            risk_amount = self.portfolio_value * self.risk_per_trade
            
            if signal.direction == 'LONG':
                price_risk = signal.entry_price - signal.stop_loss
            else:  # SHORT
                price_risk = signal.stop_loss - signal.entry_price
            
            if price_risk <= 0:
                return 0
            
            # Calculate number of shares
            shares = int(risk_amount / price_risk)
            
            # Ensure we don't exceed 20% of portfolio in any single position
            max_position_value = self.portfolio_value * 0.20
            max_shares = int(max_position_value / signal.entry_price)
            
            shares = min(shares, max_shares)
            
            # Minimum 1 share, maximum based on available capital
            available_capital = self.portfolio_value * 0.95  # Keep 5% cash
            max_affordable_shares = int(available_capital / signal.entry_price)
            
            shares = max(1, min(shares, max_affordable_shares))
            
            logger.info(f"Position size for {signal.symbol}: {shares} shares (risk: ${risk_amount:.2f})")
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trade (paper trading or real)"""
        try:
            # Check if we already have a position in this symbol
            if signal.symbol in self.positions:
                logger.info(f"Trade rejected - Already have position in {signal.symbol}")
                logger.info(f"  Current position: {self.positions[signal.symbol].direction}")
                logger.info(f"  Entry price: ${self.positions[signal.symbol].entry_price:.2f}")
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                logger.info(f"Trade rejected - Maximum positions ({self.max_positions}) reached")
                logger.info(f"  Current positions: {list(self.positions.keys())}")
                return False
            
            # Calculate position size
            quantity = self.calculate_position_size(signal)
            if quantity == 0:
                logger.info(f"Trade rejected - Position size calculation returned 0 for {signal.symbol}")
                logger.info(f"  Entry price: ${signal.entry_price:.2f}")
                logger.info(f"  Available capital: ${self.portfolio_value:.2f}")
                return False
            
            # Execute trade
            if self.paper_trading:
                success = self._execute_paper_trade(signal, quantity)
            else:
                success = self._execute_real_trade(signal, quantity)
            
            if success:
                # Create position
                position = Position(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    quantity=quantity,
                    entry_time=signal.timestamp,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    current_price=signal.entry_price,
                    unrealized_pnl=0.0,
                    sentiment_score=signal.sentiment_score
                )
                
                self.positions[signal.symbol] = position
                
                # Log to database
                self._log_trade_to_db(signal, quantity, 'OPEN')
                
                logger.info(f"Executed {signal.direction} trade: {signal.symbol} x{quantity} @ ${signal.entry_price:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {str(e)}")
            return False
    
    def _execute_paper_trade(self, signal: TradeSignal, quantity: int) -> bool:
        """Execute a paper trade through Alpaca"""
        try:
            side = 'buy' if signal.direction == 'LONG' else 'sell'
            
            # Submit order to Alpaca
            self.api.submit_order(
                symbol=signal.symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"Paper trade submitted to Alpaca: {signal.direction} {signal.symbol} x{quantity}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing paper trade: {str(e)}")
            return False
    
    def _execute_real_trade(self, signal: TradeSignal, quantity: int) -> bool:
        """Execute a real trade through broker API"""
        # TODO: Implement real broker integration (Alpaca, IBKR, etc.)
        logger.warning("Real trading not implemented yet - use paper trading mode")
        return False
    
    def check_positions(self):
        """Check existing positions for exit signals"""
        try:
            # Get all positions from Alpaca
            alpaca_positions = {p.symbol: p for p in self.api.list_positions()}
        
            for symbol, position in list(self.positions.items()):
                try:
                    # Get current position from Alpaca
                    alpaca_pos = alpaca_positions.get(symbol)
                    if alpaca_pos is None:
                        continue
                
                    current_price = float(alpaca_pos.current_price)
                    position.current_price = current_price
                    position.unrealized_pnl = float(alpaca_pos.unrealized_pl)
                
                    # Check exit conditions
                    should_exit = False
                    exit_reason = ""
                
                    if position.direction == 'LONG':
                        if current_price >= position.target_price:
                            should_exit = True
                            exit_reason = "Target reached"
                        elif current_price <= position.stop_loss:
                            should_exit = True
                            exit_reason = "Stop loss hit"
                    else:  # SHORT
                        if current_price <= position.target_price:
                            should_exit = True
                            exit_reason = "Target reached"
                        elif current_price >= position.stop_loss:
                            should_exit = True
                            exit_reason = "Stop loss hit"
                
                    # Also check time-based exit (hold for max 3 days)
                    if (datetime.now() - position.entry_time).days >= 3:
                        should_exit = True
                        exit_reason = "Time limit reached"
                
                    if should_exit:
                        self.close_position(symbol, exit_reason)
                    else:
                        logger.info(f"Position {symbol}: ${current_price:.2f} (P&L: ${position.unrealized_pnl:.2f})")
                
                except Exception as e:
                    logger.error(f"Error checking position {symbol}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error checking positions: {str(e)}")
    
    def close_position(self, symbol: str, reason: str = "Manual"):
        """Close a position"""
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return
            
            position = self.positions[symbol]
            
            # Close position through Alpaca
            self.api.close_position(symbol)
            
            # Get the exit price from the last trade
            trades = self.api.get_trades(symbol, limit=1)
            if trades:
                exit_price = float(trades[0].price)
            else:
                exit_price = position.current_price
            
            # Calculate final PnL
            if position.direction == 'LONG':
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # SHORT
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Update database
            self._update_trade_in_db(symbol, exit_price, pnl)
            
            # Remove position from local tracking
            del self.positions[symbol]
            
            logger.info(f"Closed {position.direction} position: {symbol} @ ${exit_price:.2f} | P&L: ${pnl:.2f} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {str(e)}")
    
    def _log_trade_to_db(self, signal: TradeSignal, quantity: int, status: str):
        """Log trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO trades (symbol, direction, entry_price, quantity, entry_time, 
                                  ml_probability, sentiment_score, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (signal.symbol, signal.direction, signal.entry_price, quantity,
                  signal.timestamp.isoformat(), signal.ml_probability,
                  signal.sentiment_score, signal.confidence, status))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database logging error: {str(e)}")
    
    def _update_trade_in_db(self, symbol: str, exit_price: float, pnl: float):
        """Update trade in database when closed"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                UPDATE trades 
                SET exit_price = ?, exit_time = ?, pnl = ?, status = 'CLOSED'
                WHERE symbol = ? AND status = 'OPEN'
            ''', (exit_price, datetime.now().isoformat(), pnl, symbol))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database update error: {str(e)}")
    
    def _get_account_balance(self) -> float:
        """Get account balance from broker (placeholder)"""
        # TODO: Implement real broker integration
        return 100000.0
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            # Check market status
            is_market_open = self.is_market_open()
            current_time = datetime.now(pytz.timezone('America/New_York'))
            market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Print market status header with current time
            print("\n=== Market Status ===")
            print(f"Current Time (ET): {current_time.strftime('%I:%M:%S %p')}")
            print(f"Trading Hours: 9:30 AM - 4:00 PM ET (Mon-Fri)")
            
            if is_market_open:
                print("\n🟢 Market is OPEN")
                time_to_close = market_close_time - current_time
                hours, remainder = divmod(time_to_close.seconds, 3600)
                minutes = remainder // 60
                print(f"Time until market close: {hours}h {minutes:02d}m")
            else:
                print("\n🔴 Market is CLOSED")
                
                if current_time.weekday() >= 5:  # Weekend
                    days_to_monday = (7 - current_time.weekday())
                    next_open = current_time + timedelta(days=days_to_monday)
                    next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
                    print(f"Reason: Weekend ({current_time.strftime('%A')})")
                elif current_time < market_open_time:  # Pre-market
                    next_open = market_open_time
                    print("Reason: Pre-market hours")
                else:  # After-hours
                    next_open = (current_time + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
                    while next_open.weekday() >= 5:  # Skip weekends
                        next_open += timedelta(days=1)
                    print("Reason: After-hours")
                
                # Calculate time until next market open
                time_to_open = next_open - current_time
                total_hours = time_to_open.days * 24 + time_to_open.seconds // 3600
                minutes = (time_to_open.seconds % 3600) // 60
                
                # Format next market open date
                next_open_str = next_open.strftime("%A, %B %-d")  # This will show e.g., "Friday, February 9"
                
                print(f"Next market open: {next_open_str} at 9:30 AM ET")
                print(f"Time until market open: {total_hours}h {minutes:02d}m")
            
            if not is_market_open:
                print("\nSkipping trading cycle - market is closed")
                return
            
            # Get initial portfolio summary
            summary = self.get_portfolio_summary()
            
            # Check existing positions first
            self.check_positions()
            
            # Track predictions for all symbols
            all_predictions = {}
            
            # Look for new opportunities
            for symbol in self.symbols:
                try:
                    # Get live data and generate predictions
                    data = self.get_live_data(symbol)
                    if data is None:
                        logger.warning(f"Could not get live data for {symbol}")
                        continue
                        
                    predictions = self.ml_predictor.predict(data, symbol)
                    if predictions:
                        up_prob_3d, down_prob_3d = predictions['3d']
                        confidence = max(up_prob_3d, down_prob_3d)
                        direction = 'UP' if up_prob_3d > down_prob_3d else 'DOWN'
                        sentiment_score = predictions.get('sentiment', {}).get('sentiment_score', 0.0)
                        
                        all_predictions[symbol] = {
                            'confidence': confidence,
                            'direction': direction,
                            'sentiment_score': sentiment_score
                        }
                        
                        # Only try to execute trades if not using dummy models
                        if not isinstance(self.ml_predictor.model_3d, DummyClassifier):
                            if symbol not in self.positions:
                                signal = self.generate_trade_signal(symbol)
                                if signal:
                                    self.execute_trade(signal)
                                    time.sleep(1)  # Small delay between trades
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # Get final portfolio summary
            final_summary = self.get_portfolio_summary()
            
            # Print portfolio summary
            print("\n=== Portfolio Summary ===")
            print(f"Portfolio Value: ${final_summary['portfolio_value']:,.2f}")
            print(f"Open Positions: {final_summary['open_positions']}")
            print(f"Unrealized P&L: ${final_summary['unrealized_pnl']:,.2f}")
            print(f"Total Value: ${final_summary['total_value']:,.2f}")
            
            if final_summary['positions']:
                print("\n=== Open Positions ===")
                for symbol, pos in final_summary['positions'].items():
                    print(f"{symbol}: {pos['quantity']} shares @ ${pos['entry_price']:.2f} "
                          f"(Current: ${pos['current_price']:.2f}, P&L: ${pos['unrealized_pnl']:.2f})")
            
            print("\n=== ML Predictions ===")
            if isinstance(self.ml_predictor.model_3d, DummyClassifier):
                print("⚠️  Using dummy models (test mode) - predictions are random")
                print(f"📊 Confidence Threshold: {self.confidence_threshold:.1%}\n")
            
            # Sort predictions by confidence
            sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1]['confidence'], reverse=True)
            
            for symbol, pred in sorted_predictions:
                confidence_color = '🟢' if pred['confidence'] > self.confidence_threshold else '🟡' if pred['confidence'] > 0.55 else '🔴'
                threshold_indicator = '✓' if pred['confidence'] > self.confidence_threshold else 'x'
                print(f"{symbol:<6} {confidence_color} {pred['confidence']:.1%} {pred['direction']:<4} "
                      f"[{threshold_indicator}] (Sentiment: {pred['sentiment_score']:+.2f})")
            
            if not all_predictions:
                print("No predictions available - check logs for errors")
            print("\n")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Get latest predictions for all symbols
        symbol_predictions = {}
        for symbol in self.symbols:
            data = self.get_live_data(symbol)
            if data is not None:
                predictions = self.ml_predictor.predict(data, symbol)
                if predictions:
                    up_prob_3d, down_prob_3d = predictions['3d']
                    confidence = max(up_prob_3d, down_prob_3d)
                    direction = 'UP' if up_prob_3d > down_prob_3d else 'DOWN'
                    symbol_predictions[symbol] = {
                        'confidence': confidence,
                        'direction': direction,
                        'sentiment_score': predictions.get('sentiment', {}).get('sentiment_score', 0.0)
                    }
        
        return {
            'portfolio_value': self.portfolio_value,
            'open_positions': len(self.positions),
            'unrealized_pnl': total_unrealized_pnl,
            'total_value': self.portfolio_value + total_unrealized_pnl,
            'positions': {symbol: {
                'direction': pos.direction,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'sentiment_score': pos.sentiment_score
            } for symbol, pos in self.positions.items()},
            'predictions': symbol_predictions
        }

def start_live_trading(symbols: List[str] = None, paper_trading: bool = True, testing_mode: bool = False):
    """Start the live trading bot
    
    Args:
        symbols: List of stock symbols to trade. Defaults to ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
        paper_trading: Whether to run in paper trading mode. Defaults to True
        testing_mode: Whether to run in testing mode with historical data. Defaults to False
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    bot = LiveTradingBot(
        symbols=symbols, 
        paper_trading=paper_trading,
        testing_mode=testing_mode
    )
    
    # Schedule trading cycles
    schedule.every(5).minutes.do(bot.run_trading_cycle)  # Run every 5 minutes during market hours
    
    logger.info("Started Live Trading Bot!")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
        
        # Close all positions
        for symbol in list(bot.positions.keys()):
            bot.close_position(symbol, "Bot shutdown")
        
        # Final portfolio summary
        summary = bot.get_portfolio_summary()
        logger.info(f"Final Portfolio Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    # Start in paper trading mode by default
    start_live_trading(paper_trading=True) 