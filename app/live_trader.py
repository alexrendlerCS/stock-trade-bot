import pandas as pd
import yfinance as yf
import time
import logging
from datetime import datetime, timedelta
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
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
                 confidence_threshold: float = 0.65):
        
        self.symbols = symbols
        self.paper_trading = paper_trading
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.confidence_threshold = confidence_threshold
        
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
        
        # Database for trade history
        self._init_database()
        
        logger.info(f"LiveTradingBot initialized")
        logger.info(f"Paper Trading: {paper_trading}")
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
        # Get current time in Eastern Time
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
            
            # Try to get more frequent data for a shorter period
            logger.info(f"Attempting to fetch 15-minute data for {symbol}")
            df = ticker.history(period='10d', interval='15m')
            
            if df.empty:
                logger.info(f"No intraday data available for {symbol}, falling back to daily data")
                df = ticker.history(period='10d', interval='1d')
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
            if len(df) < 5:
                logger.warning(f"Insufficient data points for {symbol}: only {len(df)} days available")
                return None
            
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
            if df is None or len(df) < 30:
                logger.warning(f"Insufficient data for {symbol}")
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
            
            # Extract probabilities
            prob_3d_up = prediction['3d'][0]
            prob_3d_down = prediction['3d'][1]
            
            # Get sentiment info
            sentiment_info = prediction.get('sentiment', {})
            sentiment_score = sentiment_info.get('sentiment_score', 0.0)
            
            current_price = df['close'].iloc[-1]
            
            logger.info("\nSignal Analysis:")
            logger.info(f"3-day Up Probability: {prob_3d_up:.3f}")
            logger.info(f"3-day Down Probability: {prob_3d_down:.3f}")
            logger.info(f"Sentiment Score: {sentiment_score:.3f}")
            logger.info(f"Required Confidence: {self.confidence_threshold:.3f}")
            
            # Determine trade direction
            if prob_3d_up > self.confidence_threshold:
                direction = 'LONG'
                confidence = prob_3d_up
                target_price = current_price * 1.03  # 3% target
                stop_loss = current_price * 0.98     # 2% stop loss
                logger.info(f"\n✅ Strong LONG signal detected:")
                logger.info(f"  Base Confidence: {prob_3d_up:.3f}")
                logger.info(f"  Sentiment Boost: {max(0, sentiment_score * 0.1):.3f}")
                logger.info(f"  Target Price: ${target_price:.2f} (+3%)")
                logger.info(f"  Stop Loss: ${stop_loss:.2f} (-2%)")
                
            elif prob_3d_down > self.confidence_threshold:
                direction = 'SHORT'
                confidence = prob_3d_down
                target_price = current_price * 0.97  # 3% target (price going down)
                stop_loss = current_price * 1.02     # 2% stop loss
                logger.info(f"\n✅ Strong SHORT signal detected:")
                logger.info(f"  Base Confidence: {prob_3d_down:.3f}")
                logger.info(f"  Sentiment Boost: {max(0, -sentiment_score * 0.1):.3f}")
                logger.info(f"  Target Price: ${target_price:.2f} (-3%)")
                logger.info(f"  Stop Loss: ${stop_loss:.2f} (+2%)")
                
            else:
                logger.info("\n❌ No actionable signal:")
                logger.info(f"  Highest Confidence: {max(prob_3d_up, prob_3d_down):.3f}")
                logger.info(f"  Required Threshold: {self.confidence_threshold:.3f}")
                logger.info(f"  Gap to Threshold: {(self.confidence_threshold - max(prob_3d_up, prob_3d_down)):.3f}")
                return None
            
            # Enhance signal with sentiment
            sentiment_boost = abs(sentiment_score) * 0.1
            if (direction == 'LONG' and sentiment_score > 0) or (direction == 'SHORT' and sentiment_score < 0):
                confidence += sentiment_boost
                logger.info(f"\nSentiment Alignment:")
                logger.info(f"  Direction matches sentiment")
                logger.info(f"  Confidence boosted by: +{sentiment_boost:.3f}")
            else:
                logger.info(f"\nSentiment Misalignment:")
                logger.info(f"  Sentiment contradicts direction")
                logger.info(f"  No confidence boost applied")
            
            # Cap confidence at 0.95
            confidence = min(confidence, 0.95)
            
            # Final confidence check
            if confidence < self.confidence_threshold:
                logger.info(f"\n❌ Final confidence ({confidence:.3f}) below threshold ({self.confidence_threshold:.3f})")
                return None
            
            signal = TradeSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                sentiment_score=sentiment_score,
                ml_probability=prob_3d_up if direction == 'LONG' else prob_3d_down,
                timestamp=datetime.now()
            )
            
            logger.info(f"\n✅ Final Signal Generated:")
            logger.info(f"  Direction: {direction}")
            logger.info(f"  Entry Price: ${current_price:.2f}")
            logger.info(f"  Target Price: ${target_price:.2f}")
            logger.info(f"  Stop Loss: ${stop_loss:.2f}")
            logger.info(f"  Final Confidence: {confidence:.3f}")
            logger.info(f"  Risk/Reward Ratio: {abs(target_price - current_price) / abs(stop_loss - current_price):.2f}")
            logger.info("="*50 + "\n")
            
            return signal
            
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
            if not self.is_market_open():
                return
            
            logger.info("\n" + "="*50)
            logger.info("Starting new trading cycle...")
            logger.info("="*50)
            
            # Check existing positions first
            if self.positions:
                logger.info("\nChecking existing positions:")
                for symbol, pos in self.positions.items():
                    logger.info(f"  {symbol}: {pos.direction} | Entry: ${pos.entry_price:.2f} | Current: ${pos.current_price:.2f} | P&L: ${pos.unrealized_pnl:.2f}")
            else:
                logger.info("\nNo open positions")
            
            self.check_positions()
            
            # Look for new opportunities
            logger.info("\nAnalyzing symbols for new opportunities:")
            for symbol in self.symbols:
                try:
                    if symbol in self.positions:
                        logger.info(f"  {symbol}: Skipping - Already have position")
                        continue
                    
                    logger.info(f"\nAnalyzing {symbol}:")
                    signal = self.generate_trade_signal(symbol)
                    
                    if signal:
                        logger.info(f"  Trade signal generated for {symbol}:")
                        logger.info(f"    Direction: {signal.direction}")
                        logger.info(f"    Confidence: {signal.confidence:.3f}")
                        logger.info(f"    Entry Price: ${signal.entry_price:.2f}")
                        logger.info(f"    Target: ${signal.target_price:.2f}")
                        logger.info(f"    Stop Loss: ${signal.stop_loss:.2f}")
                        logger.info(f"    Sentiment Score: {signal.sentiment_score:.3f}")
                        
                        self.execute_trade(signal)
                        time.sleep(1)  # Small delay between trades
                    else:
                        logger.info(f"  No actionable signal for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # Log portfolio status
            total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            logger.info("\nEnd of trading cycle summary:")
            logger.info(f"  Portfolio Value: ${self.portfolio_value:,.2f}")
            logger.info(f"  Open Positions: {len(self.positions)}")
            logger.info(f"  Unrealized P&L: ${total_pnl:,.2f}")
            logger.info("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
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
            } for symbol, pos in self.positions.items()}
        }

def start_live_trading(symbols: List[str] = None, paper_trading: bool = True):
    """Start the live trading bot"""
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    bot = LiveTradingBot(symbols=symbols, paper_trading=paper_trading)
    
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