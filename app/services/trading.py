import alpaca_trade_api as tradeapi
from app.core.config import settings
from app.db.session import get_db
from app.db.models import Trade, Position, Signal, StrategyType, TradeStatus
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from app.strategies.ema_crossover_strategy import EMACrossoverStrategy
from app.strategies.ml_strategy import MLStrategy

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Alpaca API
api = tradeapi.REST(
    settings.ALPACA_API_KEY,
    settings.ALPACA_SECRET_KEY,
    settings.ALPACA_BASE_URL
)

# Initialize strategies
ema_strategy = EMACrossoverStrategy()
ml_strategy = MLStrategy()

def execute_market_open_tasks():
    """Execute tasks at market open"""
    logger.info("Executing market open tasks")
    
    # Get current positions and update status
    update_positions()
    
    # Execute day trading strategy
    execute_day_trading_strategy()
    
    # Check and execute swing trade signals
    execute_swing_trade_signals()
    
    # Update long-term positions
    update_long_term_positions()

def execute_market_close_tasks():
    """Execute tasks at market close"""
    logger.info("Executing market close tasks")
    
    # Close all day trading positions
    close_day_trading_positions()
    
    # Update performance metrics
    update_performance_metrics()

def execute_trade(symbol: str, quantity: int, side: str, strategy_type: StrategyType, reason: str):
    """Execute a trade through Alpaca with a detailed reason"""
    try:
        # Calculate stop loss and take profit
        current_price = float(api.get_latest_trade(symbol).price)
        stop_loss = current_price * (1 - settings.STOP_LOSS_PERCENTAGE)
        take_profit = current_price * (1 + settings.TAKE_PROFIT_PERCENTAGE)
        
        # Place the order
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type='market',
            time_in_force='day'
        )
        
        # Create trade record
        db = next(get_db())
        trade = Trade(
            symbol=symbol,
            strategy_type=strategy_type,
            entry_price=current_price,
            quantity=quantity,
            status=TradeStatus.EXECUTED,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        
        logger.info(f"Executed {side} trade for {symbol}: {quantity} shares | Reason: {reason}")
        return trade
        
    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {str(e)} | Reason: {reason}")
        return None

def update_positions():
    """Update current positions and their status"""
    try:
        positions = api.list_positions()
        db = next(get_db())
        
        for position in positions:
            # Update or create position record
            db_position = db.query(Position).filter(
                Position.symbol == position.symbol,
                Position.is_active == True
            ).first()
            
            if db_position:
                db_position.current_price = float(position.current_price)
            else:
                db_position = Position(
                    symbol=position.symbol,
                    quantity=int(position.qty),
                    entry_price=float(position.avg_entry_price),
                    current_price=float(position.current_price),
                    is_active=True
                )
                db.add(db_position)
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error updating positions: {str(e)}")

def execute_day_trading_strategy():
    """Execute day trading strategy"""
    # TODO: Implement day trading strategy logic
    pass

def execute_swing_trade_signals():
    """Execute swing trading strategy"""
    try:
        # List of symbols to trade
        symbols = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "TQQQ"]
        capital = 100000  # Replace with your portfolio logic
        
        for symbol in symbols:
            # Fetch latest data
            data = fetch_latest_data(symbol)
            if data.empty:
                continue
            
            # Get signals from both strategies
            ema_signals = ema_strategy.generate_signals(data)
            ml_signals = ml_strategy.generate_signals(data)
            
            # Combine signals
            all_signals = ema_signals + ml_signals
            
            # Execute trades based on signals
            for signal in all_signals:
                # Calculate position size (1% of capital per trade)
                price = data['close'].iloc[-1]
                quantity = int(ml_strategy.calculate_position_size(capital, 0.01) // price)
                
                # Execute trade
                execute_trade(
                    symbol=signal.symbol,
                    quantity=quantity,
                    side='buy' if signal.signal_type == 'BUY' else 'sell',
                    strategy_type=signal.strategy_type,
                    reason=signal.reason
                )
    
    except Exception as e:
        logger.error(f"Error executing swing trade signals: {str(e)}")

def update_long_term_positions():
    """Update long-term investment positions"""
    # TODO: Implement long-term position management
    pass

def close_day_trading_positions():
    """Close all day trading positions at market close"""
    try:
        positions = api.list_positions()
        for position in positions:
            # Check if position is a day trade
            db = next(get_db())
            db_position = db.query(Position).filter(
                Position.symbol == position.symbol,
                Position.strategy_type == StrategyType.DAY_TRADE,
                Position.is_active == True
            ).first()
            
            if db_position:
                # Close the position
                api.submit_order(
                    symbol=position.symbol,
                    qty=position.qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                # Update position status
                db_position.is_active = False
                db.commit()
                
                logger.info(f"Closed day trading position for {position.symbol}")
                
    except Exception as e:
        logger.error(f"Error closing day trading positions: {str(e)}")

def update_performance_metrics():
    """Update daily performance metrics"""
    # TODO: Implement performance metrics calculation
    pass

def fetch_historical_data(symbol: str, start: str, end: str, timeframe: str = 'day') -> pd.DataFrame:
    """Fetch historical OHLCV data from Alpaca."""
    bars = api.get_bars(symbol, timeframe, start=start, end=end).df
    if bars.empty:
        return pd.DataFrame()
    bars = bars.reset_index()
    bars = bars.rename(columns={
        'timestamp': 'datetime',
        'trade_count': 'volume'
    })
    bars = bars[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    bars.set_index('datetime', inplace=True)
    bars.name = symbol
    return bars

def fetch_latest_data(symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch the latest OHLCV data for a symbol (default: last 30 days)."""
    tz = pytz.timezone(settings.TIMEZONE)
    end = datetime.now(tz)
    start = end - timedelta(days=lookback_days)
    return fetch_historical_data(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

def run_ml_strategy_and_trade(symbol: str, capital: float):
    """Run ML strategy and execute trades for a symbol."""
    data = fetch_latest_data(symbol)
    if data.empty:
        logger.warning(f"No data fetched for {symbol}, skipping strategy run.")
        return
    
    signals = ml_strategy.generate_signals(data)
    for signal in signals:
        # Calculate position size (1% of capital per trade)
        price = data['close'].iloc[-1]
        quantity = int(ml_strategy.calculate_position_size(capital, 0.01) // price)
        
        execute_trade(
            symbol=signal.symbol,
            quantity=quantity,
            side='buy' if signal.signal_type == 'BUY' else 'sell',
            strategy_type=signal.strategy_type,
            reason=signal.reason
        ) 