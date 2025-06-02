import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ta

symbols = ['AAPL', 'MSFT', 'TSLA']
start_date = '2024-01-01'
end_date = '2024-06-01'

all_results = []

def simple_technical_strategy(df):
    """
    Simple moving average crossover strategy
    """
    # Calculate short and long moving averages
    df['sma_5'] = ta.trend.SMAIndicator(df['close'], window=5).sma_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=10).rsi()
    
    # Drop NaN values
    df = df.dropna()
    
    if len(df) == 0:
        return None
    
    latest = df.iloc[-1]
    
    # Simple strategy rules
    if latest['sma_5'] > latest['sma_20'] and latest['rsi'] < 70:
        return 'BUY'  # Short MA above long MA and not overbought
    elif latest['sma_5'] < latest['sma_20'] and latest['rsi'] > 30:
        return 'SELL'  # Short MA below long MA and not oversold
    else:
        return 'HOLD'

for symbol in symbols:
    print(f"Backtesting {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
    
    # Fix column names
    if hasattr(df.columns, 'levels'):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Map columns
    new_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if 'close' in col and 'close' not in new_df.columns:
            new_df['close'] = df[col]
        elif 'high' in col and 'high' not in new_df.columns:
            new_df['high'] = df[col]
        elif 'low' in col and 'low' not in new_df.columns:
            new_df['low'] = df[col]
        elif 'open' in col and 'open' not in new_df.columns:
            new_df['open'] = df[col]
        elif 'volume' in col and 'volume' not in new_df.columns:
            new_df['volume'] = df[col]
    
    df = new_df
    
    required_cols = ['close', 'high', 'low', 'open', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for {symbol}: {missing_cols}")
        continue
    
    print(f"Processing {symbol} with {df.shape[0]} rows")
    
    results = []
    position = None
    entry_price = 0
    pnl = 0
    entry_date = None
    
    for date in df.index[25:]:  # Start after 25 days for moving averages
        df_slice = df.loc[:date].copy()
        signal = simple_technical_strategy(df_slice)
        
        if signal:
            current_price = df_slice['close'].iloc[-1]
            
            # Enter position
            if signal == 'BUY' and position is None:
                position = 'LONG'
                entry_price = current_price
                entry_date = date
                print(f"  BUY {symbol} at {current_price:.2f} on {date.date()}")
            elif signal == 'SELL' and position is None:
                position = 'SHORT'
                entry_price = current_price
                entry_date = date
                print(f"  SELL {symbol} at {current_price:.2f} on {date.date()}")
            
            # Close position after 3 days
            if position and entry_date and (date - entry_date).days >= 3:
                exit_price = current_price
                if position == 'LONG':
                    trade_pnl = exit_price - entry_price
                else:
                    trade_pnl = entry_price - exit_price
                
                pnl += trade_pnl
                
                print(f"  CLOSE {position} {symbol} at {exit_price:.2f}, PnL: ${trade_pnl:.2f}")
                
                results.append({
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'exit_date': date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'trade_pnl': trade_pnl,
                    'cum_pnl': pnl
                })
                position = None
                entry_date = None
    
    all_results.extend(results)
    print(f"Completed {symbol}: {len(results)} trades, Total PnL: ${pnl:.2f}")

# Summary
results_df = pd.DataFrame(all_results)
print(f"\n=== BACKTEST SUMMARY ===")
print(f"Total trades: {len(results_df)}")

if not results_df.empty:
    print("\nTrades by symbol:")
    print(results_df.groupby('symbol').agg({
        'trade_pnl': ['count', 'sum', 'mean']
    }).round(2))
    
    print(f"\nTotal PnL: ${results_df['trade_pnl'].sum():.2f}")
    print(f"Win Rate: {(results_df['trade_pnl'] > 0).mean():.2%}")
    print(f"Average Trade: ${results_df['trade_pnl'].mean():.2f}")
    
    print("\nRecent trades:")
    print(results_df.tail().to_string(index=False))
else:
    print("No trades were made in the backtest.") 