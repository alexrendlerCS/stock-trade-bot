import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from app.ml_predictor import MLPredictor

symbols = ['AAPL', 'MSFT', 'TSLA']  # You can add more symbols here
start_date = '2024-01-01'
end_date = '2024-06-01'

ml = MLPredictor()

all_results = []

for symbol in symbols:
    print(f"Backtesting {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
    
    # Print original columns for debugging
    print(f"Original columns: {df.columns.tolist()}")
    
    # Fix column names - yfinance returns MultiIndex columns or simple names
    if hasattr(df.columns, 'levels'):  # MultiIndex
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    # Clean up column names and standardize them
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Map yfinance columns to expected names - handle duplicates
    new_df = pd.DataFrame(index=df.index)
    
    # Find and map the correct columns
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
    
    # Ensure we have the required columns
    required_cols = ['close', 'high', 'low', 'open', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for {symbol}: {missing_cols}")
        continue
    
    print(f"Final DataFrame columns for {symbol}: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    results = []
    position = None
    entry_price = 0
    pnl = 0
    
    for date in df.index[30:]:  # start after 30 days to have enough lookback
        df_slice = df.loc[:date].copy()
        preds = ml.predict(df_slice)
        if preds:
            prob_3d_up, prob_3d_down = preds['3d']
            signal = None
            if prob_3d_up > 0.6:
                signal = 'BUY'
            elif prob_3d_down > 0.6:
                signal = 'SELL'
            # Simulate simple strategy: buy/sell and close after 3 days
            if signal == 'BUY' and position is None:
                position = 'LONG'
                entry_price = df_slice['close'].iloc[-1]
                entry_date = date
            elif signal == 'SELL' and position is None:
                position = 'SHORT'
                entry_price = df_slice['close'].iloc[-1]
                entry_date = date
            # Close position after 3 days
            if position and (date - entry_date).days >= 3:
                exit_price = df_slice['close'].iloc[-1]
                if position == 'LONG':
                    trade_pnl = exit_price - entry_price
                else:
                    trade_pnl = entry_price - exit_price
                pnl += trade_pnl
                results.append({
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'exit_date': date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'trade_pnl': trade_pnl,
                    'cum_pnl': pnl,
                    'prob_3d_up': prob_3d_up,
                    'prob_3d_down': prob_3d_down
                })
                position = None
        else:
            print(f"No predictions for {symbol} on {date}")
            break  # Break after first failure to avoid spam
    
    all_results.extend(results)
    print(f"Completed {symbol}: {len(results)} trades")

results_df = pd.DataFrame(all_results)
print(f"\nBacktest completed. Total trades: {len(results_df)}")
print(results_df.head())

# Summary stats
if not results_df.empty:
    print("\nSummary by symbol:")
    print(results_df.groupby('symbol')['trade_pnl'].agg(['count', 'sum', 'mean']))
    print("\nTotal PnL:", results_df['trade_pnl'].sum())
else:
    print("No trades were made in the backtest.") 