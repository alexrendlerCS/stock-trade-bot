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
    
    # Map yfinance columns to expected format
    column_mapping = {
        f'adj_close_{symbol.lower()}': 'close',
        f'close_{symbol.lower()}': 'close', 
        f'high_{symbol.lower()}': 'high',
        f'low_{symbol.lower()}': 'low',
        f'open_{symbol.lower()}': 'open',
        f'volume_{symbol.lower()}': 'volume',
        'adj_close': 'close',
        'close': 'close',
        'high': 'high', 
        'low': 'low',
        'open': 'open',
        'volume': 'volume'
    }
    
    # Apply column mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Keep only the essential columns
    essential_cols = ['close', 'high', 'low', 'open', 'volume']
    df = df[essential_cols].copy()
    
    print(f"Final DataFrame columns for {symbol}: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    print("Sample data:")
    print(df.head())
    
    # Simple backtesting logic
    trades = []
    
    for i in range(30, len(df)):  # Start from day 30 to have enough data for features
        current_data = df.iloc[:i+1]  # Data up to current day
        
        # Get prediction with news sentiment
        prediction = ml.predict(current_data, symbol=symbol)
        
        if prediction is None:
            continue
        
        # Extract prediction probabilities
        prob_3d_up = prediction['3d'][0]
        prob_3d_down = prediction['3d'][1]
        
        # Get sentiment info if available
        sentiment_info = prediction.get('sentiment', {})
        
        # Simple trading strategy based on prediction confidence
        confidence_threshold = 0.6
        
        if prob_3d_up > confidence_threshold:
            position = 'LONG'
        elif prob_3d_down > confidence_threshold:
            position = 'SHORT'
        else:
            continue  # No trade
        
        # Calculate trade results (simplified)
        entry_price = df.iloc[i]['close']
        entry_date = df.index[i]
        
        # Exit after 3 days or at end of data
        exit_idx = min(i + 3, len(df) - 1)
        exit_price = df.iloc[exit_idx]['close']
        exit_date = df.index[exit_idx]
        
        if position == 'LONG':
            pnl = (exit_price - entry_price) / entry_price * 100
        else:  # SHORT
            pnl = (entry_price - exit_price) / entry_price * 100
        
        trades.append({
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_pnl': pnl,
            'prob_3d_up': prob_3d_up,
            'prob_3d_down': prob_3d_down,
            'sentiment_score': sentiment_info.get('sentiment_score', 0),
            'news_count': sentiment_info.get('news_count', 0),
            'sentiment_strength': sentiment_info.get('sentiment_strength', 0)
        })
    
    print(f"Completed {symbol}: {len(trades)} trades")
    all_results.extend(trades)

print(f"\nBacktest completed. Total trades: {len(all_results)}")

if all_results:
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate cumulative PnL
    results_df['cum_pnl'] = results_df['trade_pnl'].cumsum()
    
    print(results_df)
    
    # Summary by symbol
    summary = results_df.groupby('symbol')['trade_pnl'].agg(['count', 'sum', 'mean'])
    print(f"\nSummary by symbol:")
    print(summary)
    
    print(f"\nTotal PnL: {results_df['trade_pnl'].sum()}")
    
    # Sentiment analysis summary
    if 'sentiment_score' in results_df.columns:
        print(f"\nSentiment Analysis Summary:")
        print(f"Average sentiment score: {results_df['sentiment_score'].mean():.3f}")
        print(f"Average news count: {results_df['news_count'].mean():.1f}")
        print(f"Average sentiment strength: {results_df['sentiment_strength'].mean():.3f}")
        
        # Correlation between sentiment and PnL
        sentiment_pnl_corr = results_df['sentiment_score'].corr(results_df['trade_pnl'])
        print(f"Correlation between sentiment and PnL: {sentiment_pnl_corr:.3f}")
    
    # Save results
    results_df.to_csv('backtest_results_with_sentiment.csv', index=False)
    print(f"\nResults saved to 'backtest_results_with_sentiment.csv'")
else:
    print("No trades executed!") 