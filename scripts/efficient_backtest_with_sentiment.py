import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from app.ml_predictor import MLPredictor
import pickle
import os
from tqdm import tqdm

def load_or_fetch_sentiment_cache(symbols, ml_predictor, cache_file='sentiment_cache.pkl'):
    """Load sentiment data from cache or fetch fresh data"""
    
    # Try to load existing cache
    sentiment_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                sentiment_cache = pickle.load(f)
            print(f"✅ Loaded sentiment cache with {len(sentiment_cache)} symbols")
        except Exception as e:
            print(f"⚠️  Could not load cache: {str(e)}")
            sentiment_cache = {}
    
    # Check which symbols need fresh sentiment data
    symbols_to_fetch = []
    for symbol in symbols:
        if symbol not in sentiment_cache:
            symbols_to_fetch.append(symbol)
        else:
            # Check if cache is recent (within 1 day)
            cache_time = datetime.fromisoformat(sentiment_cache[symbol].get('timestamp', '2020-01-01'))
            if (datetime.now() - cache_time).days > 1:
                symbols_to_fetch.append(symbol)
    
    # Fetch sentiment for symbols that need it
    if symbols_to_fetch and ml_predictor.news_analyzer:
        print(f"📰 Fetching fresh sentiment data for: {symbols_to_fetch}")
        
        for symbol in symbols_to_fetch:
            try:
                print(f"   Fetching sentiment for {symbol}...")
                sentiment_data = ml_predictor.news_analyzer.get_news_sentiment(symbol, lookback_days=7)
                sentiment_cache[symbol] = sentiment_data
                print(f"   ✅ {symbol}: sentiment={sentiment_data['sentiment_score']:.3f}, news={sentiment_data['news_count']}")
            except Exception as e:
                print(f"   ❌ Error fetching {symbol}: {str(e)}")
                # Use default sentiment if fetch fails
                sentiment_cache[symbol] = {
                    'sentiment_score': 0.0,
                    'news_count': 0,
                    'sentiment_strength': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Save updated cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(sentiment_cache, f)
            print(f"💾 Sentiment cache saved")
        except Exception as e:
            print(f"⚠️  Could not save cache: {str(e)}")
    
    elif not symbols_to_fetch:
        print(f"✅ Using cached sentiment data (all symbols up to date)")
    else:
        print(f"⚠️  No news analyzer available, using neutral sentiment")
        for symbol in symbols:
            if symbol not in sentiment_cache:
                sentiment_cache[symbol] = {
                    'sentiment_score': 0.0,
                    'news_count': 0,
                    'sentiment_strength': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
    
    return sentiment_cache

def run_efficient_backtest():
    """Run an efficient backtest with cached sentiment data"""
    
    symbols = ['AAPL', 'MSFT', 'TSLA']
    start_date = '2024-01-01'
    end_date = '2024-06-01'
    
    print("🚀 Starting Efficient Backtest with Sentiment Analysis")
    print("=" * 60)
    
    # Initialize ML predictor
    print("1. Initializing ML Predictor...")
    ml = MLPredictor()
    
    if ml.model_3d is None or ml.model_5d is None:
        print("❌ Failed to load ML models!")
        return
    
    print("✅ ML Predictor initialized")
    
    # Load or fetch sentiment data (this is the key optimization!)
    print("\n2. Loading/Fetching Sentiment Data...")
    sentiment_cache = load_or_fetch_sentiment_cache(symbols, ml)
    
    # Run backtest for each symbol
    print(f"\n3. Running Backtest for {len(symbols)} symbols...")
    all_results = []
    
    for symbol in symbols:
        print(f"\n📈 Backtesting {symbol}...")
        
        try:
            # Download price data
            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                print(f"❌ No price data for {symbol}")
                continue
            
            # Clean column names
            if hasattr(df.columns, 'levels'):  # MultiIndex
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Map columns
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
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
            
            # Keep essential columns
            essential_cols = ['close', 'high', 'low', 'open', 'volume']
            df = df[essential_cols].copy()
            
            print(f"   📊 Data shape: {df.shape}")
            
            # Get cached sentiment for this symbol
            symbol_sentiment = sentiment_cache.get(symbol, {
                'sentiment_score': 0.0,
                'news_count': 0,
                'sentiment_strength': 0.0
            })
            
            print(f"   📰 Using sentiment: score={symbol_sentiment['sentiment_score']:.3f}, news={symbol_sentiment['news_count']}")
            
            # Run backtest with progress bar
            trades = []
            trading_days = list(range(30, len(df)))  # Start from day 30
            
            for i in tqdm(trading_days, desc=f"   Trading {symbol}", unit="days"):
                current_data = df.iloc[:i+1]
                
                # Get ML prediction (WITHOUT calling news API again - we use cached sentiment)
                prediction = ml.predict(current_data)
                
                if prediction is None:
                    continue
                
                # Extract probabilities
                prob_3d_up = prediction['3d'][0]
                prob_3d_down = prediction['3d'][1]
                
                # Simple strategy
                confidence_threshold = 0.6
                
                if prob_3d_up > confidence_threshold:
                    position = 'LONG'
                elif prob_3d_down > confidence_threshold:
                    position = 'SHORT'
                else:
                    continue
                
                # Calculate trade results
                entry_price = df.iloc[i]['close']
                entry_date = df.index[i]
                
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
                    'sentiment_score': symbol_sentiment['sentiment_score'],
                    'news_count': symbol_sentiment['news_count'],
                    'sentiment_strength': symbol_sentiment['sentiment_strength']
                })
            
            print(f"   ✅ Completed {symbol}: {len(trades)} trades")
            all_results.extend(trades)
            
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {str(e)}")
            continue
    
    # Analyze results
    print(f"\n4. Analyzing Results...")
    print("=" * 60)
    
    if not all_results:
        print("❌ No trades executed!")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df['cum_pnl'] = results_df['trade_pnl'].cumsum()
    
    print(f"📊 Total Trades: {len(results_df)}")
    print(f"💰 Total PnL: {results_df['trade_pnl'].sum():.2f}%")
    print(f"📈 Win Rate: {(results_df['trade_pnl'] > 0).mean():.1%}")
    
    # Summary by symbol
    print(f"\n📋 Summary by Symbol:")
    summary = results_df.groupby('symbol').agg({
        'trade_pnl': ['count', 'sum', 'mean'],
        'sentiment_score': 'first',
        'news_count': 'first'
    }).round(3)
    print(summary)
    
    # Sentiment analysis
    print(f"\n📰 Sentiment Analysis:")
    print(f"Average sentiment score: {results_df['sentiment_score'].mean():.3f}")
    print(f"Average news count: {results_df['news_count'].mean():.1f}")
    
    # Correlation analysis
    if results_df['sentiment_score'].std() > 0:  # Only if there's variation in sentiment
        sentiment_corr = results_df['sentiment_score'].corr(results_df['trade_pnl'])
        print(f"Sentiment-PnL correlation: {sentiment_corr:.3f}")
    else:
        print("Sentiment-PnL correlation: N/A (no sentiment variation)")
    
    # Save results
    output_file = 'efficient_backtest_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    
    # Show top trades
    print(f"\n🏆 Top 5 Profitable Trades:")
    top_trades = results_df.nlargest(5, 'trade_pnl')[['symbol', 'entry_date', 'position', 'trade_pnl', 'sentiment_score']]
    print(top_trades.to_string(index=False))
    
    print(f"\n🎉 Efficient Backtest Completed Successfully!")

if __name__ == "__main__":
    run_efficient_backtest() 