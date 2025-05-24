import yfinance as yf
import pandas as pd
import ta

# Parameters
TICKERS = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ORCL', 'IBM', 'TQQQ']  # Add more stocks as needed
START_DATE = '2020-01-01'
END_DATE = None  # Use None for up to today

def process_ticker(ticker):
    print(f"\nProcessing {ticker}...")
    output_csv = f'{ticker}_features.csv'
    
    # Download historical data
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    
    # Flatten MultiIndex columns if present
    def flatten_columns(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df

    df = flatten_columns(df)

    # Remove ticker suffix from all columns (e.g., 'Close_MSFT' -> 'Close')
    df.columns = [col.split('_')[0] if '_' in col else col for col in df.columns]

    # Print columns after flattening
    print(f"Columns after flattening for {ticker}:", df.columns)

    # Robustly rename columns to standard names (handles e.g. 'Close_AAPL')
    for col in list(df.columns):
        col_lower = col.lower()
        if 'close' in col_lower and 'adj' not in col_lower:
            df.rename(columns={col: 'Close'}, inplace=True)
        if 'open' in col_lower:
            df.rename(columns={col: 'Open'}, inplace=True)
        if 'high' in col_lower:
            df.rename(columns={col: 'High'}, inplace=True)
        if 'low' in col_lower:
            df.rename(columns={col: 'Low'}, inplace=True)
        if 'adj close' in col_lower:
            df.rename(columns={col: 'Adj Close'}, inplace=True)
        if 'volume' in col_lower:
            df.rename(columns={col: 'Volume'}, inplace=True)

    # Force all price columns to Series
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.Series(df[col].values, index=df.index)

    # Drop rows with missing values
    if df.isnull().values.any():
        df = df.dropna()

    # Feature engineering
    # RSI
    rsi = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['rsi'] = rsi
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    # SMA/EMA
    for window in [10, 20, 50, 200]:
        df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

    # Calculate next day return
    df['next_return'] = df['Close'].shift(-1) / df['Close'] - 1

    # Calculate 3-day and 5-day forward returns
    df['return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
    df['return_5d'] = df['Close'].shift(-5) / df['Close'] - 1

    # Label: 1 if 3-day forward return > 1%, 0 if < -1%, else drop (for 3-day target)
    mask_3d = (df['return_3d'] > 0.01) | (df['return_3d'] < -0.01)
    df['target_3d'] = None
    df.loc[mask_3d, 'target_3d'] = (df.loc[mask_3d, 'return_3d'] > 0.01).astype(int)

    # Label: 1 if 5-day forward return > 1%, 0 if < -1%, else drop (for 5-day target)
    mask_5d = (df['return_5d'] > 0.01) | (df['return_5d'] < -0.01)
    df['target_5d'] = None
    df.loc[mask_5d, 'target_5d'] = (df.loc[mask_5d, 'return_5d'] > 0.01).astype(int)

    # Drop last row (no target)
    df = df.iloc[:-1]

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['Close'].pct_change(lag)

    # Rolling volatility (std of returns)
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['Close'].pct_change().rolling(window=window).std()

    # Rolling mean (momentum)
    for window in [5, 10, 20]:
        df[f'momentum_{window}'] = df['Close'].pct_change().rolling(window=window).mean()

    # Price ratios
    if 'Open' in df.columns:
        df['close_open_ratio'] = df['Close'] / df['Open']
    if 'High' in df.columns:
        df['close_high_ratio'] = df['Close'] / df['High']
    if 'Low' in df.columns:
        df['close_low_ratio'] = df['Close'] / df['Low']
    if 'High' in df.columns and 'Low' in df.columns:
        df['high_low_ratio'] = df['High'] / df['Low']

    # Calendar features
    if not df.index.name or df.index.name.lower() != 'date':
        df.index = pd.to_datetime(df.index)
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Add ticker column
    df['ticker'] = ticker

    # Only drop rows with NaN in all columns at the very end
    df = df.dropna(how='all')

    # Debug: Print columns and sample rows before saving
    print(f"Columns before saving for {ticker}:", df.columns)
    print(f"Sample rows before saving for {ticker}:\n", df.tail(5))

    # Save to CSV
    df.to_csv(output_csv)
    print(f"Feature CSV saved to {output_csv}")
    return df

# Process all tickers
all_dfs = []
for ticker in TICKERS:
    try:
        df = process_ticker(ticker)
        all_dfs.append(df)
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")

# Combine all dataframes if at least one was processed successfully
if all_dfs:
    combined_df = pd.concat(all_dfs)
    combined_df.to_csv('combined_features.csv')
    print("\nCombined features saved to combined_features.csv") 