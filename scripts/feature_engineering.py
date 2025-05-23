import yfinance as yf
import pandas as pd
import ta

# Parameters
TICKER = 'MSFT'
START_DATE = '2020-01-01'
END_DATE = None  # Use None for up to today
OUTPUT_CSV = f'{TICKER}_features.csv'

# Download historical data
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Flatten MultiIndex columns if present
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

df = flatten_columns(df)

# Print columns after flattening
print("Columns after flattening:", df.columns)

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

# Label: 1 if next day's return > 1%, 0 if < -1%, else drop
mask = (df['next_return'] > 0.01) | (df['next_return'] < -0.01)
df = df[mask]
df['target'] = (df['next_return'] > 0.01).astype(int)

# Drop last row (no target)
df = df.iloc[:-1]

# Save to CSV
df.to_csv(OUTPUT_CSV)
print(f"Feature CSV saved to {OUTPUT_CSV}") 