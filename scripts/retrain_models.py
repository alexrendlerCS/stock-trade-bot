import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_features(df):
    """
    Create all the features that match the original training data
    """
    features_df = df.copy()
    
    # Ensure we have the basic columns
    required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in required_cols:
        if col not in features_df.columns:
            print(f"Missing required column: {col}")
            return None
    
    print(f"Starting with {len(features_df)} rows")
    
    # Technical indicators
    print("Adding technical indicators...")
    
    # RSI
    features_df['rsi'] = ta.momentum.RSIIndicator(features_df['Close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(features_df['Close'])
    features_df['macd'] = macd.macd()
    features_df['macd_signal'] = macd.macd_signal()
    features_df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(features_df['Close'])
    features_df['bb_high'] = bb.bollinger_hband()
    features_df['bb_low'] = bb.bollinger_lband()
    features_df['bb_width'] = features_df['bb_high'] - features_df['bb_low']
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(features_df['High'], features_df['Low'], features_df['Close'])
    features_df['stoch_k'] = stoch.stoch()
    features_df['stoch_d'] = stoch.stoch_signal()
    
    # Moving Averages
    for window in [10, 20, 50, 200]:
        features_df[f'sma_{window}'] = ta.trend.SMAIndicator(features_df['Close'], window=window).sma_indicator()
        features_df[f'ema_{window}'] = ta.trend.EMAIndicator(features_df['Close'], window=window).ema_indicator()
    
    # Returns and targets
    print("Adding returns and targets...")
    features_df['next_return'] = features_df['Close'].pct_change().shift(-1)
    features_df['return_3d'] = features_df['Close'].pct_change(periods=3).shift(-3)
    features_df['return_5d'] = features_df['Close'].pct_change(periods=5).shift(-5)
    
    # Target variables (1 if return > 0, 0 otherwise)
    features_df['target_3d'] = (features_df['return_3d'] > 0).astype(int)
    features_df['target_5d'] = (features_df['return_5d'] > 0).astype(int)
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        features_df[f'return_lag_{lag}'] = features_df['Close'].pct_change().shift(lag)
    
    # Volatility (rolling standard deviation of returns)
    returns = features_df['Close'].pct_change()
    for window in [5, 10, 20]:
        features_df[f'volatility_{window}'] = returns.rolling(window=window).std()
    
    # Momentum (price change over different periods)
    for window in [5, 10, 20]:
        features_df[f'momentum_{window}'] = features_df['Close'].pct_change(periods=window)
    
    # Price ratios
    features_df['close_open_ratio'] = features_df['Close'] / features_df['Open']
    features_df['close_high_ratio'] = features_df['Close'] / features_df['High']
    features_df['close_low_ratio'] = features_df['Close'] / features_df['Low']
    features_df['high_low_ratio'] = features_df['High'] / features_df['Low']
    
    # Time features
    features_df['day_of_week'] = pd.to_datetime(features_df.index).dayofweek
    features_df['month'] = pd.to_datetime(features_df.index).month
    
    print(f"Features created. Shape: {features_df.shape}")
    
    return features_df

def download_and_prepare_data(symbols, start_date, end_date):
    """
    Download and prepare training data for multiple symbols
    """
    all_data = []
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        try:
            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                print(f"No data for {symbol}")
                continue
            
            # Clean column names
            df.columns = [col[0] if hasattr(col, '__len__') and len(col) > 1 else col for col in df.columns]
            
            # Add ticker column
            df['ticker'] = symbol
            
            # Create features
            features_df = create_comprehensive_features(df)
            
            if features_df is not None:
                all_data.append(features_df)
                print(f"Added {len(features_df)} rows for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    if not all_data:
        print("No data collected!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=False)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def train_models(df):
    """
    Train Random Forest models for 3-day and 5-day predictions
    """
    print("Preparing training data...")
    
    # Feature columns (exclude target and metadata columns)
    feature_cols = [col for col in df.columns if col not in [
        'Close', 'High', 'Low', 'Open', 'Volume', 'next_return', 
        'return_3d', 'return_5d', 'target_3d', 'target_5d', 'ticker'
    ]]
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
    
    # Remove rows with NaN values
    df_clean = df[feature_cols + ['target_3d', 'target_5d']].dropna()
    print(f"After removing NaN: {len(df_clean)} rows")
    
    if len(df_clean) < 100:
        print("Not enough clean data for training!")
        return None, None
    
    X = df_clean[feature_cols]
    y_3d = df_clean['target_3d']
    y_5d = df_clean['target_5d']
    
    print(f"Training data shape: {X.shape}")
    print(f"3d target distribution: {y_3d.value_counts().to_dict()}")
    print(f"5d target distribution: {y_5d.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_3d_train, y_3d_test = train_test_split(
        X, y_3d, test_size=0.2, random_state=42, stratify=y_3d
    )
    _, _, y_5d_train, y_5d_test = train_test_split(
        X, y_5d, test_size=0.2, random_state=42, stratify=y_5d
    )
    
    # Train 3-day model
    print("Training 3-day model...")
    model_3d = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model_3d.fit(X_train, y_3d_train)
    
    # Train 5-day model
    print("Training 5-day model...")
    model_5d = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model_5d.fit(X_train, y_5d_train)
    
    # Evaluate models
    print("\n=== 3-Day Model Performance ===")
    y_3d_pred = model_3d.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_3d_test, y_3d_pred):.3f}")
    print(classification_report(y_3d_test, y_3d_pred))
    
    print("\n=== 5-Day Model Performance ===")
    y_5d_pred = model_5d.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_5d_test, y_5d_pred):.3f}")
    print(classification_report(y_5d_test, y_5d_pred))
    
    # Feature importance
    feature_importance_3d = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_3d.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_5d = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_5d.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features for 3-day model:")
    print(feature_importance_3d.head(10))
    
    print("\nTop 10 features for 5-day model:")
    print(feature_importance_5d.head(10))
    
    # Save feature importance
    feature_importance_3d.to_csv('feature_importance_target_3d_new.csv', index=False)
    feature_importance_5d.to_csv('feature_importance_target_5d_new.csv', index=False)
    
    return model_3d, model_5d, feature_cols

def main():
    # Define symbols and date range
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    print("Starting model retraining...")
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Download and prepare data
    df = download_and_prepare_data(symbols, start_date, end_date)
    
    if df is None:
        print("Failed to prepare data!")
        return
    
    # Train models
    model_3d, model_5d, feature_cols = train_models(df)
    
    if model_3d is None or model_5d is None:
        print("Failed to train models!")
        return
    
    # Save models
    print("Saving models...")
    joblib.dump(model_3d, 'rf_model_target_3d_new.pkl')
    joblib.dump(model_5d, 'rf_model_target_5d_new.pkl')
    
    # Save feature list
    with open('model_features.txt', 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    print("Model retraining completed successfully!")
    print("Saved files:")
    print("- rf_model_target_3d_new.pkl")
    print("- rf_model_target_5d_new.pkl") 
    print("- feature_importance_target_3d_new.csv")
    print("- feature_importance_target_5d_new.csv")
    print("- model_features.txt")

if __name__ == "__main__":
    main() 