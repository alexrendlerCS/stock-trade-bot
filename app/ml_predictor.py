# Now importing news sentiment since it's working
from .news_sentiment import NewsSentimentAnalyzer
import joblib
import pandas as pd
from typing import Optional, Dict, Tuple
import logging
from dotenv import load_dotenv
import ta
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import os

# Suppress scikit-learn version mismatch warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)
load_dotenv()

class MLPredictor:
    def __init__(self):
        try:
            # Use new retrained models from data/models directory
            models_dir = os.path.join('data', 'models')
            self.model_3d = joblib.load(os.path.join(models_dir, 'rf_model_target_3d_new.pkl'))
            self.model_5d = joblib.load(os.path.join(models_dir, 'rf_model_target_5d_new.pkl'))
            
            # Load feature list
            self.feature_cols = []
            try:
                with open(os.path.join(models_dir, 'model_features.txt'), 'r') as f:
                    self.feature_cols = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.feature_cols)} features")
            except FileNotFoundError:
                logger.warning("model_features.txt not found, using fallback feature list")
                self.feature_cols = self._get_fallback_features()
            
            # Initialize news sentiment analyzer
            try:
                self.news_analyzer = NewsSentimentAnalyzer()
                logger.info("News Sentiment Analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize News Sentiment Analyzer: {str(e)}")
                self.news_analyzer = None
            
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.model_3d = None
            self.model_5d = None
            self.feature_cols = []
            self.news_analyzer = None
        
    def _get_fallback_features(self):
        """Fallback feature list if file is not found"""
        return [
            'Adj Close', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'bb_width',
            'stoch_k', 'stoch_d', 'sma_10', 'ema_10', 'sma_20', 'ema_20', 'sma_50', 'ema_50',
            'sma_200', 'ema_200', 'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
            'return_lag_10', 'volatility_5', 'volatility_10', 'volatility_20', 'momentum_5',
            'momentum_10', 'momentum_20', 'close_open_ratio', 'close_high_ratio', 'close_low_ratio',
            'high_low_ratio', 'day_of_week', 'month'
        ]
        
    def predict(self, df: pd.DataFrame, symbol: str = None) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Make predictions using the ML models with optional news sentiment.
        
        Args:
            df: Price data DataFrame
            symbol: Stock symbol for news sentiment analysis
        """
        try:
            if self.model_3d is None or self.model_5d is None:
                logger.error("Models not loaded properly")
                return None
            
            logger.info(f"\nMaking prediction for {symbol if symbol else 'unknown symbol'}:")
            logger.info(f"Input data shape: {df.shape}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
                
            # Prepare features similar to training data
            logger.info("Preparing features...")
            features_df = self._prepare_features(df)
            
            if features_df is None or len(features_df) == 0:
                logger.error("Could not prepare features")
                return None
            
            # Log technical indicators for latest data point
            latest_data = features_df.iloc[-1]
            logger.info("\nLatest Technical Indicators:")
            logger.info(f"RSI: {latest_data.get('rsi', 'N/A'):.2f}")
            logger.info(f"MACD: {latest_data.get('macd', 'N/A'):.4f}")
            logger.info(f"MACD Signal: {latest_data.get('macd_signal', 'N/A'):.4f}")
            logger.info(f"Bollinger Band Width: {latest_data.get('bb_width', 'N/A'):.2f}")
            logger.info(f"Stochastic K: {latest_data.get('stoch_k', 'N/A'):.2f}")
            logger.info(f"Stochastic D: {latest_data.get('stoch_d', 'N/A'):.2f}")
            
            # Log momentum indicators
            logger.info("\nMomentum Indicators:")
            logger.info(f"5-day Momentum: {latest_data.get('momentum_5', 'N/A'):.2%}")
            logger.info(f"10-day Momentum: {latest_data.get('momentum_10', 'N/A'):.2%}")
            logger.info(f"20-day Momentum: {latest_data.get('momentum_20', 'N/A'):.2%}")
            
            # Log volatility
            logger.info("\nVolatility Metrics:")
            logger.info(f"5-day Volatility: {latest_data.get('volatility_5', 'N/A'):.2%}")
            logger.info(f"10-day Volatility: {latest_data.get('volatility_10', 'N/A'):.2%}")
            logger.info(f"20-day Volatility: {latest_data.get('volatility_20', 'N/A'):.2%}")
            
            # Add news sentiment features if available
            sentiment_score = 0.0
            if symbol and self.news_analyzer:
                logger.info("\nGetting news sentiment features...")
                sentiment_features = self._get_sentiment_features(symbol)
                if sentiment_features:
                    logger.info("News Sentiment Analysis:")
                    for feature_name, feature_value in sentiment_features.items():
                        features_df.loc[features_df.index[-1], feature_name] = feature_value
                        logger.info(f"  {feature_name}: {feature_value:.3f}")
                    sentiment_score = sentiment_features.get('news_sentiment_score', 0.0)
            
            # Get the latest feature vector with only the model features
            latest_features = features_df[self.feature_cols].iloc[-1:].fillna(0)
            
            # Make predictions
            logger.info("\nRunning ML models...")
            pred_3d = self.model_3d.predict_proba(latest_features)[0]
            pred_5d = self.model_5d.predict_proba(latest_features)[0]
            
            # Calculate confidence scores
            up_confidence_3d = pred_3d[1]
            down_confidence_3d = pred_3d[0]
            up_confidence_5d = pred_5d[1]
            down_confidence_5d = pred_5d[0]
            
            logger.info("\nConfidence Analysis:")
            logger.info("3-Day Prediction:")
            logger.info(f"  Upward Movement Confidence: {up_confidence_3d:.3f}")
            logger.info(f"  Downward Movement Confidence: {down_confidence_3d:.3f}")
            logger.info(f"  Sentiment Contribution: {sentiment_score:.3f}")
            
            logger.info("\n5-Day Prediction:")
            logger.info(f"  Upward Movement Confidence: {up_confidence_5d:.3f}")
            logger.info(f"  Downward Movement Confidence: {down_confidence_5d:.3f}")
            logger.info(f"  Sentiment Contribution: {sentiment_score:.3f}")
            
            # Log decision factors
            logger.info("\nDecision Factors:")
            if max(up_confidence_3d, down_confidence_3d) > 0.65:
                logger.info("[PASS] 3-day confidence exceeds threshold")
            else:
                logger.info(f"[FAIL] 3-day confidence ({max(up_confidence_3d, down_confidence_3d):.3f}) below threshold (0.65)")
                
            if abs(sentiment_score) > 0.2:
                logger.info("[PASS] Strong sentiment signal detected")
            else:
                logger.info(f"[FAIL] Weak sentiment signal ({sentiment_score:.3f})")
                
            if latest_data.get('volatility_5', 0) < 0.02:
                logger.info("[PASS] Low volatility environment")
            else:
                logger.info(f"[FAIL] High volatility ({latest_data.get('volatility_5', 0):.2%})")
            
            # Get sentiment info for the response
            sentiment_info = {}
            if symbol and self.news_analyzer:
                try:
                    sentiment_data = self.news_analyzer.get_news_sentiment(symbol, lookback_days=3)
                    sentiment_info = {
                        'sentiment_score': sentiment_data['sentiment_score'],
                        'news_count': sentiment_data['news_count'],
                        'sentiment_strength': sentiment_data['sentiment_strength']
                    }
                except Exception as e:
                    logger.warning(f"Could not get sentiment info: {str(e)}")
            
            result = {
                '3d': (pred_3d[1], pred_3d[0]),  # (up_prob, down_prob)
                '5d': (pred_5d[1], pred_5d[0])   # (up_prob, down_prob)
            }
            
            # Add sentiment info if available
            if sentiment_info:
                result['sentiment'] = sentiment_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _get_sentiment_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get news sentiment features for a symbol"""
        try:
            sentiment_data = self.news_analyzer.get_news_sentiment(symbol, lookback_days=7)
            if not sentiment_data:
                logger.warning(f"No sentiment data available for {symbol}")
                return None
                
            # Create sentiment features that can enhance predictions
            sentiment_features = {
                'news_sentiment_score': float(sentiment_data['sentiment_score']),
                'news_positive_ratio': float(sentiment_data['positive_ratio']),
                'news_negative_ratio': float(sentiment_data['negative_ratio']),
                'news_neutral_ratio': float(sentiment_data['neutral_ratio']),
                'news_volume': min(float(sentiment_data['news_count']) / 20.0, 1.0),  # Normalize to 0-1 (max 20 articles)
                'sentiment_strength': float(1.0 if sentiment_data['sentiment_strength'].startswith('strong') else 
                                         0.5 if sentiment_data['sentiment_strength'].startswith('weak') else 0.0)
            }
            
            # Ensure all values are valid floats
            for key, value in sentiment_features.items():
                if not isinstance(value, (int, float)) or pd.isna(value):
                    sentiment_features[key] = 0.0
            
            logger.info(f"Generated sentiment features for {symbol}: sentiment={sentiment_data['sentiment_score']:.3f}, volume={sentiment_data['news_count']}")
            return sentiment_features
            
        except Exception as e:
            logger.warning(f"Could not get sentiment features for {symbol}: {str(e)}")
            return None
    
    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare features to match the training data exactly.
        """
        try:
            features_df = df.copy()
            
            # Ensure we have the basic price columns
            required_cols = ['close', 'high', 'low', 'open', 'volume']
            for col in required_cols:
                if col not in features_df.columns:
                    logger.error(f"Missing required column: {col}")
                    return None
            
            # Capitalize column names to match training data
            features_df.columns = [col.capitalize() for col in features_df.columns]
            
            logger.info(f"Starting feature preparation with {len(features_df)} rows")
            
            # Add Adj Close (same as Close for now)
            features_df['Adj Close'] = features_df['Close']
            
            # Technical indicators
            logger.info("Adding technical indicators...")
            
            # RSI
            logger.info("  Calculating RSI...")
            features_df['rsi'] = ta.momentum.RSIIndicator(features_df['Close']).rsi()
            
            # MACD
            logger.info("  Calculating MACD...")
            macd = ta.trend.MACD(features_df['Close'])
            features_df['macd'] = macd.macd()
            features_df['macd_signal'] = macd.macd_signal()
            features_df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            logger.info("  Calculating Bollinger Bands...")
            bb = ta.volatility.BollingerBands(features_df['Close'])
            features_df['bb_high'] = bb.bollinger_hband()
            features_df['bb_low'] = bb.bollinger_lband()
            features_df['bb_width'] = features_df['bb_high'] - features_df['bb_low']
            
            # Stochastic Oscillator
            logger.info("  Calculating Stochastic Oscillator...")
            stoch = ta.momentum.StochasticOscillator(features_df['High'], features_df['Low'], features_df['Close'])
            features_df['stoch_k'] = stoch.stoch()
            features_df['stoch_d'] = stoch.stoch_signal()
            
            # Moving Averages
            logger.info("  Calculating Moving Averages...")
            for window in [10, 20, 50, 200]:
                features_df[f'sma_{window}'] = ta.trend.SMAIndicator(features_df['Close'], window=window).sma_indicator()
                features_df[f'ema_{window}'] = ta.trend.EMAIndicator(features_df['Close'], window=window).ema_indicator()
            
            # Lagged returns
            logger.info("Adding lagged returns...")
            returns = features_df['Close'].pct_change()
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'return_lag_{lag}'] = returns.shift(lag)
            
            # Volatility
            logger.info("Adding volatility features...")
            for window in [5, 10, 20]:
                features_df[f'volatility_{window}'] = returns.rolling(window=window).std()
            
            # Momentum
            logger.info("Adding momentum features...")
            for window in [5, 10, 20]:
                features_df[f'momentum_{window}'] = features_df['Close'].pct_change(periods=window)
            
            # Price ratios
            logger.info("Adding price ratios...")
            features_df['close_open_ratio'] = features_df['Close'] / features_df['Open']
            features_df['close_high_ratio'] = features_df['Close'] / features_df['High']
            features_df['close_low_ratio'] = features_df['Close'] / features_df['Low']
            features_df['high_low_ratio'] = features_df['High'] / features_df['Low']
            
            # Time features
            logger.info("Adding time features...")
            features_df['day_of_week'] = pd.to_datetime(features_df.index).dayofweek
            features_df['month'] = pd.to_datetime(features_df.index).month
            
            logger.info("Handling NaN values...")
            # Forward fill then backward fill to handle NaN values
            features_df = features_df.ffill().bfill()
            
            # Ensure we have all required features
            missing_features = [f for f in self.feature_cols if f not in features_df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    features_df[feature] = 0
                    logger.info(f"Added missing feature {feature} with default value 0")
            
            logger.info(f"Feature preparation completed. Final shape: {features_df.shape}")
            logger.info(f"Available features: {len([f for f in self.feature_cols if f in features_df.columns])}/{len(self.feature_cols)}")
            
            if len(features_df) == 0:
                logger.error("No data remaining after feature preparation")
                return None
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None 