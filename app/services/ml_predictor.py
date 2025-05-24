import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.model_3d = None
        self.model_5d = None
        self.feature_columns = None
        self.load_models()
    
    def load_models(self):
        """Load the trained Random Forest models"""
        try:
            model_dir = Path(__file__).parent.parent.parent
            self.model_3d = joblib.load(model_dir / 'rf_model_target_3d.pkl')
            self.model_5d = joblib.load(model_dir / 'rf_model_target_5d.pkl')
            
            # Load feature importance to get feature columns
            feature_importance = pd.read_csv(model_dir / 'feature_importance_target_3d.csv')
            self.feature_columns = feature_importance['feature'].tolist()
            
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ML models: {str(e)}")
            raise
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Calculate technical indicators
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_high'] = data['sma_20'] + (data['bb_std'] * 2)
        data['bb_low'] = data['sma_20'] - (data['bb_std'] * 2)
        data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['sma_20']
        
        # Stochastic Oscillator
        low_min = data['low'].rolling(window=14).min()
        high_max = data['high'].rolling(window=14).max()
        data['stoch_k'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # Moving Averages
        for period in [10, 20, 50, 200]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # Returns and Volatility
        for period in [1, 2, 3, 5, 10]:
            data[f'return_lag_{period}'] = data['close'].pct_change(periods=period)
        
        for period in [5, 10, 20]:
            data[f'volatility_{period}'] = data['close'].pct_change().rolling(window=period).std()
            data[f'momentum_{period}'] = data['close'].pct_change(periods=period)
        
        # Price Ratios
        data['close_open_ratio'] = data['close'] / data['open']
        data['close_high_ratio'] = data['close'] / data['high']
        data['close_low_ratio'] = data['close'] / data['low']
        data['high_low_ratio'] = data['high'] / data['low']
        
        # Calendar Features
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        return data[self.feature_columns]
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Make predictions using both 3-day and 5-day models
        
        Returns:
            Dict with prediction probabilities for both models
            Format: {'3d': (prob_up, prob_down), '5d': (prob_up, prob_down)}
        """
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Get latest data point
            latest_features = features.iloc[-1:].values
            
            # Make predictions
            prob_3d = self.model_3d.predict_proba(latest_features)[0]
            prob_5d = self.model_5d.predict_proba(latest_features)[0]
            
            return {
                '3d': (prob_3d[1], prob_3d[0]),  # (prob_up, prob_down)
                '5d': (prob_5d[1], prob_5d[0])   # (prob_up, prob_down)
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None 