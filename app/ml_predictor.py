from .news_sentiment import NewsSentimentAnalyzer
import joblib
import pandas as pd
from typing import Optional, Dict, Tuple
from log import logger

class MLPredictor:
    def __init__(self):
        self.model_3d = joblib.load('rf_model_target_3d.pkl')
        self.model_5d = joblib.load('rf_model_target_5d.pkl')
        self.news_analyzer = NewsSentimentAnalyzer()
        
    def predict(self, df: pd.DataFrame) -> Optional[Dict[str, Tuple[float, float]]]:
        try:
            # Get the symbol from the DataFrame (assuming it's in the index or a column)
            symbol = df.index.name if df.index.name else df.columns[0].split('_')[0]
            
            # Get news sentiment
            sentiment_data = self.news_analyzer.get_news_sentiment(symbol)
            
            # Add sentiment features to the DataFrame
            df['news_sentiment_score'] = sentiment_data['sentiment_score']
            df['news_article_count'] = sentiment_data['article_count']
            df['news_positive_ratio'] = sentiment_data['sentiment_distribution']['positive'] / max(1, sentiment_data['article_count'])
            df['news_negative_ratio'] = sentiment_data['sentiment_distribution']['negative'] / max(1, sentiment_data['article_count'])
            
            # Make predictions
            pred_3d = self.model_3d.predict_proba(df)[-1]
            pred_5d = self.model_5d.predict_proba(df)[-1]
            
            return {
                '3d': (pred_3d[1], pred_3d[0]),  # (up_prob, down_prob)
                '5d': (pred_5d[1], pred_5d[0])   # (up_prob, down_prob)
            }
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None 