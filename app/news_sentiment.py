from newsapi import NewsApiClient
from transformers import pipeline
from datetime import datetime, timedelta
import pandas as pd
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the NewsSentimentAnalyzer with NewsAPI and sentiment analysis pipeline.
        
        Args:
            api_key: NewsAPI key. If None, will try to get from environment variable NEWS_API_KEY
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("NewsAPI key is required. Set it as NEWS_API_KEY environment variable or pass it to the constructor.")
        
        self.newsapi = NewsApiClient(api_key=self.api_key)
        # Initialize the sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        
    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Get news sentiment for a given symbol over the specified number of days.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days to look back for news
            
        Returns:
            Dictionary containing sentiment metrics
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get news articles
            articles = self.newsapi.get_everything(
                q=symbol,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            VOLATILITY_KEYWORDS = [
                "tariff", "sanction", "fed", "interest rate", "war", "regulation", "lawsuit",
                "SEC", "ban", "inflation", "recession", "bankruptcy", "crisis", "default",
                "subpoena", "investigation", "fine", "probe", "antitrust", "strike", "shutdown"
            ]
            volatility_count = 0
            volatility_articles = []
            
            if not articles['articles']:
                logger.warning(f"No news articles found for {symbol}")
                return {
                    'sentiment_score': 0,
                    'article_count': 0,
                    'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                    'volatility_article_count': 0,
                    'volatility_articles': []
                }
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles['articles']:
                text = f"{article['title']} {article['description']}".lower() if article['title'] and article['description'] else ""
                found_keywords = [kw for kw in VOLATILITY_KEYWORDS if kw in text]
                if found_keywords:
                    volatility_count += 1
                    volatility_articles.append({
                        "title": article['title'],
                        "keywords": found_keywords
                    })
                if text:
                    sentiment = self.sentiment_analyzer(text)[0]
                    sentiments.append(sentiment)
            
            # Calculate sentiment metrics
            sentiment_scores = {
                'POS': 1,
                'NEU': 0,
                'NEG': -1
            }
            
            sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
            total_score = 0
            
            for sentiment in sentiments:
                label = sentiment['label']
                score = sentiment['score']
                
                if label == 'POS':
                    sentiment_distribution['positive'] += 1
                    total_score += score
                elif label == 'NEU':
                    sentiment_distribution['neutral'] += 1
                else:  # NEG
                    sentiment_distribution['negative'] += 1
                    total_score -= score
            
            # Calculate average sentiment score
            avg_sentiment = total_score / len(sentiments) if sentiments else 0
            
            return {
                'sentiment_score': avg_sentiment,
                'article_count': len(sentiments),
                'sentiment_distribution': sentiment_distribution,
                'volatility_article_count': volatility_count,
                'volatility_articles': volatility_articles
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return {
                'sentiment_score': 0,
                'article_count': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'volatility_article_count': 0,
                'volatility_articles': []
            }
    
    def get_batch_sentiment(self, symbols: List[str], days: int = 7) -> pd.DataFrame:
        """
        Get news sentiment for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            days: Number of days to look back for news
            
        Returns:
            DataFrame with sentiment metrics for each symbol
        """
        results = []
        for symbol in symbols:
            sentiment_data = self.get_news_sentiment(symbol, days)
            sentiment_data['symbol'] = symbol
            results.append(sentiment_data)
        
        return pd.DataFrame(results) 