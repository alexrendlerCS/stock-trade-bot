import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
from dotenv import load_dotenv
from app.core.config import settings

logger = logging.getLogger(__name__)
load_dotenv()

class NewsSentimentAnalyzer:
    """
    News sentiment analyzer that fetches and analyzes financial news using Finlight API
    """
    
    def __init__(self):
        """Initialize the news sentiment analyzer with API configuration"""
        self.api_url = "https://api.finlight.me/v1/articles"
        self.api_key = settings.FINLIGHT_API_KEY
        self.cache = {}
        self.api_calls = 0
        self.last_reset = datetime.now()
        self.daily_calls = 0
        self.max_daily_calls = 166  # ~5,000/30 days for free tier
        
        # Cache configuration
        self.cache_duration = {
            'short_term': 1800,  # 30 minutes for recent news
            'long_term': 7200    # 2 hours for older news
        }
        
        if not self.api_key:
            logger.error("No Finlight API key found in configuration")
        else:
            logger.info("Finlight API key loaded successfully")
    
    def _should_reset_counter(self) -> bool:
        """Check if we should reset the daily API call counter"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_calls = 0
            self.last_reset = now
            return True
        return False
    
    def _can_make_api_call(self) -> bool:
        """Check if we can make an API call within our limits"""
        self._should_reset_counter()
        return self.daily_calls < self.max_daily_calls
    
    def _get_cache_key(self, symbol: str, lookback_days: int) -> str:
        """Generate a cache key for a specific request"""
        return f"{symbol}_{lookback_days}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if a cache entry is still valid"""
        if not cache_entry or 'timestamp' not in cache_entry:
            return False
            
        age = time.time() - cache_entry['timestamp']
        # Use shorter cache duration for recent news
        if cache_entry.get('lookback_days', 7) <= 1:
            return age < self.cache_duration['short_term']
        return age < self.cache_duration['long_term']
    
    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse datetime string from API response"""
        try:
            # Try parsing with milliseconds
            return datetime.strptime(date_str.split('.')[0], '%Y-%m-%dT%H:%M:%S')
        except (ValueError, AttributeError):
            try:
                # Try parsing without milliseconds
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            except (ValueError, AttributeError):
                logger.warning(f"Could not parse datetime: {date_str}")
                return None
    
    def _analyze_article_sentiment(self, article: Dict) -> float:
        """Analyze sentiment of a single article using title and summary"""
        text = article['title']
        if article.get('summary'):
            text += ' ' + article['summary']
            
        # Positive keywords
        positive_words = ['surge', 'jump', 'soar', 'gain', 'rise', 'climb', 'boost', 'bullish', 'upgrade', 
                         'growth', 'profit', 'success', 'positive', 'strong', 'beat', 'exceed', 'outperform']
        
        # Negative keywords
        negative_words = ['fall', 'drop', 'decline', 'slip', 'tumble', 'crash', 'bearish', 'downgrade',
                         'loss', 'weak', 'risk', 'threat', 'miss', 'below', 'underperform', 'warning']
        
        text = text.lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 1.0
        elif negative_count > positive_count:
            return -1.0
        return 0.0

    def get_news_sentiment(self, symbol: str, lookback_days: int = 3) -> Optional[Dict]:
        """
        Get news sentiment for a symbol with caching and rate limiting
        Returns: Dict with sentiment_score, sentiment_strength, news_count, etc.
        """
        if not self.api_key:
            logger.error("No Finlight API key available")
            return self._get_default_sentiment()
            
        cache_key = self._get_cache_key(symbol, lookback_days)
        
        # Check cache first
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.debug(f"Using cached news data for {symbol}")
            return self.cache[cache_key]['data']
        
        # Check API limits
        if not self._can_make_api_call():
            logger.warning(f"API rate limit reached, using default sentiment for {symbol}")
            return self._handle_rate_limit(cache_key)
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Make API request
            params = {
                'query': symbol,
                'from': start_str,
                'to': end_str,
                'language': 'en',
                'pageSize': 20,
                'order': 'DESC'
            }
            
            logger.info(f"Making Finlight API request for {symbol}")
            
            response = requests.get(
                self.api_url,
                params=params,
                headers={'X-API-KEY': self.api_key},
                timeout=10
            )
            
            self.daily_calls += 1
            
            if response.status_code == 200:
                articles_data = response.json()
                articles = articles_data.get('articles', [])
                
                logger.info(f"Retrieved {len(articles)} articles for {symbol}")
                
                # Calculate sentiment metrics
                total_articles = len(articles)
                if total_articles == 0:
                    logger.warning(f"No articles found for {symbol}")
                    return self._get_default_sentiment()
                
                # Calculate sentiment scores
                sentiments = [self._analyze_article_sentiment(article) for article in articles]
                positive_count = sum(1 for s in sentiments if s > 0)
                negative_count = sum(1 for s in sentiments if s < 0)
                neutral_count = sum(1 for s in sentiments if s == 0)
                
                # Calculate ratios
                positive_ratio = positive_count / total_articles
                negative_ratio = negative_count / total_articles
                neutral_ratio = neutral_count / total_articles
                
                # Calculate final sentiment score (-1 to 1)
                sentiment_score = sum(sentiments) / len(sentiments)
                
                # Determine sentiment strength
                if abs(sentiment_score) < 0.2:
                    strength = 'neutral'
                elif abs(sentiment_score) < 0.5:
                    strength = 'weak_positive' if sentiment_score > 0 else 'weak_negative'
                else:
                    strength = 'strong_positive' if sentiment_score > 0 else 'strong_negative'
                
                # Create result with all required fields
                result = {
                    'sentiment_score': sentiment_score,
                    'sentiment_strength': strength,
                    'news_count': total_articles,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'positive_ratio': positive_ratio,
                    'negative_ratio': negative_ratio,
                    'neutral_ratio': neutral_ratio,
                    'timestamp': time.time()
                }
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time(),
                    'lookback_days': lookback_days
                }
                
                return result
            
            else:
                logger.error(f"API request failed with status {response.status_code}")
                return self._handle_error(cache_key)
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return self._handle_error(cache_key)
    
    def _handle_rate_limit(self, cache_key: str) -> Dict:
        """Handle rate limit by returning cached data or default values"""
        if cache_key in self.cache:
            logger.info(f"Using cached data due to rate limit")
            return self.cache[cache_key]['data']
        return self._get_default_sentiment()
    
    def _handle_error(self, cache_key: str) -> Dict:
        """Handle API errors by returning default sentiment and caching it briefly"""
        result = {
            'sentiment_score': 0.0,
            'sentiment_strength': 'neutral',
            'news_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 0.0,
            'timestamp': time.time()
        }
        
        # Cache error result for a short time
        self.cache[cache_key] = {
            'data': result,
            'timestamp': time.time(),
            'lookback_days': 1  # Short cache for errors
        }
        
        return result
    
    def _get_default_sentiment(self) -> Dict:
        """Return default sentiment when no data is available"""
        return {
            'sentiment_score': 0.0,
            'sentiment_strength': 'neutral',
            'news_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 0.0,
            'timestamp': time.time()
        }
    
    def get_api_usage(self) -> Dict:
        """Get current API usage statistics"""
        return {
            'daily_calls': self.daily_calls,
            'remaining_calls': self.max_daily_calls - self.daily_calls,
            'last_reset': self.last_reset.isoformat()
        }
    
    def get_multiple_sentiments(self, symbols: List[str], lookback_days: int = 7) -> Dict[str, Dict]:
        """Get news sentiment for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_news_sentiment(symbol, lookback_days)
                # Small delay to respect API rate limits
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
                results[symbol] = self._get_default_sentiment()
        
        return results 