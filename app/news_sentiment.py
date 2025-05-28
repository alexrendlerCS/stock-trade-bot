import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class NewsSentimentAnalyzer:
    """
    News sentiment analyzer that fetches and analyzes financial news using Finlight API
    """
    
    def __init__(self):
        """Initialize the news sentiment analyzer"""
        self.api_key = os.getenv('FINLIGHT_API_KEY')
        self.base_url = 'https://api.finlight.me/v1'
        self.cache = {}
        self.daily_calls = 0
        self.last_reset = datetime.now()
        self.max_daily_calls = 166  # ~5,000/30 days for free tier
        
        # Cache configuration
        self.cache_duration = {
            'short_term': 1800,  # 30 minutes for recent news
            'long_term': 7200    # 2 hours for older news
        }
        
        if not self.api_key:
            logger.error("No FINLIGHT_API_KEY found in environment variables")
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
            logger.warning(f"API call limit reached for today. Using cached data if available.")
            return self.cache.get(cache_key, {}).get('data', self._get_default_sentiment())
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Make API request for articles
            headers = {'X-API-KEY': self.api_key}
            params = {
                'query': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'pageSize': 20,
                'order': 'DESC'
            }
            
            response = requests.get(
                f"{self.base_url}/articles",
                headers=headers,
                params=params,
                timeout=10
            )
            
            self.daily_calls += 1
            
            if response.status_code == 200:
                articles_data = response.json()
                articles = articles_data.get('articles', [])
                
                # Calculate sentiment metrics
                total_articles = len(articles)
                if total_articles == 0:
                    return self._get_default_sentiment()
                
                # Process sentiment from articles
                positive_count = sum(1 for a in articles if a.get('sentiment') == 'positive')
                negative_count = sum(1 for a in articles if a.get('sentiment') == 'negative')
                neutral_count = sum(1 for a in articles if a.get('sentiment') == 'neutral')
                
                # Calculate sentiment score (-1 to 1)
                sentiment_scores = []
                for article in articles:
                    sentiment = article.get('sentiment')
                    confidence = article.get('confidence', 0.5)
                    
                    if sentiment == 'positive':
                        sentiment_scores.append(confidence)
                    elif sentiment == 'negative':
                        sentiment_scores.append(-confidence)
                    else:
                        sentiment_scores.append(0)
                
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                # Determine sentiment strength
                if abs(avg_sentiment) < 0.2:
                    strength = 'neutral'
                elif abs(avg_sentiment) < 0.5:
                    strength = 'weak ' + ('positive' if avg_sentiment > 0 else 'negative')
                else:
                    strength = 'strong ' + ('positive' if avg_sentiment > 0 else 'negative')
                
                # Calculate recent sentiment (last 24h)
                recent_articles = []
                for article in articles:
                    pub_date = self._parse_datetime(article.get('publishDate', ''))
                    if pub_date and pub_date > datetime.now() - timedelta(days=1):
                        recent_articles.append(article)
                
                recent_sentiment = avg_sentiment if recent_articles else 0
                
                result = {
                    'sentiment_score': avg_sentiment,
                    'sentiment_strength': strength,
                    'news_count': total_articles,
                    'positive_ratio': positive_count / total_articles,
                    'negative_ratio': negative_count / total_articles,
                    'news_volume': total_articles,
                    'recent_sentiment': recent_sentiment
                }
                
                # Cache the results
                self.cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time(),
                    'lookback_days': lookback_days
                }
                
                return result
                
            elif response.status_code == 429:  # Rate limit
                logger.warning(f"Rate limit reached for Finlight API")
                return self._handle_rate_limit(cache_key)
            else:
                logger.error(f"Error fetching news for {symbol}: {response.status_code}")
                return self._handle_error(cache_key)
                
        except Exception as e:
            logger.error(f"Error in news sentiment analysis for {symbol}: {str(e)}")
            return self._handle_error(cache_key)
    
    def _handle_rate_limit(self, cache_key: str) -> Dict:
        """Handle rate limit by returning cached data or default values"""
        if cache_key in self.cache:
            logger.info(f"Using cached data due to rate limit")
            return self.cache[cache_key]['data']
        return self._get_default_sentiment()
    
    def _handle_error(self, cache_key: str) -> Dict:
        """Handle errors by returning cached data or default values"""
        if cache_key in self.cache:
            return self.cache[cache_key]['data']
        return self._get_default_sentiment()
    
    def _get_default_sentiment(self) -> Dict:
        """Return default sentiment values when no data is available"""
        return {
            'sentiment_score': 0.0,
            'sentiment_strength': 'neutral',
            'news_count': 0,
            'positive_ratio': 0.5,
            'negative_ratio': 0.5,
            'news_volume': 0,
            'recent_sentiment': 0.0
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