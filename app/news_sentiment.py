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
    News sentiment analyzer that fetches and analyzes financial news
    """
    
    def __init__(self):
        """Initialize the news sentiment analyzer"""
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')  # Optional NewsAPI key
        self.finnhub_token = os.getenv('FINNHUB_TOKEN')  # Optional Finnhub token
        
        # Initialize sentiment analysis
        self.sentiment_analyzer = None
        self._init_sentiment_analyzer()
        
        logger.info("News Sentiment Analyzer initialized")
    
    def _init_sentiment_analyzer(self):
        """Initialize the sentiment analysis model"""
        try:
            # Try to use FinBERT (financial domain specific BERT)
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import pipeline
            
            # Use FinBERT for financial sentiment analysis
            model_name = "ProsusAI/finbert"
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=-1  # Use CPU
            )
            logger.info("FinBERT sentiment analyzer loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {str(e)}")
            try:
                # Fallback to general sentiment analysis
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1
                )
                logger.info("Fallback sentiment analyzer loaded")
            except Exception as e2:
                logger.error(f"Could not load any sentiment analyzer: {str(e2)}")
                self.sentiment_analyzer = None
    
    def get_news_sentiment(self, symbol: str, lookback_days: int = 7) -> Dict:
        """
        Get news sentiment for a stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            lookback_days: Number of days to look back for news
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            news_data = self._fetch_news(symbol, lookback_days)
            
            if not news_data:
                logger.warning(f"No news data found for {symbol}")
                return self._default_sentiment()
            
            sentiment_results = self._analyze_sentiment(news_data)
            
            return {
                'symbol': symbol,
                'news_count': len(news_data),
                'sentiment_score': sentiment_results['overall_sentiment'],
                'positive_ratio': sentiment_results['positive_ratio'],
                'negative_ratio': sentiment_results['negative_ratio'],
                'neutral_ratio': sentiment_results['neutral_ratio'],
                'sentiment_strength': sentiment_results['sentiment_strength'],
                'recent_sentiment': sentiment_results['recent_sentiment'],
                'news_volume': sentiment_results['news_volume'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return self._default_sentiment()
    
    def _fetch_news(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news articles for a symbol"""
        news_articles = []
        
        # Try Alpha Vantage News API first
        if self.alpha_vantage_key:
            av_news = self._fetch_alpha_vantage_news(symbol, lookback_days)
            news_articles.extend(av_news)
        
        # Try Finnhub News API
        if self.finnhub_token and len(news_articles) < 10:
            finnhub_news = self._fetch_finnhub_news(symbol, lookback_days)
            news_articles.extend(finnhub_news)
        
        # Try NewsAPI as fallback
        if self.news_api_key and len(news_articles) < 5:
            newsapi_news = self._fetch_newsapi_news(symbol, lookback_days)
            news_articles.extend(newsapi_news)
        
        # Remove duplicates based on title similarity
        news_articles = self._deduplicate_news(news_articles)
        
        return news_articles[:50]  # Limit to 50 articles
    
    def _fetch_alpha_vantage_news(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news from Alpha Vantage API"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'time_from': start_date.strftime('%Y%m%dT%H%M'),
                'time_to': end_date.strftime('%Y%m%dT%H%M'),
                'limit': 50,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' not in data:
                logger.warning(f"No news feed in Alpha Vantage response for {symbol}")
                return []
            
            articles = []
            for item in data['feed']:
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', 'Alpha Vantage'),
                    'published': item.get('time_published', ''),
                    'url': item.get('url', ''),
                    'text': f"{item.get('title', '')} {item.get('summary', '')}"
                })
            
            logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news for {symbol}: {str(e)}")
            return []
    
    def _fetch_finnhub_news(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news from Finnhub API"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_token
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for item in data:
                articles.append({
                    'title': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', 'Finnhub'),
                    'published': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                    'url': item.get('url', ''),
                    'text': f"{item.get('headline', '')} {item.get('summary', '')}"
                })
            
            logger.info(f"Fetched {len(articles)} articles from Finnhub for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {str(e)}")
            return []
    
    def _fetch_newsapi_news(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get company name for better search results
            company_names = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google Alphabet',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'META': 'Meta Facebook',
                'NVDA': 'NVIDIA',
                'NFLX': 'Netflix'
            }
            
            search_term = f"{company_names.get(symbol, symbol)} stock"
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': search_term,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for item in data.get('articles', []):
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('description', ''),
                    'source': item.get('source', {}).get('name', 'NewsAPI'),
                    'published': item.get('publishedAt', ''),
                    'url': item.get('url', ''),
                    'text': f"{item.get('title', '')} {item.get('description', '')}"
                })
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news for {symbol}: {str(e)}")
            return []
    
    def _deduplicate_news(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            # Simple deduplication - could be improved with more sophisticated similarity
            title_words = set(title.split())
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                # If 80% of words overlap, consider it a duplicate
                overlap = len(title_words & seen_words)
                if overlap > 0 and overlap / max(len(title_words), len(seen_words)) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate and title:
                unique_articles.append(article)
                seen_titles.add(title)
        
        return unique_articles
    
    def _analyze_sentiment(self, news_articles: List[Dict]) -> Dict:
        """Analyze sentiment of news articles"""
        if not news_articles or not self.sentiment_analyzer:
            return {
                'overall_sentiment': 0.0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34,
                'sentiment_strength': 0.0,
                'recent_sentiment': 0.0,
                'news_volume': 0
            }
        
        sentiments = []
        recent_sentiments = []  # Last 2 days
        
        recent_cutoff = datetime.now() - timedelta(days=2)
        
        for article in news_articles:
            try:
                text = article.get('text', '')[:512]  # Limit text length
                if not text.strip():
                    continue
                
                # Get sentiment
                result = self.sentiment_analyzer(text)
                
                if isinstance(result, list) and len(result) > 0:
                    sentiment_item = result[0]
                    label = sentiment_item['label'].upper()
                    score = sentiment_item['score']
                    
                    # Convert to numeric sentiment (-1 to 1)
                    if 'POSITIVE' in label or label == 'LABEL_2':
                        sentiment_score = score
                    elif 'NEGATIVE' in label or label == 'LABEL_0':
                        sentiment_score = -score
                    else:  # NEUTRAL or LABEL_1
                        sentiment_score = 0.0
                    
                    sentiments.append(sentiment_score)
                    
                    # Check if recent article
                    try:
                        pub_date = datetime.fromisoformat(article.get('published', '').replace('Z', '+00:00'))
                        if pub_date.replace(tzinfo=None) >= recent_cutoff:
                            recent_sentiments.append(sentiment_score)
                    except:
                        # If can't parse date, assume it's recent
                        recent_sentiments.append(sentiment_score)
                
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for article: {str(e)}")
                continue
        
        if not sentiments:
            return {
                'overall_sentiment': 0.0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34,
                'sentiment_strength': 0.0,
                'recent_sentiment': 0.0,
                'news_volume': len(news_articles)
            }
        
        # Calculate metrics
        overall_sentiment = sum(sentiments) / len(sentiments)
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        total_count = len(sentiments)
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        neutral_ratio = neutral_count / total_count
        
        # Sentiment strength (average absolute value)
        sentiment_strength = sum(abs(s) for s in sentiments) / len(sentiments)
        
        # Recent sentiment (last 2 days)
        recent_sentiment = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else overall_sentiment
        
        return {
            'overall_sentiment': overall_sentiment,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'sentiment_strength': sentiment_strength,
            'recent_sentiment': recent_sentiment,
            'news_volume': len(news_articles)
        }
    
    def _default_sentiment(self) -> Dict:
        """Return default neutral sentiment when no data is available"""
        return {
            'symbol': '',
            'news_count': 0,
            'sentiment_score': 0.0,
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34,
            'sentiment_strength': 0.0,
            'recent_sentiment': 0.0,
            'news_volume': 0,
            'timestamp': datetime.now().isoformat()
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
                results[symbol] = self._default_sentiment()
        
        return results 