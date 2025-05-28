import sys
import os
import time
from datetime import datetime
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.news_sentiment import NewsSentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_finlight_integration():
    """Test the Finlight API integration"""
    try:
        logger.info("Initializing News Sentiment Analyzer...")
        analyzer = NewsSentimentAnalyzer()
        
        # Test API key is set
        if not analyzer.api_key:
            logger.error("❌ Finlight API key not found in environment variables")
            return False
        
        logger.info("✓ API key found")
        
        # Test stocks to analyze
        test_stocks = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in test_stocks:
            logger.info(f"\nTesting sentiment analysis for {symbol}...")
            
            # Get sentiment
            sentiment = analyzer.get_news_sentiment(symbol, lookback_days=3)
            
            if sentiment:
                logger.info(f"✓ Successfully retrieved sentiment data:")
                logger.info(f"  - Sentiment Score: {sentiment['sentiment_score']:.3f}")
                logger.info(f"  - Sentiment Strength: {sentiment['sentiment_strength']}")
                logger.info(f"  - News Count: {sentiment['news_count']}")
                logger.info(f"  - Recent Sentiment: {sentiment['recent_sentiment']:.3f}")
            else:
                logger.error(f"❌ Failed to get sentiment for {symbol}")
                return False
            
            # Check API usage
            usage = analyzer.get_api_usage()
            logger.info("\nAPI Usage Statistics:")
            logger.info(f"  - Daily Calls: {usage['daily_calls']}")
            logger.info(f"  - Remaining Calls: {usage['remaining_calls']}")
            logger.info(f"  - Last Reset: {usage['last_reset']}")
            
            # Small delay between requests
            time.sleep(1)
        
        logger.info("\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting Finlight API integration test...")
    logger.info("=" * 50)
    
    # Run the test
    success = test_finlight_integration()
    
    if success:
        logger.info("\n✅ All tests passed! Finlight integration is working correctly.")
    else:
        logger.error("\n❌ Some tests failed. Please check the logs above for details.") 