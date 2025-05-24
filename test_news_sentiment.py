import os
from app.news_sentiment import NewsSentimentAnalyzer
import json

def test_news_sentiment():
    print("Testing News Sentiment Analyzer...")
    
    # Test 1: Initialize the analyzer
    print("\n1. Initializing sentiment analyzer...")
    try:
        analyzer = NewsSentimentAnalyzer()
        print("✅ News Sentiment Analyzer initialized successfully")
        
        if analyzer.sentiment_analyzer is None:
            print("⚠️  Warning: No sentiment model loaded (this is expected if no API keys are set)")
            print("   The analyzer will still work but return neutral sentiment")
        else:
            print("✅ Sentiment model loaded successfully")
            
    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        return False
    
    # Test 2: Test sentiment analysis with sample text
    print("\n2. Testing sentiment analysis...")
    if analyzer.sentiment_analyzer:
        try:
            # Test with different sentiment texts
            test_texts = [
                "Apple stock surges on strong quarterly earnings and positive outlook",
                "Tesla faces regulatory scrutiny and production challenges",
                "Microsoft announces new AI partnership, boosting investor confidence"
            ]
            
            for text in test_texts:
                print(f"\nAnalyzing: '{text[:60]}...'")
                # Simulate the sentiment analysis process
                result = analyzer.sentiment_analyzer(text)
                print(f"Result: {result}")
                
        except Exception as e:
            print(f"❌ Sentiment analysis failed: {str(e)}")
            return False
    else:
        print("⚠️  Skipping sentiment test (no model loaded)")
    
    # Test 3: Test news fetching (will work even without API keys by returning default sentiment)
    print("\n3. Testing news sentiment retrieval...")
    try:
        # Test with AAPL
        sentiment_result = analyzer.get_news_sentiment('AAPL', lookback_days=7)
        
        print("✅ News sentiment retrieval successful")
        print(f"Symbol: {sentiment_result['symbol']}")
        print(f"News count: {sentiment_result['news_count']}")
        print(f"Sentiment score: {sentiment_result['sentiment_score']:.3f}")
        print(f"Positive ratio: {sentiment_result['positive_ratio']:.3f}")
        print(f"Negative ratio: {sentiment_result['negative_ratio']:.3f}")
        print(f"Neutral ratio: {sentiment_result['neutral_ratio']:.3f}")
        
        if sentiment_result['news_count'] == 0:
            print("⚠️  No news articles found (expected without API keys)")
            print("   This will return neutral sentiment by default")
        
    except Exception as e:
        print(f"❌ News sentiment retrieval failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test multiple symbols
    print("\n4. Testing multiple symbols...")
    try:
        symbols = ['AAPL', 'MSFT', 'TSLA']
        multi_results = analyzer.get_multiple_sentiments(symbols, lookback_days=5)
        
        print("✅ Multiple symbols test successful")
        for symbol, result in multi_results.items():
            print(f"{symbol}: sentiment={result['sentiment_score']:.3f}, news_count={result['news_count']}")
            
    except Exception as e:
        print(f"❌ Multiple symbols test failed: {str(e)}")
        return False
    
    # Test 5: Check API key configuration
    print("\n5. Checking API key configuration...")
    api_keys = {
        'ALPHA_VANTAGE_API_KEY': analyzer.alpha_vantage_key,
        'NEWS_API_KEY': analyzer.news_api_key,
        'FINNHUB_TOKEN': analyzer.finnhub_token
    }
    
    configured_apis = []
    for key_name, key_value in api_keys.items():
        if key_value:
            configured_apis.append(key_name)
            print(f"✅ {key_name} is configured")
        else:
            print(f"⚠️  {key_name} not configured (will use default/fallback)")
    
    if configured_apis:
        print(f"✅ {len(configured_apis)} API(s) configured: {', '.join(configured_apis)}")
    else:
        print("⚠️  No API keys configured - using default sentiment")
        print("   To enable real news fetching, add API keys to .env file:")
        print("   - ALPHA_VANTAGE_API_KEY=your_key")
        print("   - NEWS_API_KEY=your_key")
        print("   - FINNHUB_TOKEN=your_token")
    
    return True

if __name__ == "__main__":
    print("🔍 News Sentiment Analyzer Test Suite")
    print("=" * 50)
    
    success = test_news_sentiment()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 News Sentiment Analyzer Test PASSED!")
        print("✅ Ready to integrate with ML Predictor")
    else:
        print("❌ News Sentiment Analyzer Test FAILED!") 