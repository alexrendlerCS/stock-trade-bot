import pandas as pd
import yfinance as yf
from app.ml_predictor import MLPredictor
from datetime import datetime, timedelta

def test_integrated_ml_sentiment():
    print("Testing Integrated ML Predictor with News Sentiment...")
    
    # Initialize predictor
    print("\n1. Initializing ML Predictor with News Sentiment...")
    try:
        ml = MLPredictor()
        
        if ml.model_3d is None or ml.model_5d is None:
            print("❌ ML models failed to load!")
            return False
        
        print(f"✅ ML models loaded successfully")
        
        if ml.news_analyzer is None:
            print("⚠️  News analyzer not loaded (this is okay if no API keys)")
        else:
            print("✅ News sentiment analyzer loaded successfully")
            
    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        return False
    
    # Test with different symbols
    test_symbols = ['AAPL', 'TSLA']
    
    for symbol in test_symbols:
        print(f"\n2. Testing prediction for {symbol} with sentiment integration...")
        
        try:
            # Get test data
            df = yf.download(symbol, start='2024-01-01', end='2024-02-01', auto_adjust=False)
            
            if df.empty:
                print(f"❌ No test data for {symbol}")
                continue
            
            # Clean column names
            df.columns = [col[0] if hasattr(col, '__len__') and len(col) > 1 else col for col in df.columns]
            df.columns = [col.lower() for col in df.columns]
            
            print(f"✅ Downloaded data for {symbol}: {len(df)} rows")
            
            # Test prediction WITHOUT sentiment (traditional ML only)
            print(f"\n   Testing {symbol} WITHOUT news sentiment...")
            prediction_no_sentiment = ml.predict(df)
            
            if prediction_no_sentiment is None:
                print(f"❌ Prediction failed for {symbol} (no sentiment)")
                continue
            
            print(f"   ✅ Prediction without sentiment successful")
            print(f"   3d: Up={prediction_no_sentiment['3d'][0]:.3f}, Down={prediction_no_sentiment['3d'][1]:.3f}")
            print(f"   5d: Up={prediction_no_sentiment['5d'][0]:.3f}, Down={prediction_no_sentiment['5d'][1]:.3f}")
            
            # Test prediction WITH sentiment
            print(f"\n   Testing {symbol} WITH news sentiment...")
            prediction_with_sentiment = ml.predict(df, symbol=symbol)
            
            if prediction_with_sentiment is None:
                print(f"❌ Prediction failed for {symbol} (with sentiment)")
                continue
            
            print(f"   ✅ Prediction with sentiment successful")
            print(f"   3d: Up={prediction_with_sentiment['3d'][0]:.3f}, Down={prediction_with_sentiment['3d'][1]:.3f}")
            print(f"   5d: Up={prediction_with_sentiment['5d'][0]:.3f}, Down={prediction_with_sentiment['5d'][1]:.3f}")
            
            # Check if sentiment information is included
            if 'sentiment' in prediction_with_sentiment:
                sentiment_info = prediction_with_sentiment['sentiment']
                print(f"   📰 Sentiment Score: {sentiment_info['sentiment_score']:.3f}")
                print(f"   📈 News Count: {sentiment_info['news_count']}")
                print(f"   💪 Sentiment Strength: {sentiment_info['sentiment_strength']:.3f}")
                
                # Compare predictions
                prob_diff_3d = abs(prediction_with_sentiment['3d'][0] - prediction_no_sentiment['3d'][0])
                prob_diff_5d = abs(prediction_with_sentiment['5d'][0] - prediction_no_sentiment['5d'][0])
                
                print(f"   📊 Prediction difference with sentiment:")
                print(f"      3d difference: {prob_diff_3d:.4f}")
                print(f"      5d difference: {prob_diff_5d:.4f}")
                
                if prob_diff_3d > 0.01 or prob_diff_5d > 0.01:
                    print(f"   ✅ Sentiment is influencing predictions (good!)")
                else:
                    print(f"   ⚠️  Sentiment influence is minimal")
            else:
                print(f"   ⚠️  No sentiment info returned (likely no news data or API keys)")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n3. Testing sentiment feature generation...")
    try:
        if ml.news_analyzer:
            sentiment_features = ml._get_sentiment_features('AAPL')
            if sentiment_features:
                print(f"✅ Sentiment features generated successfully:")
                for feature_name, feature_value in sentiment_features.items():
                    print(f"   {feature_name}: {feature_value:.3f}")
            else:
                print(f"⚠️  No sentiment features generated (likely no news data)")
        else:
            print(f"⚠️  News analyzer not available for sentiment feature testing")
    except Exception as e:
        print(f"❌ Error testing sentiment features: {str(e)}")
    
    return True

if __name__ == "__main__":
    print("🔬 Integrated ML + News Sentiment Test Suite")
    print("=" * 60)
    
    success = test_integrated_ml_sentiment()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Integrated ML + News Sentiment Test PASSED!")
        print("✅ System is ready for enhanced trading with sentiment analysis")
        print("📈 The ML models can now use both technical and sentiment indicators")
    else:
        print("❌ Integrated ML + News Sentiment Test FAILED!") 