import pandas as pd
import yfinance as yf
from app.ml_predictor import MLPredictor
from datetime import datetime, timedelta

def test_ml_predictor():
    print("Testing ML Predictor...")
    
    # Initialize predictor
    ml = MLPredictor()
    
    if ml.model_3d is None or ml.model_5d is None:
        print("❌ Models failed to load!")
        return False
    
    print(f"✅ Models loaded successfully")
    print(f"✅ Feature list loaded: {len(ml.feature_cols)} features")
    
    # Get some test data
    print("\nDownloading test data...")
    df = yf.download('AAPL', start='2024-01-01', end='2024-02-01', auto_adjust=False)
    
    if df.empty:
        print("❌ No test data available")
        return False
    
    # Clean column names for consistency
    df.columns = [col[0] if hasattr(col, '__len__') and len(col) > 1 else col for col in df.columns]
    df.columns = [col.lower() for col in df.columns]
    
    print(f"✅ Test data downloaded: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Test prediction
    print("\nTesting prediction...")
    try:
        predictions = ml.predict(df)
        
        if predictions is None:
            print("❌ Prediction failed!")
            return False
        
        print("✅ Prediction successful!")
        print(f"3-day prediction: Up={predictions['3d'][0]:.3f}, Down={predictions['3d'][1]:.3f}")
        print(f"5-day prediction: Up={predictions['5d'][0]:.3f}, Down={predictions['5d'][1]:.3f}")
        
        # Validate probabilities
        for period in ['3d', '5d']:
            up_prob, down_prob = predictions[period]
            total_prob = up_prob + down_prob
            if abs(total_prob - 1.0) > 0.01:  # Allow small floating point errors
                print(f"⚠️  Warning: {period} probabilities don't sum to 1.0: {total_prob}")
            else:
                print(f"✅ {period} probabilities sum correctly: {total_prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_predictor()
    if success:
        print("\n🎉 ML Predictor Test PASSED! Ready for News Sentiment Integration.")
    else:
        print("\n❌ ML Predictor Test FAILED!") 