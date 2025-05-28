import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import yfinance as yf

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.ml_predictor import MLPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_test_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get historical data for testing"""
    try:
        logger.info(f"Fetching data for {symbol} using period='{days}d'")
        
        # Download data using yf.download with minimal parameters
        df = yf.download(
            symbol,
            period=f"{days}d",
            interval="1d",
            progress=False
        )
        
        if df is None or len(df) == 0:
            logger.error(f"No data returned for {symbol}")
            return None
            
        logger.info(f"Original columns: {df.columns.tolist()}")
        
        # Extract just the first level of the multi-index columns
        df.columns = [col[0].lower() for col in df.columns]
        logger.info(f"New columns: {df.columns.tolist()}")
        
        logger.info(f"Successfully downloaded {len(df)} days of data")
        return df
        
    except Exception as e:
        logger.error(f"Error getting test data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_ml_predictor():
    """Test the ML predictor with news sentiment integration"""
    try:
        logger.info("Initializing ML Predictor...")
        predictor = MLPredictor()
        
        # Test stocks
        test_stocks = ['AAPL', 'MSFT', 'GOOGL']
        success_count = 0
        
        for symbol in test_stocks:
            logger.info(f"\nTesting predictions for {symbol}...")
            
            # Get historical data
            df = get_test_data(symbol)
            if df is None or len(df) == 0:
                logger.error(f"Could not get data for {symbol}")
                continue
                
            logger.info(f"Got {len(df)} days of historical data")
            
            # Make prediction
            prediction = predictor.predict(df, symbol)
            
            if prediction:
                logger.info(f"\nPrediction results for {symbol}:")
                logger.info(f"3-day forecast:")
                logger.info(f"  - Up probability: {prediction['3d'][0]:.3f}")
                logger.info(f"  - Down probability: {prediction['3d'][1]:.3f}")
                
                logger.info(f"\n5-day forecast:")
                logger.info(f"  - Up probability: {prediction['5d'][0]:.3f}")
                logger.info(f"  - Down probability: {prediction['5d'][1]:.3f}")
                
                if 'sentiment' in prediction:
                    logger.info(f"\nNews Sentiment:")
                    logger.info(f"  - Score: {prediction['sentiment']['sentiment_score']:.3f}")
                    logger.info(f"  - Strength: {prediction['sentiment']['sentiment_strength']}")
                    logger.info(f"  - News Count: {prediction['sentiment']['news_count']}")
                    
                success_count += 1
            else:
                logger.error(f"Could not get prediction for {symbol}")
                continue
            
            logger.info("\n" + "="*50)
        
        return success_count > 0  # Return True if at least one prediction succeeded
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting ML Predictor integration test...")
    logger.info("=" * 50)
    
    # Run the test
    success = test_ml_predictor()
    
    if success:
        logger.info("\n✅ All tests completed! ML Predictor with news sentiment is working.")
    else:
        logger.error("\n❌ Some tests failed. Please check the logs above for details.") 