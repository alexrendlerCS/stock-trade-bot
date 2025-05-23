import alpaca_trade_api as tradeapi
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpaca_connection():
    """Test the connection to Alpaca API"""
    try:
        # Initialize Alpaca API
        api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            settings.ALPACA_BASE_URL
        )
        
        # Test account information
        account = api.get_account()
        logger.info(f"Successfully connected to Alpaca!")
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Test market data
        aapl = api.get_latest_trade('AAPL')
        logger.info(f"Latest AAPL trade: ${float(aapl.price):,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Alpaca: {str(e)}")
        return False

if __name__ == "__main__":
    test_alpaca_connection() 