from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Stock Trading Bot"
    
    # Alpaca Configuration
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./trading_bot.db"
    
    # Trading Configuration
    MAX_POSITION_SIZE: float = 0.01  # 1% of portfolio per trade
    STOP_LOSS_PERCENTAGE: float = 0.03  # 3% stop loss
    TAKE_PROFIT_PERCENTAGE: float = 0.05  # 5% take profit
    
    # Portfolio Allocation
    LONG_TERM_ALLOCATION: float = 0.50  # 50% for long-term investments
    SWING_TRADE_ALLOCATION: float = 0.30  # 30% for swing trades
    DAY_TRADE_ALLOCATION: float = 0.20  # 20% for day trading
    
    # Scheduler Configuration
    MARKET_OPEN_TIME: str = "06:30"  # 6:30 AM ET
    MARKET_CLOSE_TIME: str = "16:00"  # 4:00 PM ET
    TIMEZONE: str = "US/Eastern"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_bot.log"
    
    class Config:
        case_sensitive = True

settings = Settings() 