from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from app.core.config import settings
import pytz
import logging

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=settings.LOG_FILE
)
logger = logging.getLogger(__name__)

# Create scheduler
scheduler = BackgroundScheduler(timezone=pytz.timezone(settings.TIMEZONE))

def schedule_trading_tasks():
    """Schedule all trading-related tasks"""
    
    # Schedule market open tasks
    scheduler.add_job(
        'app.services.trading:execute_market_open_tasks',
        CronTrigger(
            hour=settings.MARKET_OPEN_TIME.split(':')[0],
            minute=settings.MARKET_OPEN_TIME.split(':')[1],
            timezone=settings.TIMEZONE
        ),
        id='market_open_tasks',
        replace_existing=True
    )
    
    # Schedule market close tasks
    scheduler.add_job(
        'app.services.trading:execute_market_close_tasks',
        CronTrigger(
            hour=settings.MARKET_CLOSE_TIME.split(':')[0],
            minute=settings.MARKET_CLOSE_TIME.split(':')[1],
            timezone=settings.TIMEZONE
        ),
        id='market_close_tasks',
        replace_existing=True
    )
    
    # Schedule daily performance update
    scheduler.add_job(
        'app.services.performance:update_daily_performance',
        CronTrigger(
            hour='16',  # 4 PM ET
            minute='30',
            timezone=settings.TIMEZONE
        ),
        id='daily_performance_update',
        replace_existing=True
    )
    
    logger.info("Trading tasks scheduled successfully")

# Initialize scheduler with trading tasks
schedule_trading_tasks() 