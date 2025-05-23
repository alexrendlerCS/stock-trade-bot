from app.services.trading import execute_trade
from app.db.models import StrategyType

if __name__ == "__main__":
    symbol = "AAPL"
    quantity = 1
    side = "buy"
    strategy_type = StrategyType.DAY_TRADE
    reason = "Test trade from script"
    trade = execute_trade(symbol, quantity, side, strategy_type, reason)
    if trade:
        print(f"Test trade executed: {trade.symbol}, {trade.quantity} shares, {trade.strategy_type}, {trade.reason}")
    else:
        print("Test trade failed.") 