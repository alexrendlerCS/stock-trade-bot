from app.core.config import settings
import alpaca_trade_api as tradeapi

if __name__ == "__main__":
    print("Closing all open Alpaca positions...")
    api = tradeapi.REST(
        settings.ALPACA_API_KEY,
        settings.ALPACA_SECRET_KEY,
        settings.ALPACA_BASE_URL
    )
    positions = api.list_positions()
    for position in positions:
        print(f"Closing {position.symbol} ({position.qty} shares)...")
        api.submit_order(
            symbol=position.symbol,
            qty=position.qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
    print("Done.") 