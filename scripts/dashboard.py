import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from app.core.config import settings
import alpaca_trade_api as tradeapi

# Database connection
engine = create_engine(settings.DATABASE_URL)

# Alpaca API connection
api = tradeapi.REST(
    settings.ALPACA_API_KEY,
    settings.ALPACA_SECRET_KEY,
    settings.ALPACA_BASE_URL
)

st.set_page_config(page_title="AI Stock Trading Bot Dashboard", layout="wide")
st.title("🤖 AI Stock Trading Bot Dashboard")

# Portfolio metrics
account = api.get_account()
portfolio_value = float(account.equity)

# Get portfolio history for gain/loss calculations
ph_24h = api.get_portfolio_history(period="1D", timeframe="5Min")
ph_1m = api.get_portfolio_history(period="1M", timeframe="1D")
ph_1y = api.get_portfolio_history(period="1A", timeframe="1D")

print("24hr equity:", ph_24h.equity)
print("1M equity:", ph_1m.equity)
print("1Y equity:", ph_1y.equity)

def calc_gain_loss(history):
    equity = history.equity
    start = next((v for v in equity if v > 0), None)
    end = equity[-1]
    if start is None or end is None:
        return 0.0, 0.0
    gain = end - start
    pct = (gain / start) * 100 if start != 0 else 0.0
    return gain, pct

gain_24h, pct_24h = calc_gain_loss(ph_24h)
gain_1m, pct_1m = calc_gain_loss(ph_1m)
gain_1y, pct_1y = calc_gain_loss(ph_1y)

# Display metrics at the top
colA, colB, colC, colD = st.columns(4)
colA.metric("Portfolio Value", f"${portfolio_value:,.2f}")
colB.metric("24hr Gain/Loss", f"${gain_24h:,.2f}", f"{pct_24h:.2f}%")
colC.metric("1M Gain/Loss", f"${gain_1m:,.2f}", f"{pct_1m:.2f}%")
colD.metric("1Y Gain/Loss", f"${gain_1y:,.2f}", f"{pct_1y:.2f}%")

# Sidebar - Profile
st.sidebar.header("Bot Profile")
st.sidebar.write(f"**Project:** {settings.PROJECT_NAME}")
st.sidebar.write(f"**Broker:** Alpaca (Paper)")
st.sidebar.write(f"**Base URL:** {settings.ALPACA_BASE_URL}")

# Load trades
def load_trades():
    query = "SELECT id, symbol, strategy_type, entry_price, exit_price, quantity, status, entry_time, exit_time, pnl, reason FROM trades ORDER BY entry_time DESC LIMIT 100"
    return pd.read_sql(query, engine)

# Load positions
def load_positions():
    query = "SELECT symbol, strategy_type, quantity, entry_price, current_price, entry_time, is_active FROM positions WHERE is_active = 1"
    return pd.read_sql(query, engine)

# Load performance
def load_performance():
    query = "SELECT date, strategy_type, total_trades, winning_trades, losing_trades, total_pnl, win_rate, sharpe_ratio, max_drawdown FROM performance ORDER BY date DESC LIMIT 30"
    return pd.read_sql(query, engine)

# Main dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recent Trades")
    trades_df = load_trades()
    st.dataframe(trades_df, use_container_width=True)
    if not trades_df.empty:
        st.write("**Last Trade Reason:**")
        st.info(trades_df.iloc[0]['reason'])

with col2:
    st.subheader("Current Portfolio")
    positions_df = load_positions()
    st.dataframe(positions_df, use_container_width=True)

st.subheader("Performance Metrics (Last 30 Days)")
perf_df = load_performance()
st.dataframe(perf_df, use_container_width=True)

# Equity curve (if available)
if not perf_df.empty:
    st.line_chart(perf_df.set_index('date')['total_pnl'], use_container_width=True)

st.caption("Made with ❤️ using Streamlit and FastAPI | v1.0.0") 