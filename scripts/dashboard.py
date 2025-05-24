import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from app.core.config import settings
import alpaca_trade_api as tradeapi
import pytz
from datetime import datetime, timedelta
import re
from app.services.ml_predictor import MLPredictor

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
print("24hr timestamps:", getattr(ph_24h, 'timestamp', None))
print("1M equity:", ph_1m.equity)
print("1M timestamps:", getattr(ph_1m, 'timestamp', None))
print("1Y equity:", ph_1y.equity)
print("1Y timestamps:", getattr(ph_1y, 'timestamp', None))

# Helper to check if history is valid (not all zeros, enough points)
def is_valid_history(history, min_points=2):
    equity = [v for v in history.equity if v > 0]
    return len(equity) >= min_points

# Improved gain/loss calculation with time zone alignment for 24hr
local_tz = pytz.timezone(settings.TIMEZONE)
now = datetime.now(local_tz)

def calc_gain_loss(history, period_label):
    equity = history.equity
    timestamps = getattr(history, 'timestamp', None)
    # Remove zeros
    nonzero = [(i, v) for i, v in enumerate(equity) if v > 0]
    if len(nonzero) < 2:
        return None, None
    # For 24hr, align to last 24hr window using timestamps
    if period_label == '24h' and timestamps is not None:
        times = [datetime.fromtimestamp(ts, local_tz) for ts in timestamps]
        cutoff = now - timedelta(hours=24)
        window = [(t, v) for (t, v) in zip(times, equity) if v > 0 and t >= cutoff]
        if len(window) < 2:
            return None, None
        start = window[0][1]
        end = window[-1][1]
    else:
        # Use first and last nonzero for 1M/1Y, but only if enough data and first nonzero is early enough
        if len(nonzero) < 5:
            return None, None
        first_nonzero_index = nonzero[0][0]
        total_len = len(equity)
        if first_nonzero_index > int(0.2 * total_len):
            return None, None
        start = nonzero[0][1]
        end = nonzero[-1][1]
    if start is None or end is None:
        return None, None
    gain = end - start
    pct = (gain / start) * 100 if start != 0 else 0.0
    return gain, pct

gain_24h, pct_24h = calc_gain_loss(ph_24h, '24h')
gain_1m, pct_1m = calc_gain_loss(ph_1m, '1m')
gain_1y, pct_1y = calc_gain_loss(ph_1y, '1y')

# Display metrics at the top
colA, colB, colC, colD = st.columns(4)
colA.metric("Portfolio Value", f"${portfolio_value:,.2f}")
if gain_24h is not None:
    colB.metric("24hr Gain/Loss", f"${gain_24h:,.2f}", f"{pct_24h:.2f}%")
else:
    colB.metric("24hr Gain/Loss", "N/A", "N/A")
if gain_1m is not None:
    colC.metric("1M Gain/Loss", f"${gain_1m:,.2f}", f"{pct_1m:.2f}%")
else:
    colC.metric("1M Gain/Loss", "N/A", "N/A")
if gain_1y is not None:
    colD.metric("1Y Gain/Loss", f"${gain_1y:,.2f}", f"{pct_1y:.2f}%")
else:
    colD.metric("1Y Gain/Loss", "N/A", "N/A")

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

# --- ML Signal Confidence Visualization ---
st.subheader("ML Signal Confidence (Recent)")

# Extract ML signals from trades (by strategy_type or reason/indicators)
ml_trades = trades_df[trades_df['strategy_type'].str.lower().str.contains('ml|machine', na=False) |
                     trades_df['reason'].str.contains('ML model', na=False)]

# Try to extract confidence from indicators or reason
def extract_confidence(row):
    # Try to extract from indicators string
    if 'indicators' in row and isinstance(row['indicators'], str):
        match = re.search(r'3d_up=([0-9.]+)%', row['indicators'])
        if match:
            return float(match.group(1)) / 100
        match = re.search(r'3d_down=([0-9.]+)%', row['indicators'])
        if match:
            return float(match.group(1)) / 100
    # Try to extract from reason string
    if 'reason' in row and isinstance(row['reason'], str):
        match = re.search(r'3-day probability=([0-9.]+)%', row['reason'])
        if match:
            return float(match.group(1)) / 100
    return None

if not ml_trades.empty:
    ml_trades = ml_trades.copy()
    ml_trades['confidence'] = ml_trades.apply(extract_confidence, axis=1)
    ml_trades['entry_time'] = pd.to_datetime(ml_trades['entry_time'])
    ml_trades = ml_trades.sort_values('entry_time')

    # --- Interactive filters ---
    symbols = ml_trades['symbol'].unique().tolist()
    selected_symbols = st.multiselect("Select symbol(s) to display", symbols, default=symbols)
    signal_types = ml_trades['signal_type'].dropna().unique().tolist() if 'signal_type' in ml_trades.columns else []
    if signal_types:
        selected_signal_types = st.multiselect("Select signal type(s)", signal_types, default=signal_types)
    else:
        selected_signal_types = []

    filtered = ml_trades[ml_trades['symbol'].isin(selected_symbols)]
    if selected_signal_types:
        filtered = filtered[filtered['signal_type'].isin(selected_signal_types)]

    st.dataframe(filtered[['entry_time', 'symbol', 'signal_type', 'confidence', 'reason']], use_container_width=True)
    if not filtered.empty:
        st.line_chart(
            filtered.set_index('entry_time')[['confidence']],
            use_container_width=True
        )
    else:
        st.info("No ML signals for selected filters.")
else:
    st.info("No recent ML signals found in trades.")

# --- Test ML Strategy Button ---
st.sidebar.subheader("Test ML Strategy")
# Symbol selector for test
all_symbols = [s for s in trades_df['symbol'].unique() if isinstance(s, str)]
default_symbol = all_symbols[0] if all_symbols else 'TQQQ'
test_symbol = st.sidebar.selectbox("Select symbol for test", all_symbols or ['TQQQ'], index=0)
run_test = st.sidebar.button("Run Test ML Strategy")
finish_test = st.sidebar.button("Finish Test (Remove Test Trades)")

if run_test:
    # Fetch latest data for the symbol (last 30 days)
    import yfinance as yf
    df = yf.download(test_symbol, period='30d', auto_adjust=False)
    if not df.empty:
        # Flatten columns if they are tuples (MultiIndex)
        df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [col.lower() for col in df.columns]
        # If 'close' is missing but 'adj close' exists, use it
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        if 'close' not in df.columns:
            st.sidebar.error("No 'close' price found in data for symbol.")
        else:
            predictor = MLPredictor()
            try:
                preds = predictor.predict(df)
                prob_3d_up, prob_3d_down = preds['3d']
                prob_5d_up, prob_5d_down = preds['5d']
                # Insert a test trade into the DB
                import sqlalchemy
                with engine.begin() as conn:
                    conn.execute(sqlalchemy.text('''
                        INSERT INTO trades (symbol, strategy_type, entry_price, quantity, status, entry_time, reason)
                        VALUES (:symbol, :strategy_type, :entry_price, :quantity, :status, :entry_time, :reason)
                    '''), {
                        'symbol': test_symbol,
                        'strategy_type': 'ML_TEST',
                        'entry_price': float(df['close'].iloc[-1]),
                        'quantity': 1,
                        'status': 'EXECUTED',
                        'entry_time': datetime.now().isoformat(),
                        'reason': f'TEST_ML_TRADE 3d_up={prob_3d_up:.2%}, 5d_up={prob_5d_up:.2%}, 3d_down={prob_3d_down:.2%}, 5d_down={prob_5d_down:.2%}'
                    })
                st.sidebar.success(f"Test ML trade inserted for {test_symbol}.")
            except Exception as e:
                st.sidebar.error(f"ML prediction failed: {e}")
    else:
        st.sidebar.error("No data found for symbol.")

if finish_test:
    import sqlalchemy
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("DELETE FROM trades WHERE reason LIKE 'TEST_ML_TRADE%'"))
    st.sidebar.success("All test trades removed.")

st.caption("Made with ❤️ using Streamlit and FastAPI | v1.0.0") 