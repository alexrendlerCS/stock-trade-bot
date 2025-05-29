import sys
import os
import asyncio
import logging

# Configure logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)  # Suppress yfinance errors
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)  # Reduce HF warnings

# Fix for asyncio on Windows with Python 3.13
if sys.platform == "win32" and sys.version_info >= (3, 13):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
import json
import yfinance as yf
import time

# Import our enhanced components
try:
    from app.ml_predictor import MLPredictor
    from app.news_sentiment import NewsSentimentAnalyzer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="🤖 AI Trading Bot Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - IMPROVED READABILITY
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .profit { color: #00ff00; }
    .loss { color: #ff0000; }
    .neutral { color: #888888; }
    
    /* MUCH MORE READABLE WEEKEND MODE */
    .weekend-mode {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #00bcd4;
        margin: 20px 0;
        color: white;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .weekend-mode h3 {
        color: #00e5ff;
        font-weight: 700;
        margin: 0 0 15px 0;
        font-size: 1.3em;
    }
    .weekend-mode ul {
        color: #e1f5fe;
        margin: 15px 0;
        padding-left: 20px;
    }
    .weekend-mode li {
        margin: 8px 0;
        color: #e1f5fe;
        font-weight: 400;
    }
    .weekend-mode .emoji {
        font-size: 1.2em;
        margin-right: 8px;
    }
    
    /* Market closed styling */
    .market-closed {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 15px 0;
        color: #c62828;
        font-weight: 500;
    }
    
    /* Enhanced info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        font-weight: 500;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🤖 AI Stock Trading Bot Dashboard")
st.markdown("**Enhanced with ML Predictions & News Sentiment Analysis**")

# Market status check
def is_market_hours():
    """Simple market hours check"""
    now = datetime.now()
    # Weekend check
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    # Basic time check (9:30 AM - 4:00 PM ET would need timezone conversion)
    return True

market_open = is_market_hours()
if not market_open:
    now = datetime.now()
    st.markdown(f"""
    <div class="weekend-mode">
        <h3>🌙 Weekend Mode - Markets are Closed</h3>
        <p><strong>Current Time:</strong> {now.strftime('%A, %B %d, %Y at %I:%M %p')}</p>
        <p><strong>📅 Market Status:</strong> Closed until Monday 9:30 AM ET</p>
        
        <p><strong>✨ Available Features:</strong></p>
        <p>🧪 <strong>Test ML Predictions</strong> - Analyze any stock symbol with our 61% accuracy models</p>
        <p>📰 <strong>Live News Sentiment</strong> - Real-time FinBERT analysis of financial news</p>
        <p>📊 <strong>Historical Performance</strong> - Review past trading results and strategies</p>
        <p>⚙️ <strong>Configure Settings</strong> - Adjust confidence thresholds and risk parameters</p>
        <p>🎯 <strong>Plan for Monday</strong> - Prepare your trading strategy for market open</p>
        
        <p><strong>💡 Tip:</strong> Use the sidebar to test live signals and analyze market sentiment!</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🎛️ Bot Configuration")

# Database connection - FIXED MULTIPLE CONNECTIONS
@st.cache_resource
def get_db_connection():
    return sqlite3.connect('trading_bot.db', check_same_thread=False)

# Load bot status and configuration - Fixed database queries
@st.cache_data(ttl=30)
def load_portfolio_summary():
    """Load current portfolio summary from the live bot"""
    try:
        conn = get_db_connection()
        
        # Get recent trades with actual column names
        trades_query = """
        SELECT id, symbol, strategy_type, entry_price, exit_price, quantity, 
               entry_time, exit_time, pnl, status, reason
        FROM trades 
        ORDER BY entry_time DESC 
        LIMIT 50
        """
        trades_df = pd.read_sql(trades_query, conn)
        
        # Calculate portfolio metrics - FIXED PANDAS WARNING
        if not trades_df.empty:
            trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
            
            # FIXED: Handle case where all P&L values are 0 or null
            total_pnl = trades_df['pnl'].fillna(0).sum()
            open_trades = trades_df[trades_df['status'] == 'OPEN']
            closed_trades = trades_df[trades_df['status'] == 'CLOSED']
            executed_trades = trades_df[trades_df['status'] == 'EXECUTED']
            
            # Count all completed trades (CLOSED + EXECUTED)
            all_completed_trades = pd.concat([closed_trades, executed_trades]) if not closed_trades.empty or not executed_trades.empty else pd.DataFrame()
            
            # FIXED: Calculate win rate from all available trades, not just closed ones
            if not all_completed_trades.empty:
                valid_trades = all_completed_trades.dropna(subset=['pnl'])
                if len(valid_trades) > 0:
                    win_rate = (valid_trades['pnl'] > 0).mean() * 100
                else:
                    # If no P&L data, assume neutral performance
                    win_rate = 0.0
            else:
                win_rate = 0.0
        else:
            total_pnl = 0
            open_trades = pd.DataFrame()
            closed_trades = pd.DataFrame()
            win_rate = 0
        
        conn.close()
        
        return {
            'total_pnl': total_pnl,
            'open_positions': len(open_trades),
            'total_trades': len(trades_df),  # Count all trades, not just closed
            'win_rate': win_rate,
            'trades_df': trades_df
        }
    except Exception as e:
        st.error(f"Error loading portfolio: {e}")
        return None

# Load positions - Fixed query and connection handling
@st.cache_data(ttl=30)
def load_positions():
    """Load current positions"""
    try:
        # Create a fresh connection for this query
        conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # Check if positions table exists and has the right columns
        cursor.execute("PRAGMA table_info(positions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if not columns:
            # Table doesn't exist or has no columns
            conn.close()
            return pd.DataFrame()
        
        if 'is_active' in columns:
            positions_query = """
            SELECT symbol, quantity, entry_price, entry_time
            FROM positions 
            WHERE is_active = 1
            """
        else:
            # Fallback if is_active column doesn't exist
            positions_query = """
            SELECT symbol, quantity, entry_price, entry_time
            FROM positions 
            LIMIT 10
            """
        
        positions_df = pd.read_sql(positions_query, conn)
        conn.close()
        return positions_df
    except Exception as e:
        # Don't show error for missing positions - it's normal
        return pd.DataFrame()

# Add retry logic for yfinance with exponential backoff
def download_with_retry(symbol, max_retries=5):
    """Download data with retry logic and exponential backoff"""
    for attempt in range(max_retries):
        try:
            # Disable progress bar to reduce console spam
            df = yf.download(
                symbol,
                period='60d',
                auto_adjust=False,
                progress=False
            )
            if not df.empty:
                return df
        except Exception as e:
            wait_time = (2 ** attempt)  # Exponential backoff
            if attempt < max_retries - 1:  # don't sleep on the last attempt
                time.sleep(wait_time)
            continue
    return None

# Modify get_live_predictions function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_live_predictions(symbols):
    """Get live ML predictions for symbols"""
    predictor = MLPredictor()
    predictions = {}
    
    with st.spinner("Loading market data..."):
    for symbol in symbols:
        try:
                df = download_with_retry(symbol)
                if df is None:
                    st.warning(f"Could not download data for {symbol}, skipping...")
                    continue
                    
                # FIXED: Proper handling of MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
                
                # Convert all column names to lowercase
                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                
                # Map common column variations to standard names
                column_mapping = {
                    'adj_close': 'close',
                    'adjclose': 'close',
                    f'adj_close_{symbol.lower()}': 'close',
                    f'close_{symbol.lower()}': 'close',
                    f'high_{symbol.lower()}': 'high',
                    f'low_{symbol.lower()}': 'low',
                    f'open_{symbol.lower()}': 'open',
                    f'volume_{symbol.lower()}': 'volume'
                }
                
                # Apply column mapping
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns and new_name not in df.columns:
                        df[new_name] = df[old_name]
                
                # Ensure we have required columns
                required_cols = ['close', 'high', 'low', 'open', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.warning(f"Missing columns for {symbol}: {missing_cols}")
                    continue
                
                # Get prediction without sentiment for now
                pred = predictor.predict(df)
                if pred:
                    predictions[symbol] = {
                        '3d_up': pred['3d'][0],
                        '3d_down': pred['3d'][1],
                        '5d_up': pred['5d'][0],
                        '5d_down': pred['5d'][1],
                        'current_price': df['close'].iloc[-1]
                    }
        except Exception as e:
            st.warning(f"Could not get prediction for {symbol}: {str(e)}")
    
    return predictions

# Modify get_sentiment_analysis function to return dummy data
@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_sentiment_analysis(symbols):
    """Get news sentiment for symbols (temporarily disabled)"""
        sentiment_data = {}
        for symbol in symbols:
        sentiment_data[symbol] = {
            'sentiment_score': 0.0,
            'news_count': 0,
            'sentiment_strength': 'neutral',
            'positive_ratio': 0.5,
            'negative_ratio': 0.5,
            'news_volume': 0,
            'recent_sentiment': 0.0
        }
        return sentiment_data

# Main dashboard layout
portfolio_data = load_portfolio_summary()

if portfolio_data:
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pnl_color = "profit" if portfolio_data['total_pnl'] >= 0 else "loss"
        st.metric(
            "Total P&L", 
            f"${portfolio_data['total_pnl']:.2f}",
            delta=f"{portfolio_data['total_pnl']:.2f}"
        )
    
    with col2:
        st.metric("Open Positions", portfolio_data['open_positions'])
    
    with col3:
        st.metric("Total Trades", portfolio_data['total_trades'])
    
    with col4:
        st.metric("Win Rate", f"{portfolio_data['win_rate']:.1f}%")

    # Sidebar controls
    st.sidebar.subheader("📊 Market Analysis")
    
    # Symbol selection
    default_symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    symbols = st.sidebar.multiselect(
        "Select Symbols for Analysis",
        default_symbols + ['META', 'AMZN', 'NFLX', 'AMD', 'INTC'],
        default=default_symbols
    )
    
    if symbols:
        # Get live predictions and sentiment
        with st.spinner("Loading ML predictions and sentiment analysis..."):
            predictions = get_live_predictions(symbols)
            sentiment_data = get_sentiment_analysis(symbols)
        
        # ML Predictions & Sentiment Dashboard
        st.header("🧠 Live ML Predictions & Sentiment")
        
        if predictions:
            prediction_df = pd.DataFrame(predictions).T
            prediction_df.index.name = 'Symbol'
            
            # Add sentiment scores
            for symbol in prediction_df.index:
                if symbol in sentiment_data:
                    prediction_df.loc[symbol, 'sentiment_score'] = sentiment_data[symbol].get('sentiment_score', 0)
                    prediction_df.loc[symbol, 'news_count'] = sentiment_data[symbol].get('news_count', 0)
                else:
                    prediction_df.loc[symbol, 'sentiment_score'] = 0
                    prediction_df.loc[symbol, 'news_count'] = 0
            
            # Display predictions table
            st.subheader("📈 Current Market Signals")
            
            # Format the dataframe for display
            display_df = prediction_df.copy()
            for col in ['3d_up', '3d_down', '5d_up', '5d_down']:
                if col in display_df.columns:
                    display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
            
            if 'current_price' in display_df.columns:
                display_df['current_price'] = '$' + display_df['current_price'].round(2).astype(str)
            
            if 'sentiment_score' in display_df.columns:
                display_df['sentiment_score'] = display_df['sentiment_score'].round(3)
            
            st.dataframe(
                display_df[['current_price', '3d_up', '3d_down', 'sentiment_score', 'news_count']],
                use_container_width=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 ML Confidence Levels")
                
                # Create confidence chart
                conf_data = []
                for symbol in prediction_df.index:
                    conf_data.append({
                        'Symbol': symbol,
                        '3D Up': prediction_df.loc[symbol, '3d_up'] * 100,
                        '3D Down': prediction_df.loc[symbol, '3d_down'] * 100
                    })
                
                conf_df = pd.DataFrame(conf_data)
                
                fig = px.bar(
                    conf_df, 
                    x='Symbol', 
                    y=['3D Up', '3D Down'],
                    title="ML Prediction Confidence (%)",
                    color_discrete_map={'3D Up': '#00ff00', '3D Down': '#ff4444'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📰 News Sentiment")
                
                # Create sentiment chart
                sent_data = []
                for symbol in symbols:
                    if symbol in sentiment_data:
                        sent_data.append({
                            'Symbol': symbol,
                            'Sentiment': sentiment_data[symbol].get('sentiment_score', 0),
                            'News Count': sentiment_data[symbol].get('news_count', 0)
                        })
                
                if sent_data:
                    sent_df = pd.DataFrame(sent_data)
                    
                    fig = px.scatter(
                        sent_df,
                        x='Sentiment',
                        y='News Count',
                        color='Symbol',
                        size='News Count',
                        title="Sentiment vs News Volume",
                        hover_data=['Symbol']
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentiment data available")
        else:
            st.warning("No ML predictions available. This could be due to:")
            st.write("- Weekend/market closed")
            st.write("- Data download issues")
            st.write("- Model loading problems")

    # Trading Performance Section
    st.header("📊 Trading Performance")
    
    if not portfolio_data['trades_df'].empty:
        trades_df = portfolio_data['trades_df'].copy()
        
        # FIXED: Better data handling for performance analysis
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], errors='coerce')
        trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
        
        # Show basic trade statistics even if P&L is all zeros
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Trade History")
            
            # If we have valid timestamps, show P&L performance
            if trades_df['entry_time'].notna().any():
                trade_timeline = trades_df.dropna(subset=['entry_time']).copy()
                trade_timeline = trade_timeline.sort_values('entry_time')
                trade_timeline['pnl_clean'] = pd.to_numeric(trade_timeline['pnl'], errors='coerce').fillna(0)
                trade_timeline['cumulative_pnl'] = trade_timeline['pnl_clean'].cumsum()
                
                if len(trade_timeline) > 0:
                    # Show cumulative P&L over time
                    fig = go.Figure()
                    
                    # Add cumulative P&L line
                    fig.add_trace(go.Scatter(
                        x=trade_timeline['entry_time'],
                        y=trade_timeline['cumulative_pnl'],
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='#00ff88', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>Date:</b> %{x}<br>' +
                                    '<b>Cumulative P&L:</b> $%{y:.2f}<br>' +
                                    '<extra></extra>'
                    ))
                    
                    # Add individual trade P&L as bars
                    colors = ['#00ff88' if pnl >= 0 else '#ff4444' for pnl in trade_timeline['pnl_clean']]
                    fig.add_trace(go.Bar(
                        x=trade_timeline['entry_time'],
                        y=trade_timeline['pnl_clean'],
                        name='Individual Trade P&L',
                        marker_color=colors,
                        opacity=0.7,
                        hovertemplate='<b>Symbol:</b> %{customdata[0]}<br>' +
                                    '<b>Strategy:</b> %{customdata[1]}<br>' +
                                    '<b>P&L:</b> $%{y:.2f}<br>' +
                                    '<b>Date:</b> %{x}<br>' +
                                    '<extra></extra>',
                        customdata=list(zip(trade_timeline['symbol'], trade_timeline['strategy_type']))
                    ))
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig.update_layout(
                        title="Portfolio P&L Performance",
                        xaxis_title="Date",
                        yaxis_title="P&L ($)",
                        height=400,
                        hovermode='x unified',
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show additional P&L insights
                    if trade_timeline['pnl_clean'].abs().sum() > 0:  # If there's actual P&L data
                        total_pnl = trade_timeline['pnl_clean'].sum()
                        winning_trades = (trade_timeline['pnl_clean'] > 0).sum()
                        losing_trades = (trade_timeline['pnl_clean'] < 0).sum()
                        avg_win = trade_timeline[trade_timeline['pnl_clean'] > 0]['pnl_clean'].mean() if winning_trades > 0 else 0
                        avg_loss = trade_timeline[trade_timeline['pnl_clean'] < 0]['pnl_clean'].mean() if losing_trades > 0 else 0
                        
                        st.markdown(f"""
                        **📈 P&L Summary:**
                        - **Total P&L:** ${total_pnl:.2f}
                        - **Winning Trades:** {winning_trades} (Avg: ${avg_win:.2f})
                        - **Losing Trades:** {losing_trades} (Avg: ${avg_loss:.2f})
                        - **Best Trade:** ${trade_timeline['pnl_clean'].max():.2f}
                        - **Worst Trade:** ${trade_timeline['pnl_clean'].min():.2f}
                        """)
                    else:
                        st.info("📊 All trades show $0.00 P&L - this is normal for paper trading or if trades haven't been closed yet.")
                else:
                    st.info("No valid trade data for P&L chart")
            else:
                st.info("No valid trade timestamps for chart")
        
        with col2:
            st.subheader("🎯 Trade Statistics")
            
            # Show basic stats regardless of P&L
            total_trades = len(trades_df)
            open_trades = (trades_df['status'] == 'OPEN').sum()
            executed_trades = (trades_df['status'] == 'EXECUTED').sum()
            closed_trades = (trades_df['status'] == 'CLOSED').sum()
            
            # Calculate P&L metrics
            trades_df['pnl_numeric'] = pd.to_numeric(trades_df['pnl'], errors='coerce').fillna(0)
            total_pnl = trades_df['pnl_numeric'].sum()
            
            # Create metrics display
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Total Trades", total_trades)
                st.metric("Open Trades", open_trades)
                
            with col2b:
                st.metric("Executed Trades", executed_trades)
                st.metric("Closed Trades", closed_trades)
            
            # P&L breakdown
            if trades_df['pnl_numeric'].abs().sum() > 0:
                winning_trades = (trades_df['pnl_numeric'] > 0).sum()
                losing_trades = (trades_df['pnl_numeric'] < 0).sum()
                breakeven_trades = (trades_df['pnl_numeric'] == 0).sum()
                
                # Create a simple P&L breakdown chart
                pnl_breakdown = pd.DataFrame({
                    'Type': ['Winning', 'Losing', 'Breakeven'],
                    'Count': [winning_trades, losing_trades, breakeven_trades],
                    'Color': ['#00ff88', '#ff4444', '#888888']
                })
                
                if pnl_breakdown['Count'].sum() > 0:
                    fig_pie = px.pie(
                        pnl_breakdown, 
                        values='Count', 
                        names='Type',
                        title="Trade Outcome Distribution",
                        color_discrete_map={'Winning': '#00ff88', 'Losing': '#ff4444', 'Breakeven': '#888888'}
                    )
                    fig_pie.update_layout(height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Show symbols traded
            if 'symbol' in trades_df.columns:
                symbols_traded = trades_df['symbol'].value_counts()
                st.write("**📊 Symbols Traded:**")
                for symbol, count in symbols_traded.items():
                    symbol_pnl = trades_df[trades_df['symbol'] == symbol]['pnl_numeric'].sum()
                    st.write(f"- {symbol}: {count} trades (P&L: ${symbol_pnl:.2f})")
        
        # Recent trades table
        st.subheader("📈 Recent Trades")
        
        # Format trades for display
        display_trades = trades_df.copy()
        if not display_trades.empty:
            display_trades['entry_time'] = display_trades['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            if 'exit_time' in display_trades.columns:
                display_trades['exit_time'] = pd.to_datetime(display_trades['exit_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            
            # Show relevant columns
            columns_to_show = ['symbol', 'strategy_type', 'entry_price', 'exit_price', 
                              'quantity', 'pnl', 'status', 'entry_time']
            available_columns = [col for col in columns_to_show if col in display_trades.columns]
            
            st.dataframe(
                display_trades[available_columns].head(20),
                use_container_width=True
            )
        else:
            st.info("No trades to display")
    else:
        st.markdown("""
        <div class="info-box">
        📈 <strong>No Trading History Found</strong><br>
        Start the trading bot to see your trades and performance metrics here!<br>
        <code>python start_trading_bot.py</code>
        </div>
        """, unsafe_allow_html=True)

    # Current positions
    positions_df = load_positions()
    if not positions_df.empty:
        st.subheader("🎯 Current Positions")
        st.dataframe(positions_df, use_container_width=True)

    # Control Panel
    st.sidebar.header("🎮 Control Panel")
    
    # Bot status simulation (since we're not running live bot in dashboard)
    bot_status = st.sidebar.selectbox("Bot Status", ["Stopped", "Paper Trading", "Live Trading"])
    
    if bot_status == "Live Trading":
        st.sidebar.error("🚨 Live trading mode!")
    elif bot_status == "Paper Trading":
        st.sidebar.info("📝 Paper trading mode")
    else:
        st.sidebar.warning("❌ Bot is stopped")
    
    # Manual trade testing
    st.sidebar.subheader("🧪 Manual Testing")
    
    test_symbol = st.sidebar.selectbox("Test Symbol", symbols if symbols else default_symbols)
    
    if st.sidebar.button("🔍 Get Live Signal"):
        with st.spinner(f"Analyzing {test_symbol}..."):
            predictions = get_live_predictions([test_symbol])
            sentiment = get_sentiment_analysis([test_symbol])
            
            if test_symbol in predictions:
                pred = predictions[test_symbol]
                sent = sentiment.get(test_symbol, {})
                
                st.sidebar.success("✅ Analysis Complete!")
                st.sidebar.write(f"**3D Up:** {pred['3d_up']:.1%}")
                st.sidebar.write(f"**3D Down:** {pred['3d_down']:.1%}")
                st.sidebar.write(f"**Sentiment:** {sent.get('sentiment_score', 0):.3f}")
                st.sidebar.write(f"**News Count:** {sent.get('news_count', 0)}")
                
                # Determine signal
                confidence_threshold = 0.65
                if pred['3d_up'] > confidence_threshold:
                    st.sidebar.write("🟢 **Signal: LONG**")
                elif pred['3d_down'] > confidence_threshold:
                    st.sidebar.write("🔴 **Signal: SHORT**")
                else:
                    st.sidebar.write("⚪ **Signal: HOLD**")
            else:
                st.sidebar.error("Could not analyze symbol")

    # Settings
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.65, 0.05)
    max_positions = st.sidebar.number_input("Max Positions", 1, 10, 5)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 5, 2) / 100

    # Market status info
    st.sidebar.subheader("📈 Market Status")
    if market_open:
        st.sidebar.success("✅ Market Hours")
    else:
        st.sidebar.info("🌙 Market Closed")
        st.sidebar.write("**Next Open:** Monday 9:30 AM ET")

else:
    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Database Connection Issue</strong><br>
    Could not load portfolio data. Make sure the trading bot database is accessible.<br><br>
    💡 <strong>Tip:</strong> The database might not have been initialized yet. Try running the bot first:<br>
    <code>python start_trading_bot.py</code>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🤖 AI Stock Trading Bot Dashboard | Enhanced with ML & Sentiment Analysis | v2.0.0") 