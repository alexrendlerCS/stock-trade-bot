#!/usr/bin/env python3
"""
Test script to show expected dashboard behavior during weekends
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import sqlite3
import pandas as pd
from datetime import datetime

def check_weekend_dashboard():
    """Check what the dashboard should show during weekends"""
    
    print("🧪 Weekend Dashboard Test")
    print("=" * 50)
    
    # Check market hours
    now = datetime.now()
    is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    print(f"\n📅 Current time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")
    print(f"🌅 Is weekend: {'Yes' if is_weekend else 'No'}")
    
    if is_weekend:
        print("\n✅ **Expected Dashboard Behavior (Weekend Mode):**")
        print("   📅 Blue 'Weekend Mode' banner should be displayed")
        print("   🧪 ML prediction testing should be available")
        print("   📰 News sentiment analysis should work")
        print("   📊 Historical performance should be shown")
        print("   🌙 'Market Closed' status in sidebar")
        print("   ⚙️ Bot configuration options available")
    else:
        print("\n✅ **Expected Dashboard Behavior (Market Hours):**")
        print("   📈 Live trading metrics")
        print("   🔄 Real-time updates")
        print("   ✅ 'Market Hours' status in sidebar")
    
    # Check database content
    print("\n📊 Database Content Check:")
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        # Check trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0]
        print(f"   📈 Total trades in database: {trade_count}")
        
        if trade_count > 0:
            cursor.execute("SELECT symbol, strategy_type, pnl, status FROM trades ORDER BY entry_time DESC LIMIT 5")
            recent_trades = cursor.fetchall()
            print(f"   📋 Recent trades:")
            for trade in recent_trades:
                print(f"      {trade[0]} | {trade[1]} | P&L: ${trade[2] or 0:.2f} | {trade[3]}")
        else:
            print("   📝 No trades found - this is normal for a new setup")
        
        # Check positions
        cursor.execute("SELECT COUNT(*) FROM positions WHERE is_active = 1")
        position_count = cursor.fetchone()[0]
        print(f"   🎯 Active positions: {position_count}")
        
        conn.close()
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
    
    # Test ML predictions availability
    print("\n🧠 ML Prediction Availability:")
    try:
        from app.ml_predictor import MLPredictor
        predictor = MLPredictor()
        if predictor.model_3d and predictor.model_5d:
            print("   ✅ ML models loaded - predictions available")
            print("   🎯 Can test signals for: AAPL, MSFT, TSLA, NVDA, GOOGL")
        else:
            print("   ❌ ML models not loaded")
    except Exception as e:
        print(f"   ❌ ML predictor error: {e}")
    
    # Test news sentiment availability
    print("\n📰 News Sentiment Availability:")
    try:
        from app.news_sentiment import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        if analyzer.sentiment_analyzer:
            print("   ✅ News sentiment analyzer loaded")
            print("   📰 Can analyze current market sentiment")
        else:
            print("   ⚠️  News sentiment analyzer not loaded (API keys needed)")
    except Exception as e:
        print(f"   ❌ News sentiment error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 **What You Should See on Dashboard:**")
    
    if is_weekend:
        print("\n🌙 **Weekend Mode Dashboard:**")
        print("   1. Blue banner: 'Weekend Mode - Markets are closed'")
        print("   2. Portfolio metrics: Total P&L, positions, trades, win rate")
        print("   3. ML Predictions section with test symbols")
        print("   4. Interactive charts (if data available)")
        print("   5. Historical performance graphs")
        print("   6. Sidebar: Market Status = 'Market Closed'")
        print("   7. Manual testing button for live signals")
        print("   8. Configuration sliders and options")
        
        print("\n✅ **Available Functions:**")
        print("   🧪 Test ML predictions on any symbol")
        print("   📰 Analyze current news sentiment")
        print("   📊 Review past trading performance")
        print("   ⚙️ Adjust bot settings")
        print("   🎯 Plan trading strategies for Monday")
        
    else:
        print("\n📈 **Live Market Dashboard:**")
        print("   1. Real-time portfolio updates")
        print("   2. Live trading signals")
        print("   3. Active position monitoring")
        print("   4. Real-time news sentiment")
    
    print(f"\n🌐 **Dashboard URL:** http://localhost:8503")
    print("📝 **Dashboard file:** scripts/dashboard.py")
    
    return True

if __name__ == "__main__":
    check_weekend_dashboard() 