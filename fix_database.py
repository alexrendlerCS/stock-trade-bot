import sqlite3
import os

def fix_database():
    try:
        # Connect to database
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        # Get existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Existing tables: {tables}")
        
        # Drop existing trades table if it exists
        cursor.execute("DROP TABLE IF EXISTS trades;")
        print("Dropped existing trades table")
        
        # Create trades table with correct schema
        cursor.execute('''
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                pnl REAL,
                ml_probability REAL,
                sentiment_score REAL,
                confidence REAL,
                status TEXT DEFAULT 'OPEN'
            )
        ''')
        print("Created new trades table with correct schema")
        
        # Create portfolio_history table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                num_positions INTEGER NOT NULL,
                daily_pnl REAL
            )
        ''')
        print("Ensured portfolio_history table exists")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        print("\nDatabase schema fixed successfully!")
        
    except Exception as e:
        print(f"Error fixing database: {str(e)}")
        if conn:
            conn.close()

if __name__ == "__main__":
    print("Starting database fix...")
    fix_database() 