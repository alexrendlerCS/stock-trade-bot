from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

Base = declarative_base()

class StrategyType(enum.Enum):
    LONG_TERM = "long_term"
    SWING_TRADE = "swing_trade"
    DAY_TRADE = "day_trade"

class TradeStatus(enum.Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    strategy_type = Column(Enum(StrategyType))
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Integer)
    status = Column(Enum(TradeStatus))
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float, nullable=True)
    reason = Column(String)
    
    # Relationships
    position_id = Column(Integer, ForeignKey("positions.id"))
    position = relationship("Position", back_populates="trades")

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    strategy_type = Column(Enum(StrategyType))
    quantity = Column(Integer)
    entry_price = Column(Float)
    current_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="position")

class Performance(Base):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    strategy_type = Column(Enum(StrategyType))
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    total_pnl = Column(Float)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)

class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    strategy_type = Column(Enum(StrategyType))
    signal_type = Column(String)  # "BUY" or "SELL"
    strength = Column(Float)  # Signal strength/confidence
    timestamp = Column(DateTime, default=datetime.utcnow)
    indicators = Column(String)  # JSON string of indicator values
    executed = Column(Boolean, default=False) 