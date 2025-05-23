from typing import List
import pandas as pd
from app.strategies.base_strategy import BaseStrategy
from app.db.models import StrategyType, Signal
import logging

logger = logging.getLogger(__name__)

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(StrategyType.SWING_TRADE)
        self.fast_period = 50
        self.slow_period = 200
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
    
    def get_trade_reason(self, signal_type: str, ema_fast: float, ema_slow: float, rsi: float) -> str:
        """Generate a detailed explanation for the trade."""
        if signal_type == 'BUY':
            return (
                f"BUY: 50/200 EMA bullish crossover (EMA50={ema_fast:.2f} > EMA200={ema_slow:.2f}) "
                f"with RSI={rsi:.2f} (confirmation: not overbought)."
            )
        elif signal_type == 'SELL':
            return (
                f"SELL: 50/200 EMA bearish crossover (EMA50={ema_fast:.2f} < EMA200={ema_slow:.2f}) "
                f"with RSI={rsi:.2f} (confirmation: not oversold)."
            )
        else:
            return "Unknown signal type."

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on EMA crossover and RSI"""
        if not self.validate_data(data):
            logger.error("Invalid data format for EMA Crossover strategy")
            return []
        
        # Calculate indicators
        data['ema_fast'] = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        signals = []
        
        # Get the latest values
        current_price = data['close'].iloc[-1]
        ema_fast = data['ema_fast'].iloc[-1]
        ema_slow = data['ema_slow'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        
        # Previous values for crossover detection
        prev_ema_fast = data['ema_fast'].iloc[-2]
        prev_ema_slow = data['ema_slow'].iloc[-2]
        
        # Generate signals
        if (prev_ema_fast < prev_ema_slow) and (ema_fast > ema_slow):
            # Bullish crossover
            if rsi < self.rsi_overbought:
                reason = self.get_trade_reason('BUY', ema_fast, ema_slow, rsi)
                signal = Signal(
                    symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                    strategy_type=self.strategy_type,
                    signal_type='BUY',
                    strength=0.7,  # Moderate strength
                    indicators=f"EMA Crossover: Fast={ema_fast:.2f}, Slow={ema_slow:.2f}, RSI={rsi:.2f}",
                    reason=reason
                )
                signals.append(signal)
                logger.info(f"Generated BUY signal: {reason}")
        
        elif (prev_ema_fast > prev_ema_slow) and (ema_fast < ema_slow):
            # Bearish crossover
            if rsi > self.rsi_oversold:
                reason = self.get_trade_reason('SELL', ema_fast, ema_slow, rsi)
                signal = Signal(
                    symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                    strategy_type=self.strategy_type,
                    signal_type='SELL',
                    strength=0.7,  # Moderate strength
                    indicators=f"EMA Crossover: Fast={ema_fast:.2f}, Slow={ema_slow:.2f}, RSI={rsi:.2f}",
                    reason=reason
                )
                signals.append(signal)
                logger.info(f"Generated SELL signal: {reason}")
        
        return signals
    
    def calculate_position_size(self, capital: float, risk_per_trade: float) -> float:
        """Calculate position size based on risk management rules"""
        # Use 1% of capital per trade as default
        position_size = capital * risk_per_trade
        return position_size 