from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from app.db.models import StrategyType, Signal
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, strategy_type: StrategyType):
        self.strategy_type = strategy_type
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on the strategy logic"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, capital: float, risk_per_trade: float) -> float:
        """Calculate the position size based on risk management rules"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the input data has required columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators used by the strategy"""
        indicators = {}
        
        # Calculate basic indicators
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # Store indicators
        indicators['sma_20'] = data['sma_20'].iloc[-1]
        indicators['sma_50'] = data['sma_50'].iloc[-1]
        indicators['sma_200'] = data['sma_200'].iloc[-1]
        indicators['rsi'] = data['rsi'].iloc[-1]
        indicators['macd'] = data['macd'].iloc[-1]
        indicators['signal'] = data['signal'].iloc[-1]
        
        return indicators
    
    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for the strategy"""
        returns = data['close'].pct_change()
        
        metrics = {
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': (data['close'] / data['close'].cummax() - 1).min()
        }
        
        return metrics
    
    def log_strategy_execution(self, symbol: str, signals: List[Signal]):
        """Log strategy execution details"""
        logger.info(f"Strategy {self.name} executed for {symbol}")
        logger.info(f"Generated {len(signals)} signals")
        for signal in signals:
            logger.info(f"Signal: {signal.signal_type} with strength {signal.strength}") 