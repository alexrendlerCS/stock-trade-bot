from typing import List
import pandas as pd
from app.strategies.base_strategy import BaseStrategy
from app.db.models import StrategyType, Signal
from app.services.ml_predictor import MLPredictor
import logging

logger = logging.getLogger(__name__)

class MLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(StrategyType.SWING_TRADE)
        self.predictor = MLPredictor()
        self.probability_threshold = 0.6  # Minimum probability to generate a signal
    
    def get_trade_reason(self, signal_type: str, prob_3d: float, prob_5d: float) -> str:
        """Generate a detailed explanation for the trade."""
        if signal_type == 'BUY':
            return (
                f"BUY: ML model predicts upward movement with "
                f"3-day probability={prob_3d:.2%} and "
                f"5-day probability={prob_5d:.2%}"
            )
        elif signal_type == 'SELL':
            return (
                f"SELL: ML model predicts downward movement with "
                f"3-day probability={prob_3d:.2%} and "
                f"5-day probability={prob_5d:.2%}"
            )
        else:
            return "Unknown signal type."

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on ML predictions"""
        if not self.validate_data(data):
            logger.error("Invalid data format for ML strategy")
            return []
        
        signals = []
        
        try:
            # Get ML predictions
            predictions = self.predictor.predict(data)
            if not predictions:
                return []
            
            prob_3d_up, prob_3d_down = predictions['3d']
            prob_5d_up, prob_5d_down = predictions['5d']
            
            # Generate signals based on prediction probabilities
            if prob_3d_up > self.probability_threshold and prob_5d_up > self.probability_threshold:
                # Strong buy signal
                reason = self.get_trade_reason('BUY', prob_3d_up, prob_5d_up)
                signal = Signal(
                    symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                    strategy_type=self.strategy_type,
                    signal_type='BUY',
                    strength=prob_3d_up,  # Use probability as signal strength
                    indicators=f"ML Prediction: 3d_up={prob_3d_up:.2%}, 5d_up={prob_5d_up:.2%}",
                    reason=reason
                )
                signals.append(signal)
                logger.info(f"Generated BUY signal: {reason}")
            
            elif prob_3d_down > self.probability_threshold and prob_5d_down > self.probability_threshold:
                # Strong sell signal
                reason = self.get_trade_reason('SELL', prob_3d_down, prob_5d_down)
                signal = Signal(
                    symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                    strategy_type=self.strategy_type,
                    signal_type='SELL',
                    strength=prob_3d_down,  # Use probability as signal strength
                    indicators=f"ML Prediction: 3d_down={prob_3d_down:.2%}, 5d_down={prob_5d_down:.2%}",
                    reason=reason
                )
                signals.append(signal)
                logger.info(f"Generated SELL signal: {reason}")
        
        except Exception as e:
            logger.error(f"Error generating ML signals: {str(e)}")
        
        return signals
    
    def calculate_position_size(self, capital: float, risk_per_trade: float) -> float:
        """Calculate position size based on risk management rules"""
        # Use 1% of capital per trade as default
        position_size = capital * risk_per_trade
        return position_size 