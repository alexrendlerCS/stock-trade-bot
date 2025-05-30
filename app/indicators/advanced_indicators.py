import pandas as pd
import numpy as np
from typing import Dict, Optional
import ta
import logging
from .insider_trading import InsiderTracker

logger = logging.getLogger(__name__)

class AdvancedIndicators:
    """Advanced technical and market indicators for enhanced trading confidence"""
    
    def __init__(self):
        self.vix_threshold = 20  # VIX threshold for high volatility
        self.volume_ma_period = 20  # Period for volume moving average
        self.insider_tracker = InsiderTracker()  # Initialize insider tracking
    
    def calculate_market_regime(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect market regime (trending vs ranging)
        Returns regime strength between -1 (strong downtrend) to 1 (strong uptrend)
        """
        try:
            # Calculate ADX for trend strength
            adx = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            
            # Calculate DMI (Directional Movement Index)
            dmi_pos = adx.adx_pos()
            dmi_neg = adx.adx_neg()
            
            # Get latest values
            adx_value = adx.adx().iloc[-1]
            dmi_pos_value = dmi_pos.iloc[-1]
            dmi_neg_value = dmi_neg.iloc[-1]
            
            # Determine regime
            if adx_value > 25:  # Strong trend
                if dmi_pos_value > dmi_neg_value:
                    regime_type = 'strong_uptrend'
                    strength = min(adx_value / 100, 1.0)
                else:
                    regime_type = 'strong_downtrend'
                    strength = -min(adx_value / 100, 1.0)
            else:  # Ranging market
                regime_type = 'ranging'
                strength = 0.0
            
            return {
                'regime_type': regime_type,
                'strength': strength,
                'adx': adx_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {str(e)}")
            return {'regime_type': 'unknown', 'strength': 0.0, 'adx': 0.0}
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze volume profile and money flow
        Returns volume-based confidence metrics
        """
        try:
            # Calculate Money Flow Index
            mfi = ta.volume.MFIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=14
            ).money_flow_index()
            
            # Volume moving average
            volume_ma = df['volume'].rolling(window=self.volume_ma_period).mean()
            
            # Volume trend (comparing current to moving average)
            current_volume = df['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # On-Balance Volume
            obv = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()
            
            # Calculate OBV trend
            obv_trend = (obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) if abs(obv.iloc[-5]) > 0 else 0
            
            return {
                'mfi': mfi.iloc[-1],
                'volume_ratio': volume_ratio,
                'obv_trend': obv_trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {'mfi': 50.0, 'volume_ratio': 1.0, 'obv_trend': 0.0}
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Calculate dynamic support and resistance levels
        Returns distance from current price to levels
        """
        try:
            # Calculate pivot points
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            pivot = typical_price.rolling(window=window).mean()
            
            # Calculate support and resistance
            r1 = 2 * pivot - df['low']
            s1 = 2 * pivot - df['high']
            
            current_price = df['close'].iloc[-1]
            
            # Calculate distances
            dist_to_r1 = (r1.iloc[-1] - current_price) / current_price
            dist_to_s1 = (current_price - s1.iloc[-1]) / current_price
            
            return {
                'resistance_distance': dist_to_r1,
                'support_distance': dist_to_s1,
                'pivot': pivot.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {'resistance_distance': 0.0, 'support_distance': 0.0, 'pivot': 0.0}
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate advanced volatility metrics
        Returns volatility-based confidence indicators
        """
        try:
            # ATR (Average True Range)
            atr = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).average_true_range()
            
            # Keltner Channel
            kc = ta.volatility.KeltnerChannel(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=20
            )
            
            current_price = df['close'].iloc[-1]
            kc_mid = kc.keltner_channel_mband().iloc[-1]
            kc_high = kc.keltner_channel_hband().iloc[-1]
            kc_low = kc.keltner_channel_lband().iloc[-1]
            
            # Position within channel
            kc_position = (current_price - kc_low) / (kc_high - kc_low) if (kc_high - kc_low) > 0 else 0.5
            
            return {
                'atr': atr.iloc[-1],
                'atr_ratio': atr.iloc[-1] / current_price,
                'kc_position': kc_position
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {'atr': 0.0, 'atr_ratio': 0.0, 'kc_position': 0.5}
    
    def enhance_confidence(self, 
                         df: pd.DataFrame, 
                         base_confidence: float,
                         direction: str,
                         symbol: str = None) -> float:
        """
        Enhance the base confidence score using advanced indicators
        Returns adjusted confidence between 0 and 1
        """
        try:
            # Calculate all metrics
            regime = self.calculate_market_regime(df)
            volume = self.calculate_volume_profile(df)
            sr_levels = self.calculate_support_resistance(df)
            volatility = self.calculate_volatility_metrics(df)
            
            # Initialize confidence adjustments
            adjustments = 0.0
            
            # 1. Market Regime Adjustment
            if direction == 'LONG' and regime['strength'] > 0:
                adjustments += regime['strength'] * 0.1
            elif direction == 'SHORT' and regime['strength'] < 0:
                adjustments += abs(regime['strength']) * 0.1
            
            # 2. Volume Profile Adjustment
            if volume['volume_ratio'] > 1.2:  # Higher than normal volume
                if direction == 'LONG' and volume['mfi'] > 60:
                    adjustments += 0.05
                elif direction == 'SHORT' and volume['mfi'] < 40:
                    adjustments += 0.05
            
            # 3. Support/Resistance Adjustment
            if direction == 'LONG' and sr_levels['support_distance'] < 0.02:
                adjustments += 0.05  # Near support
            elif direction == 'SHORT' and sr_levels['resistance_distance'] < 0.02:
                adjustments += 0.05  # Near resistance
            
            # 4. Volatility Adjustment
            if volatility['atr_ratio'] < 0.02:  # Low volatility
                adjustments += 0.05
            
            # 5. Insider Trading Adjustment
            if symbol:
                insider_adj = self.insider_tracker.get_confidence_adjustment(symbol, direction)
                adjustments += insider_adj
                logger.info(f"Insider trading adjustment for {symbol}: {insider_adj:.3f}")
            
            # Apply adjustments to base confidence
            enhanced_confidence = base_confidence + adjustments
            
            # Ensure confidence stays between 0 and 1
            enhanced_confidence = max(0.0, min(1.0, enhanced_confidence))
            
            logger.info(f"Enhanced confidence from {base_confidence:.3f} to {enhanced_confidence:.3f}")
            logger.info(f"Adjustments: {adjustments:.3f}")
            
            return enhanced_confidence
            
        except Exception as e:
            logger.error(f"Error enhancing confidence: {str(e)}")
            return base_confidence  # Return original confidence if error occurs 