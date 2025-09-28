# strategies/alpha/technical_alpha.py

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class AlphaSignalType(Enum):
    """Alpha signal types for clear categorization"""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    QUALITY = "quality"


@dataclass
class AlphaSignal:
    """Standardized alpha signal structure"""

    symbol: str
    signal_type: AlphaSignalType
    score: float  # -1 to 1, strength of signal
    confidence: float  # 0 to 1, confidence level
    components: Dict[str, float]  # breakdown of signal components
    timestamp: pd.Timestamp
    metadata: Dict = None


class TechnicalAlphaModule:
    """
    Optimized technical alpha factor calculator with batch processing
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Factor calculation parameters
        self.mean_reversion_window = kwargs.get("mean_reversion_window", 60)
        self.volume_lookback = kwargs.get("volume_lookback", 10)
        self.efficiency_window = kwargs.get("efficiency_window", 20)
        self.volatility_windows = kwargs.get("volatility_windows", [5, 20])
        self.trend_window = kwargs.get("trend_window", 50)

        # Signal generation thresholds
        self.min_confidence = kwargs.get("min_confidence", 0.3)
        self.signal_threshold = kwargs.get("signal_threshold", 0.2)

        # Pre-computed factor weights
        self.factor_weights = {
            "mean_reversion": kwargs.get("mean_reversion_weight", 0.25),
            "volume_profile": kwargs.get("volume_weight", 0.15),
            "price_efficiency": kwargs.get("efficiency_weight", 0.20),
            "volatility": kwargs.get("volatility_weight", 0.15),
            "trend_quality": kwargs.get("trend_weight", 0.25),
        }

        # Normalize weights once
        total_weight = sum(self.factor_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            factor = 1.0 / total_weight
            self.factor_weights = {
                k: v * factor for k, v in self.factor_weights.items()
            }

        # Pre-compute constants
        self.sqrt_252 = np.sqrt(252)

        # Pre-compute signal type mapping
        self.signal_type_map = {
            "mean_reversion": AlphaSignalType.MEAN_REVERSION,
            "volume_profile": AlphaSignalType.VOLUME,
            "price_efficiency": AlphaSignalType.TREND,
            "volatility": AlphaSignalType.VOLATILITY,
            "trend_quality": AlphaSignalType.TREND,
        }

        # Cache for intermediate calculations
        self.calculation_cache = {}

    def calculate_batch_alpha_signals(
        self, preprocessed_data: Dict[str, Dict], timestamp: pd.Timestamp
    ) -> Dict[str, AlphaSignal]:
        """
        Batch calculation of technical alpha signals

        Args:
            preprocessed_data: Dict of symbol -> preprocessed data
            timestamp: Current timestamp

        Returns:
            Dictionary of symbol -> AlphaSignal
        """
        if not preprocessed_data:
            return {}

        alpha_signals = {}

        # Process all symbols in batch
        for symbol, data_dict in preprocessed_data.items():
            try:
                # Skip if insufficient data
                if not self._has_sufficient_data(data_dict):
                    continue

                # Calculate all factors at once
                factors = self._calculate_all_factors_fast(data_dict, symbol)

                if not factors:
                    continue

                # Create signal from factors
                signal = self._create_technical_signal(factors, symbol, timestamp)

                if signal and signal.confidence >= self.min_confidence:
                    alpha_signals[symbol] = signal

            except Exception as e:
                self.logger.debug(
                    f"Error calculating technical alpha for {symbol}: {e}"
                )

        return alpha_signals

    def _has_sufficient_data(self, data_dict: Dict) -> bool:
        """Quick check for sufficient data"""
        if "prices" not in data_dict:
            return False

        prices = data_dict["prices"]
        min_required = max(
            self.mean_reversion_window, self.trend_window, max(self.volatility_windows)
        )

        return len(prices) >= min_required

    def _calculate_all_factors_fast(
        self, symbol_data: Dict, symbol: str
    ) -> Dict[str, Tuple[float, float]]:
        """Fast calculation of all technical factors"""
        factors = {}
        prices = symbol_data["prices"]

        # Pre-calculate commonly used values
        current_price = prices[-1]

        # Mean Reversion - Vectorized
        mr_score, mr_conf = self._calc_mean_reversion_vectorized(prices, current_price)
        if mr_score is not None:
            factors["mean_reversion"] = (mr_score, mr_conf)

        # Volume Profile - Only if volume data exists
        if symbol_data.get("volumes") is not None:
            vol_score, vol_conf = self._calc_volume_profile_vectorized(
                prices, symbol_data["volumes"]
            )
            if vol_score is not None:
                factors["volume_profile"] = (vol_score, vol_conf)

        # Price Efficiency - Vectorized
        eff_score, eff_conf = self._calc_efficiency_vectorized(prices)
        if eff_score is not None:
            factors["price_efficiency"] = (eff_score, eff_conf)

        # Volatility Factor - Vectorized
        vol_factor, vol_conf = self._calc_volatility_vectorized(prices)
        if vol_factor is not None:
            factors["volatility"] = (vol_factor, vol_conf)

        # Trend Quality - Vectorized
        trend_score, trend_conf = self._calc_trend_vectorized(prices)
        if trend_score is not None:
            factors["trend_quality"] = (trend_score, trend_conf)

        return factors

    def _calc_mean_reversion_vectorized(
        self, prices: np.ndarray, current_price: float
    ) -> Tuple[Optional[float], float]:
        """Vectorized mean reversion calculation"""
        if len(prices) < self.mean_reversion_window:
            return None, 0.0

        window = min(self.mean_reversion_window, len(prices))
        prices_window = prices[-window:]

        # Vectorized calculations
        mean = np.mean(prices_window)
        std = np.std(prices_window)

        if std == 0:
            return 0.0, 0.0

        z_score = (current_price - mean) / std

        # Fast confidence calculation
        rolling_mean = pd.Series(prices).rolling(20, min_periods=1).mean()
        mean_stability = 1.0 - np.std(rolling_mean.dropna()) / mean if mean > 0 else 0
        confidence = min(1.0, mean_stability * 1.5)

        # Fast score mapping
        if z_score < -2.5:
            score = 0.8
        elif z_score < -1.5:
            score = 0.6
        elif z_score < -0.5:
            score = 0.3
        elif z_score <= 0.5:
            score = 0.0
        elif z_score <= 1.5:
            score = -0.3
        elif z_score <= 2.5:
            score = -0.6
        else:
            score = -0.8

        return score, confidence

    def _calc_volume_profile_vectorized(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """Vectorized volume profile analysis"""
        lookback = min(self.volume_lookback, len(prices) - 1, len(volumes))

        if lookback < 5:
            return None, 0.0

        # Vectorized price changes and volume
        price_changes = np.diff(prices[-lookback - 1 :]) / prices[-lookback - 1 : -1]
        vol_data = volumes[-lookback:]

        if len(price_changes) != len(vol_data):
            return None, 0.0

        avg_vol = np.mean(vol_data)
        if avg_vol == 0:
            return None, 0.0

        # Vectorized volume-weighted directional movement
        normalized_vol = vol_data / avg_vol

        # Up and down moves
        up_mask = price_changes > 0
        up_score = (
            np.sum(price_changes[up_mask] * normalized_vol[up_mask])
            if np.any(up_mask)
            else 0
        )
        down_score = (
            np.sum(price_changes[~up_mask] * (2 - normalized_vol[~up_mask]))
            if np.any(~up_mask)
            else 0
        )

        total_score = (up_score + down_score) / len(price_changes)

        # Confidence from volume consistency
        vol_consistency = 1.0 - np.std(vol_data) / avg_vol
        confidence = min(1.0, vol_consistency * 0.8 + 0.2)

        # Smooth scaling
        score = np.tanh(total_score * 20)

        return score, confidence

    def _calc_efficiency_vectorized(
        self, prices: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """Vectorized efficiency ratio calculation"""
        if len(prices) < self.efficiency_window:
            return None, 0.0

        prices_array = prices[-self.efficiency_window :]

        # Vectorized calculations
        direction = abs(prices_array[-1] - prices_array[0])
        daily_changes = np.abs(np.diff(prices_array))
        volatility = np.sum(daily_changes)

        if volatility == 0:
            return 0.0, 0.0

        efficiency = direction / volatility

        # Trend consistency for confidence
        positive_changes = np.sum(np.diff(prices_array) > 0)
        trend_consistency = abs(positive_changes - len(daily_changes) / 2) / (
            len(daily_changes) / 2
        )
        confidence = min(1.0, trend_consistency)

        # Fast score mapping
        if efficiency > 0.6:
            score = 1.0
        elif efficiency > 0.4:
            score = 0.5
        elif efficiency > 0.2:
            score = 0.0
        else:
            score = -0.5

        return score, confidence

    def _calc_volatility_vectorized(
        self, prices: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """Vectorized volatility analysis"""
        if len(prices) < max(self.volatility_windows):
            return None, 0.0

        # Vectorized return calculation
        returns = np.diff(prices) / prices[:-1]

        if len(returns) < max(self.volatility_windows):
            return None, 0.0

        short_window = self.volatility_windows[0]
        long_window = self.volatility_windows[1]

        # Vectorized volatility calculation
        recent_vol = np.std(returns[-short_window:]) * self.sqrt_252
        historical_vol = np.std(returns[-long_window:]) * self.sqrt_252

        if historical_vol == 0:
            return 0.0, 0.0

        vol_ratio = recent_vol / historical_vol

        # Confidence from volatility persistence
        vol_persistence = 1.0 - abs(vol_ratio - 1.0)
        confidence = min(1.0, vol_persistence)

        # Fast scoring
        if vol_ratio < 0.5:
            score = 0.8
        elif vol_ratio < 0.7:
            score = 0.4
        elif vol_ratio < 1.3:
            score = 0.0
        elif vol_ratio < 2.0:
            score = -0.4
        else:
            score = -0.8

        return score, confidence

    def _calc_trend_vectorized(
        self, prices: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """Vectorized trend quality calculation"""
        if len(prices) < self.trend_window:
            return None, 0.0

        window = min(self.trend_window, len(prices))
        prices_array = prices[-window:]

        # Vectorized linear regression
        x = np.arange(len(prices_array))

        # Fast polyfit
        coeffs = np.polyfit(x, prices_array, 1)
        slope = coeffs[0]

        # R-squared calculation
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((prices_array - y_pred) ** 2)
        ss_tot = np.sum((prices_array - np.mean(prices_array)) ** 2)

        if ss_tot == 0:
            return 0.0, 0.0

        r_squared = max(0, min(1, 1 - (ss_res / ss_tot)))

        # Normalize slope
        price_range = np.ptp(prices_array)  # max - min
        normalized_slope = (
            slope / price_range * len(prices_array) if price_range > 0 else 0
        )

        # Use R-squared as confidence
        confidence = r_squared

        # Fast scoring based on R-squared and slope
        if r_squared > 0.7:
            if normalized_slope > 0.5:
                score = 1.0
            elif normalized_slope > 0:
                score = 0.5
            elif normalized_slope > -0.5:
                score = -0.5
            else:
                score = -1.0
        elif r_squared > 0.4:
            score = np.sign(normalized_slope) * 0.3
        else:
            score = -0.2

        return score, confidence

    def _create_technical_signal(
        self,
        factors: Dict[str, Tuple[float, float]],
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> Optional[AlphaSignal]:
        """Create technical signal from factors"""
        if not factors:
            return None

        # Fast weighted combination
        total_score = 0.0
        total_confidence = 0.0
        components = {}

        for factor_name, (score, confidence) in factors.items():
            weight = self.factor_weights.get(factor_name, 0.0)
            total_score += score * weight
            total_confidence += confidence * weight
            components[factor_name] = score

        # Early exit for weak signals
        if abs(total_score) < self.signal_threshold:
            return None

        # Determine dominant factor
        dominant_factor = max(
            factors.items(),
            key=lambda x: abs(x[1][0]) * self.factor_weights.get(x[0], 0),
        )[0]

        signal_type = self.signal_type_map.get(dominant_factor, AlphaSignalType.TREND)

        return AlphaSignal(
            symbol=symbol,
            signal_type=signal_type,
            score=total_score,
            confidence=total_confidence,
            components=components,
            timestamp=timestamp,
            metadata={"dominant_factor": dominant_factor, "factor_count": len(factors)},
        )

    # Legacy interface for backward compatibility
    def calculate_alpha_signals(
        self, market_data: pd.DataFrame, symbols: List[str], timestamp: pd.Timestamp
    ) -> Dict[str, AlphaSignal]:
        """Legacy interface - converts to batch format"""
        preprocessed_data = {}

        for symbol in symbols:
            try:
                if symbol in market_data.index.get_level_values("symbol"):
                    symbol_data = market_data.xs(symbol, level="symbol")
                    symbol_data = symbol_data[symbol_data.index <= timestamp]

                    if len(symbol_data) >= 20:
                        preprocessed_data[symbol] = {
                            "prices": symbol_data["close"].values,
                            "volumes": (
                                symbol_data["volume"].values
                                if "volume" in symbol_data
                                else None
                            ),
                            "highs": (
                                symbol_data["high"].values
                                if "high" in symbol_data
                                else None
                            ),
                            "lows": (
                                symbol_data["low"].values
                                if "low" in symbol_data
                                else None
                            ),
                            "dataframe": symbol_data,
                        }
            except Exception as e:
                self.logger.debug(f"Error extracting data for {symbol}: {e}")

        return self.calculate_batch_alpha_signals(preprocessed_data, timestamp)
