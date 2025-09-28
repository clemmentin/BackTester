import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .technical_alpha import AlphaSignal, AlphaSignalType


class MomentumAlphaModule:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Multi-timeframe momentum windows
        self.momentum_windows = {
            "ultra_short": kwargs.get("momentum_ultra_short", 5),
            "short": kwargs.get("momentum_short", 10),
            "medium": kwargs.get("momentum_medium", 20),
            "long": kwargs.get("momentum_long", 60),
            "ultra_long": kwargs.get("momentum_ultra_long", 120),
        }

        # Momentum calculation parameters
        self.min_momentum_threshold = kwargs.get("min_momentum_threshold", -0.15)
        self.momentum_acceleration = kwargs.get("momentum_acceleration", True)
        self.volume_confirmation = kwargs.get("volume_confirmation", True)
        self.relative_strength = kwargs.get("relative_strength", True)

        # Pre-computed timeframe weights
        self.timeframe_weights = {
            "ultra_short": kwargs.get("weight_ultra_short", 0.10),
            "short": kwargs.get("weight_short", 0.25),
            "medium": kwargs.get("weight_medium", 0.35),
            "long": kwargs.get("weight_long", 0.20),
            "ultra_long": kwargs.get("weight_ultra_long", 0.10),
        }

        # Normalize weights once
        total_weight = sum(self.timeframe_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.timeframe_weights = {
                k: v / total_weight for k, v in self.timeframe_weights.items()
            }

        # Risk adjustment parameters
        self.volatility_scaling = kwargs.get("volatility_scaling", True)
        self.max_volatility_threshold = kwargs.get("max_volatility_threshold", 0.40)
        self.target_volatility = kwargs.get("target_volatility", 0.15)

        # Signal filtering
        self.min_confidence = kwargs.get("min_confidence", 0.3)
        self.require_positive_medium = kwargs.get("require_positive_medium", True)

        # Pre-compute constants
        self.sqrt_252 = np.sqrt(252)

        # Cache for market momentum
        self.market_momentum_cache = {}
        self.cache_timestamp = None

        # Pre-compute regime multipliers
        self.regime_multipliers = {
            "BULL": 1.2,
            "STRONG_BULL": 1.3,
            "NORMAL": 1.0,
            "BEAR": 0.7,
            "CRISIS": 0.5,
        }

    def calculate_batch_momentum_signals(
        self,
        preprocessed_data: Dict[str, Dict],
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, AlphaSignal]:
        if not preprocessed_data:
            return {}

        # Update market momentum once for all symbols
        if self.relative_strength:
            self._update_market_momentum_batch(preprocessed_data, timestamp)

        momentum_signals = {}

        # Batch compute all momentum scores
        all_momentum_data = {}

        for symbol, data_dict in preprocessed_data.items():
            try:
                prices = data_dict["prices"]

                # Skip if insufficient data
                min_required = max(self.momentum_windows.values())
                if len(prices) < min_required:
                    continue

                # Calculate all timeframe momentums at once
                momentum_scores = self._calculate_all_momentums_vectorized(prices)

                if momentum_scores:
                    all_momentum_data[symbol] = {
                        "scores": momentum_scores,
                        "prices": prices,
                        "volumes": data_dict.get("volumes"),
                    }

            except Exception as e:
                self.logger.debug(f"Error calculating momentum for {symbol}: {e}")

        # Now compute signals with all data available
        regime_multiplier = self.regime_multipliers.get(market_regime, 1.0)

        for symbol, mom_data in all_momentum_data.items():
            signal = self._create_momentum_signal(
                symbol, mom_data, timestamp, market_regime, regime_multiplier
            )

            if signal and signal.confidence >= self.min_confidence:
                momentum_signals[symbol] = signal

        return momentum_signals

    def _calculate_all_momentums_vectorized(
        self, prices: np.ndarray
    ) -> Dict[str, float]:
        """Vectorized momentum calculation for all timeframes"""
        momentum_scores = {}
        current_price = prices[-1]

        # Calculate all momentums in one pass
        for timeframe, window in self.momentum_windows.items():
            if len(prices) >= window:
                momentum_scores[timeframe] = (current_price / prices[-window]) - 1

        return momentum_scores

    def _create_momentum_signal(
        self,
        symbol: str,
        mom_data: Dict,
        timestamp: pd.Timestamp,
        market_regime: str,
        regime_multiplier: float,
    ) -> Optional[AlphaSignal]:
        """Create momentum signal from computed data"""
        momentum_scores = mom_data["scores"]
        prices = mom_data["prices"]
        volumes = mom_data["volumes"]

        # Fast factor calculations
        acceleration_factor = 1.0
        if self.momentum_acceleration:
            acceleration_factor = self._calculate_acceleration_fast(momentum_scores)

        volume_factor = 1.0
        if self.volume_confirmation and volumes is not None:
            volume_factor = self._calculate_volume_confirmation_fast(prices, volumes)

        relative_strength_factor = 1.0
        if self.relative_strength:
            relative_strength_factor = self._calculate_relative_strength_fast(
                momentum_scores, symbol
            )

        volatility_factor = 1.0
        if self.volatility_scaling:
            volatility_factor = self._calculate_volatility_adjustment_fast(prices)

        # Combine factors efficiently
        base_score = sum(
            momentum_scores.get(tf, 0) * self.timeframe_weights.get(tf, 0)
            for tf in self.momentum_windows.keys()
        )

        # Apply all factors at once
        composite_score = (
            base_score
            * acceleration_factor
            * volume_factor
            * relative_strength_factor
            * volatility_factor
            * regime_multiplier
        )

        # Quick confidence calculation
        positive_count = sum(1 for v in momentum_scores.values() if v > 0)
        consistency = positive_count / len(momentum_scores) if momentum_scores else 0

        # Factor alignment
        factors = [
            acceleration_factor,
            volume_factor,
            relative_strength_factor,
            volatility_factor,
        ]
        factor_mean = sum(factors) / len(factors)
        factor_alignment = (factor_mean - 1.0 + 0.5) * 0.4

        confidence = consistency * 0.6 + factor_alignment
        confidence = np.clip(confidence, 0, 1)

        # Apply medium momentum filter if required
        if self.require_positive_medium and momentum_scores.get("medium", 0) < 0:
            composite_score *= 0.5

        # Clip final score
        composite_score = np.clip(composite_score, -1, 1)

        # Create signal
        components = {
            "momentum_composite": composite_score,
            "acceleration": acceleration_factor,
            "volume_confirmation": volume_factor,
            "relative_strength": relative_strength_factor,
            "volatility_adjustment": volatility_factor,
        }
        components.update({f"momentum_{k}": v for k, v in momentum_scores.items()})

        return AlphaSignal(
            symbol=symbol,
            signal_type=AlphaSignalType.MOMENTUM,
            score=composite_score,
            confidence=confidence,
            components=components,
            timestamp=timestamp,
            metadata={
                "market_regime": market_regime,
                "timeframes_positive": positive_count,
                "strongest_timeframe": max(
                    momentum_scores.items(), key=lambda x: abs(x[1])
                )[0],
            },
        )

    def _calculate_acceleration_fast(self, momentum_scores: Dict[str, float]) -> float:
        """Fast acceleration calculation"""
        if len(momentum_scores) < 3:
            return 1.0

        # Pre-computed groupings
        shorter_scores = [
            momentum_scores.get("ultra_short", 0),
            momentum_scores.get("short", 0),
        ]
        longer_scores = [
            momentum_scores.get("long", 0),
            momentum_scores.get("ultra_long", 0),
        ]

        shorter_avg = sum(shorter_scores) / len(shorter_scores)
        longer_avg = sum(longer_scores) / len(longer_scores)

        if abs(longer_avg) > 1e-6:
            acceleration = shorter_avg / abs(longer_avg)
            return np.clip(0.5 + acceleration * 0.5, 0.5, 1.5)

        return 1.0

    def _calculate_volume_confirmation_fast(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> float:
        """Fast volume confirmation"""
        window = min(20, len(prices) - 1, len(volumes))
        if window < 5:
            return 1.0

        # Use slicing for efficiency
        price_changes = np.diff(prices[-window - 1 :])
        volume_data = volumes[-window:]

        if len(price_changes) != len(volume_data):
            return 1.0

        avg_volume = np.mean(volume_data)
        if avg_volume == 0:
            return 1.0

        # Vectorized calculation
        up_mask = price_changes > 0
        up_volume = np.mean(volume_data[up_mask]) if np.any(up_mask) else avg_volume
        down_volume = np.mean(volume_data[~up_mask]) if np.any(~up_mask) else avg_volume

        if down_volume > 0:
            volume_ratio = up_volume / down_volume
            return np.clip(0.8 + (volume_ratio - 1) * 0.2, 0.8, 1.2)

        return 1.1

    def _calculate_relative_strength_fast(
        self, momentum_scores: Dict[str, float], symbol: str
    ) -> float:
        """Fast relative strength calculation"""
        if not self.market_momentum_cache:
            return 1.0

        symbol_avg_momentum = sum(momentum_scores.values()) / len(momentum_scores)
        market_avg_momentum = self.market_momentum_cache.get("market_average", 0)

        if abs(market_avg_momentum) < 1e-6:
            return 1.0

        relative_strength = symbol_avg_momentum - market_avg_momentum
        return np.clip(1.0 + relative_strength, 0.7, 1.3)

    def _calculate_volatility_adjustment_fast(self, prices: np.ndarray) -> float:
        """Fast volatility adjustment"""
        if len(prices) < 20:
            return 1.0

        # Vectorized return calculation
        returns = np.diff(prices[-20:]) / prices[-20:-1]
        volatility = np.std(returns) * self.sqrt_252

        # Pre-computed thresholds
        if volatility > self.max_volatility_threshold:
            return 0.5
        elif volatility > self.target_volatility * 2:
            return 0.7
        elif volatility < self.target_volatility * 0.5:
            return 1.2
        else:
            return 1.0

    def _update_market_momentum_batch(
        self, preprocessed_data: Dict, timestamp: pd.Timestamp
    ):
        """Batch update market momentum statistics"""
        # Only update if needed (cache for 5 minutes)
        if (
            self.cache_timestamp
            and (timestamp - self.cache_timestamp).total_seconds() < 300
        ):
            return

        # Sample first 20 symbols for efficiency
        sample_symbols = list(preprocessed_data.keys())[:20]

        if not sample_symbols:
            return

        # Vectorized calculation
        all_momentums = []
        medium_window = self.momentum_windows["medium"]

        for symbol in sample_symbols:
            data = preprocessed_data[symbol]
            prices = data["prices"]

            if len(prices) >= medium_window:
                momentum = (prices[-1] / prices[-medium_window]) - 1
                all_momentums.append(momentum)

        if all_momentums:
            self.market_momentum_cache = {
                "market_average": np.mean(all_momentums),
                "market_median": np.median(all_momentums),
                "market_std": np.std(all_momentums),
                "timestamp": timestamp,
            }
            self.cache_timestamp = timestamp

    # Legacy interface for backward compatibility
    def calculate_momentum_signals(
        self,
        market_data: pd.DataFrame,
        symbols: List[str],
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, AlphaSignal]:
        """Legacy interface - converts to batch format"""
        preprocessed_data = {}

        for symbol in symbols:
            try:
                if symbol in market_data.index.get_level_values("symbol"):
                    symbol_data = market_data.xs(symbol, level="symbol")
                    symbol_data = symbol_data[symbol_data.index <= timestamp]

                    min_required = max(self.momentum_windows.values())
                    if len(symbol_data) >= min_required:
                        preprocessed_data[symbol] = {
                            "prices": symbol_data["close"].values,
                            "volumes": (
                                symbol_data["volume"].values
                                if "volume" in symbol_data
                                else None
                            ),
                        }
            except Exception as e:
                self.logger.debug(f"Error extracting data for {symbol}: {e}")

        return self.calculate_batch_momentum_signals(
            preprocessed_data, timestamp, market_regime
        )
