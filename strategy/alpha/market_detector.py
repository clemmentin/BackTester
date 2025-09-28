# strategies/alpha/market_detector.py

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime classifications"""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NORMAL = "normal"
    BEAR = "bear"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    VOLATILE = "volatile"


@dataclass
class MarketState:
    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    trend_strength: float
    volatility_regime: str  # 'low', 'normal', 'elevated', 'high'
    breadth: float  # Market breadth (% of advancing symbols)
    timestamp: pd.Timestamp


class MarketDetector:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Detection parameters
        self.lookback_short = kwargs.get("lookback_short", 20)
        self.lookback_medium = kwargs.get("lookback_medium", 50)
        self.lookback_long = kwargs.get("lookback_long", 200)

        # Regime thresholds
        self.bull_threshold = kwargs.get("bull_threshold", 0.05)
        self.strong_bull_threshold = kwargs.get("strong_bull_threshold", 0.15)
        self.bear_threshold = kwargs.get("bear_threshold", -0.05)
        self.crisis_threshold = kwargs.get("crisis_threshold", -0.15)
        self.recovery_threshold = kwargs.get("recovery_threshold", 0.10)

        # Volatility thresholds
        self.normal_volatility = kwargs.get("normal_volatility", 0.15)
        self.high_volatility = kwargs.get("high_volatility", 0.30)
        self.extreme_volatility = kwargs.get("extreme_volatility", 0.50)

        # Market breadth parameters
        self.breadth_bull_threshold = kwargs.get("breadth_bull", 0.65)
        self.breadth_bear_threshold = kwargs.get("breadth_bear", 0.35)

        # Early warning system
        self.enable_early_warning = kwargs.get("enable_early_warning", True)
        self.divergence_threshold = kwargs.get("divergence_threshold", 0.25)

        # State tracking
        self.regime_history = deque(maxlen=kwargs.get("history_length", 30))
        self.volatility_history = deque(maxlen=kwargs.get("history_length", 30))
        self.breadth_history = deque(maxlen=kwargs.get("history_length", 30))

        # Cache
        self.last_detection_time = None
        self.last_market_state = None
        self.early_warning_active = False

    def detect_market_state(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        symbols: List[str] = None,
    ) -> MarketState:
        # Use cache if recently calculated
        if (
            self.last_detection_time
            and self.last_market_state
            and (timestamp - self.last_detection_time).seconds < 300
        ):  # 5 min cache
            return self.last_market_state

        # Extract market index data (SPY or similar)
        index_analysis = self._analyze_market_index(market_data, timestamp)

        # Calculate market breadth
        market_breadth = self._calculate_market_breadth(market_data, timestamp, symbols)

        # Analyze volatility regime
        volatility_analysis = self._analyze_volatility_regime(market_data, timestamp)

        # Detect divergences and early warnings
        warnings = self._detect_early_warnings(
            index_analysis, market_breadth, volatility_analysis
        )

        # Combine all factors to determine regime
        regime, confidence = self._determine_market_regime(
            index_analysis, market_breadth, volatility_analysis, warnings
        )

        # Determine risk level
        risk_level = self._assess_risk_level(regime, volatility_analysis, warnings)

        # Create market state
        market_state = MarketState(
            regime=regime,
            confidence=confidence,
            indicators={
                "index_return_20d": index_analysis.get("return_20d", 0),
                "index_return_50d": index_analysis.get("return_50d", 0),
                "trend_strength": index_analysis.get("trend_strength", 0),
                "volatility": volatility_analysis.get("current_vol", 0),
                "volatility_ratio": volatility_analysis.get("vol_ratio", 1),
                "market_breadth": market_breadth,
                "drawdown": index_analysis.get("drawdown", 0),
                "divergence_score": warnings.get("divergence_score", 0),
            },
            risk_level=risk_level,
            trend_strength=index_analysis.get("trend_strength", 0),
            volatility_regime=volatility_analysis.get("regime", "normal"),
            breadth=market_breadth,
            timestamp=timestamp,
        )

        # Update history and cache
        self._update_history(market_state)
        self.last_detection_time = timestamp
        self.last_market_state = market_state

        return market_state

    def _analyze_market_index(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """Analyze market index (SPY) for regime detection"""
        index_symbol = "SPY"  # Could be parameterized

        try:
            if index_symbol in market_data.index.get_level_values("symbol"):
                index_data = market_data.xs(index_symbol, level="symbol")
                index_data = index_data[index_data.index <= timestamp]

                if len(index_data) < self.lookback_long:
                    return self._default_index_analysis()

                prices = index_data["close"].values
                current_price = prices[-1]

                # Calculate returns
                ret_20d = (
                    (current_price / prices[-self.lookback_short]) - 1
                    if len(prices) >= self.lookback_short
                    else 0
                )
                ret_50d = (
                    (current_price / prices[-self.lookback_medium]) - 1
                    if len(prices) >= self.lookback_medium
                    else 0
                )
                ret_200d = (current_price / prices[-self.lookback_long]) - 1

                # Calculate moving averages
                ma_50 = (
                    np.mean(prices[-self.lookback_medium :])
                    if len(prices) >= self.lookback_medium
                    else current_price
                )
                ma_200 = np.mean(prices[-self.lookback_long :])

                # Trend strength (position relative to MAs)
                trend_strength = 0.0
                if current_price > ma_200:
                    trend_strength += 0.5
                if current_price > ma_50:
                    trend_strength += 0.3
                if ma_50 > ma_200:
                    trend_strength += 0.2

                # Calculate drawdown
                recent_high = (
                    np.max(prices[-self.lookback_medium :])
                    if len(prices) >= self.lookback_medium
                    else current_price
                )
                drawdown = (current_price - recent_high) / recent_high

                # Momentum indicators
                momentum_score = ret_20d * 0.5 + ret_50d * 0.3 + ret_200d * 0.2

                return {
                    "return_20d": ret_20d,
                    "return_50d": ret_50d,
                    "return_200d": ret_200d,
                    "ma_50": ma_50,
                    "ma_200": ma_200,
                    "trend_strength": trend_strength,
                    "drawdown": drawdown,
                    "momentum_score": momentum_score,
                    "price_vs_ma200": (current_price - ma_200) / ma_200,
                }

        except Exception as e:
            self.logger.error(f"Error analyzing market index: {e}")

        return self._default_index_analysis()

    def _calculate_market_breadth(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        symbols: List[str] = None,
    ) -> float:
        """Calculate market breadth (advancing vs declining)"""
        if symbols is None:
            symbols = market_data.index.get_level_values("symbol").unique()[
                :50
            ]  # Top 50

        advancing = 0
        declining = 0

        for symbol in symbols:
            try:
                if symbol in market_data.index.get_level_values("symbol"):
                    symbol_data = market_data.xs(symbol, level="symbol")
                    symbol_data = symbol_data[symbol_data.index <= timestamp]

                    if len(symbol_data) >= self.lookback_short:
                        prices = symbol_data["close"].values
                        ret = (prices[-1] / prices[-self.lookback_short]) - 1

                        if ret > 0.02:  # 2% threshold
                            advancing += 1
                        elif ret < -0.02:
                            declining += 1
            except:
                continue

        total = advancing + declining
        if total > 0:
            breadth = advancing / total
            self.breadth_history.append(breadth)
            return breadth

        return 0.5  # Neutral if no data

    def _analyze_volatility_regime(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """Analyze volatility regime"""
        index_symbol = "SPY"

        try:
            if index_symbol in market_data.index.get_level_values("symbol"):
                index_data = market_data.xs(index_symbol, level="symbol")
                index_data = index_data[index_data.index <= timestamp]

                if len(index_data) >= self.lookback_short:
                    prices = index_data["close"].values

                    price_window = prices[-self.lookback_short :]
                    if len(price_window) > 1:
                        returns = np.diff(price_window) / price_window[:-1]
                        current_vol = np.std(returns) * np.sqrt(252)
                    else:
                        current_vol = self.normal_volatility

                    if len(prices) >= self.lookback_medium:
                        hist_window = prices[-self.lookback_medium :]
                        if len(hist_window) > 1:
                            hist_returns = np.diff(hist_window) / hist_window[:-1]
                            historical_vol = np.std(hist_returns) * np.sqrt(252)
                        else:
                            historical_vol = current_vol
                    else:
                        historical_vol = current_vol

                    vol_ratio = (
                        current_vol / historical_vol if historical_vol > 0 else 1.0
                    )

                    # Determine volatility regime
                    if current_vol < self.normal_volatility * 0.7:
                        regime = "low"
                    elif current_vol < self.normal_volatility * 1.3:
                        regime = "normal"
                    elif current_vol < self.high_volatility:
                        regime = "elevated"
                    else:
                        regime = "high"

                    self.volatility_history.append(current_vol)

                    return {
                        "current_vol": current_vol,
                        "historical_vol": historical_vol,
                        "vol_ratio": vol_ratio,
                        "regime": regime,
                        "vol_percentile": self._calculate_vol_percentile(current_vol),
                    }

        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")

        return {
            "current_vol": self.normal_volatility,
            "historical_vol": self.normal_volatility,
            "vol_ratio": 1.0,
            "regime": "normal",
            "vol_percentile": 50,
        }

    def _detect_early_warnings(
        self, index_analysis: Dict, market_breadth: float, volatility_analysis: Dict
    ) -> Dict:
        """Detect early warning signals"""
        warnings = {
            "divergence_score": 0,
            "breadth_warning": False,
            "volatility_spike": False,
            "trend_breakdown": False,
        }

        if not self.enable_early_warning:
            return warnings

        # Breadth divergence
        if index_analysis["return_20d"] > 0 and market_breadth < 0.4:
            warnings["breadth_warning"] = True
            warnings["divergence_score"] += 0.3

        # Volatility spike
        if volatility_analysis["vol_ratio"] > 2.0:
            warnings["volatility_spike"] = True
            warnings["divergence_score"] += 0.3

        # Trend breakdown
        if (
            index_analysis["trend_strength"] < 0.3
            and index_analysis["drawdown"] < -0.05
        ):
            warnings["trend_breakdown"] = True
            warnings["divergence_score"] += 0.4

        # Update early warning status
        if warnings["divergence_score"] > self.divergence_threshold:
            if not self.early_warning_active:
                self.logger.warning(f"Early warning activated: {warnings}")
            self.early_warning_active = True
        else:
            self.early_warning_active = False

        return warnings

    def _determine_market_regime(
        self,
        index_analysis: Dict,
        market_breadth: float,
        volatility_analysis: Dict,
        warnings: Dict,
    ) -> Tuple[MarketRegime, float]:
        """Determine market regime from all factors"""

        # Base regime from returns
        ret_20d = index_analysis["return_20d"]
        ret_50d = index_analysis["return_50d"]
        drawdown = index_analysis["drawdown"]

        # Initial classification
        if (
            drawdown < self.crisis_threshold
            or volatility_analysis["current_vol"] > self.extreme_volatility
        ):
            base_regime = MarketRegime.CRISIS
        elif ret_20d > self.strong_bull_threshold and ret_50d > self.bull_threshold:
            base_regime = MarketRegime.STRONG_BULL
        elif (
            ret_20d > self.bull_threshold
            and market_breadth > self.breadth_bull_threshold
        ):
            base_regime = MarketRegime.BULL
        elif (
            ret_20d < self.bear_threshold
            or market_breadth < self.breadth_bear_threshold
        ):
            base_regime = MarketRegime.BEAR
        elif volatility_analysis["current_vol"] > self.high_volatility:
            base_regime = MarketRegime.VOLATILE
        elif ret_20d > 0 and drawdown < -0.10:  # Recovering from drawdown
            base_regime = MarketRegime.RECOVERY
        else:
            base_regime = MarketRegime.NORMAL

        # Adjust for warnings
        if warnings["divergence_score"] > self.divergence_threshold:
            if base_regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]:
                base_regime = MarketRegime.NORMAL  # Downgrade
            elif base_regime == MarketRegime.NORMAL:
                base_regime = MarketRegime.BEAR  # Downgrade

        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            base_regime, index_analysis, market_breadth, volatility_analysis
        )

        return base_regime, confidence

    def _calculate_regime_confidence(
        self,
        regime: MarketRegime,
        index_analysis: Dict,
        market_breadth: float,
        volatility_analysis: Dict,
    ) -> float:
        """Calculate confidence in regime detection"""
        confidence_factors = []

        # Trend alignment
        if regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
            if index_analysis["trend_strength"] > 0.7:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.5)

        # Breadth confirmation
        if regime == MarketRegime.BULL and market_breadth > 0.6:
            confidence_factors.append(1.0)
        elif regime == MarketRegime.BEAR and market_breadth < 0.4:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)

        # Volatility consistency
        if volatility_analysis["regime"] == "normal":
            confidence_factors.append(0.8)
        elif volatility_analysis["regime"] in ["low", "elevated"]:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)

        # Historical consistency (if we have history)
        if len(self.regime_history) >= 5:
            recent_regimes = [s.regime for s in list(self.regime_history)[-5:]]
            consistency = sum(1 for r in recent_regimes if r == regime) / 5
            confidence_factors.append(consistency)

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _assess_risk_level(
        self, regime: MarketRegime, volatility_analysis: Dict, warnings: Dict
    ) -> str:
        """Assess overall market risk level"""

        risk_score = 0

        # Regime risk
        regime_risk = {
            MarketRegime.CRISIS: 1.0,
            MarketRegime.BEAR: 0.7,
            MarketRegime.VOLATILE: 0.6,
            MarketRegime.NORMAL: 0.3,
            MarketRegime.RECOVERY: 0.4,
            MarketRegime.BULL: 0.2,
            MarketRegime.STRONG_BULL: 0.1,
        }
        risk_score += regime_risk.get(regime, 0.5) * 0.4

        # Volatility risk
        vol_risk = {"low": 0.1, "normal": 0.3, "elevated": 0.6, "high": 1.0}
        risk_score += vol_risk.get(volatility_analysis["regime"], 0.5) * 0.3

        # Warning risk
        risk_score += warnings["divergence_score"] * 0.3

        # Classify risk level
        if risk_score < 0.25:
            return "low"
        elif risk_score < 0.5:
            return "medium"
        elif risk_score < 0.75:
            return "high"
        else:
            return "extreme"

    def _update_history(self, market_state: MarketState):
        """Update historical tracking"""
        self.regime_history.append(market_state)

    def _default_index_analysis(self) -> Dict[str, float]:
        """Return default analysis when data is insufficient"""
        return {
            "return_20d": 0,
            "return_50d": 0,
            "return_200d": 0,
            "ma_50": 0,
            "ma_200": 0,
            "trend_strength": 0.5,
            "drawdown": 0,
            "momentum_score": 0,
            "price_vs_ma200": 0,
        }

    def _calculate_vol_percentile(self, current_vol: float) -> float:
        """Calculate volatility percentile from history"""
        if len(self.volatility_history) < 20:
            return 50.0

        sorted_vols = sorted(self.volatility_history)
        position = np.searchsorted(sorted_vols, current_vol)
        percentile = (position / len(sorted_vols)) * 100

        return percentile

    def get_regime_summary(self) -> Dict:
        """Get summary of recent regime history"""
        if not self.regime_history:
            return {}

        recent_regimes = list(self.regime_history)[-20:]
        regime_counts = {}

        for state in recent_regimes:
            regime = state.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            "current_regime": (
                self.last_market_state.regime.value
                if self.last_market_state
                else "unknown"
            ),
            "regime_distribution": regime_counts,
            "avg_volatility": (
                np.mean(list(self.volatility_history)) if self.volatility_history else 0
            ),
            "avg_breadth": (
                np.mean(list(self.breadth_history)) if self.breadth_history else 0.5
            ),
            "early_warning": self.early_warning_active,
        }
