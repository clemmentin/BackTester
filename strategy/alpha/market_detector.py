import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Dict, List, Tuple, Optional

import config
import numpy as np
import pandas as pd

from .ms_garch_detector import MsGarchDetector
from config.general_config import RISK_ON_SYMBOLS
import faiss


class MarketRegime(Enum):
    """ """

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NORMAL = "normal"
    BEAR = "bear"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    VOLATILE = "volatile"


class MacroRegime(Enum):
    """ """

    OFFENSE = "offense"
    DEFENSE = "defense"
    SIDEWAYS = "sideways"


@dataclass
class MarketState:
    """ """

    regime: MarketRegime
    macro_regime: MacroRegime
    confidence: float
    indicators: Dict[str, float]
    environment_risk_level: str
    trend_strength: float
    volatility_regime: str
    breadth: float
    macro_signals: Dict[str, float]
    is_sideways: bool
    timestamp: pd.Timestamp
    # MODIFIED: Replaced similarity_risk_adjustment with a new dual-factor system.
    transition_adjustment: float = 1.0
    deviation_vector: Optional[np.ndarray] = None


class MacroEnhancer:
    """ """

    def __init__(self, config_params: Dict):
        self.logger = logging.getLogger(__name__)
        self.enabled = config_params.get("enable_macro_enhancement", False)
        self.indicators_subset = config_params.get("macro_indicators_subset", [])
        self.thresholds = config_params.get("macro_thresholds", {})
        self.logger.info(
            f"MacroEnhancer initialized: enabled={self.enabled}, "
            f"indicators={len(self.indicators_subset)}"
        )

    def analyze_macro_conditions(
        self, macro_data: Optional[pd.DataFrame], timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """

        Args:
          macro_data: Optional[pd.DataFrame]:
          timestamp: pd.Timestamp:

        Returns:

        """
        if not self.enabled or macro_data is None or macro_data.empty:
            return self._default_signals()
        try:
            macro_snapshot = macro_data.loc[:timestamp].iloc[-1]
            signals = {
                "recession_risk": 0.0,
                "policy_tightening_risk": 0.0,
                "crisis_warning": 0.0,
                "macro_confidence_adj": 0.0,
            }
            if "T10Y2Y" in macro_snapshot and not pd.isna(macro_snapshot["T10Y2Y"]):
                yield_spread = macro_snapshot["T10Y2Y"]
                inversion_threshold = self.thresholds.get(
                    "yield_curve_inversion", -0.20
                )
                if yield_spread < inversion_threshold:
                    signals["recession_risk"] = min(
                        1.0, abs(yield_spread) / abs(inversion_threshold)
                    )
            if "VIXCLS" in macro_snapshot and not pd.isna(macro_snapshot["VIXCLS"]):
                vix_level = macro_snapshot["VIXCLS"]
                crisis_threshold = self.thresholds.get("vix_crisis_level", 35)
                if vix_level > crisis_threshold:
                    signals["crisis_warning"] = min(
                        1.0, (vix_level - 20) / (crisis_threshold - 20)
                    )
            if "UNRATE" in macro_snapshot and not pd.isna(macro_snapshot["UNRATE"]):
                lookback_data = macro_data.loc[:timestamp].tail(90)
                if len(lookback_data) >= 60:
                    current_unrate = macro_snapshot["UNRATE"]
                    past_unrate = (
                        lookback_data.iloc[-60]["UNRATE"]
                        if "UNRATE" in lookback_data.columns
                        else current_unrate
                    )
                    unemployment_change = current_unrate - past_unrate
                    spike_threshold = self.thresholds.get("unemployment_spike", 0.50)
                    if unemployment_change > spike_threshold:
                        signals["recession_risk"] = max(
                            signals["recession_risk"],
                            min(1.0, unemployment_change / spike_threshold),
                        )
            if "CPIAUCSL" in macro_snapshot and not pd.isna(macro_snapshot["CPIAUCSL"]):
                lookback_data = macro_data.loc[:timestamp].tail(365)
                if len(lookback_data) >= 252:
                    current_cpi = macro_snapshot["CPIAUCSL"]
                    past_cpi = (
                        lookback_data.iloc[-252]["CPIAUCSL"]
                        if "CPIAUCSL" in lookback_data.columns
                        else current_cpi
                    )
                    inflation_yoy = (current_cpi / past_cpi - 1) if past_cpi > 0 else 0
                    high_inflation_threshold = self.thresholds.get(
                        "inflation_high", 0.05
                    )
                    if inflation_yoy > high_inflation_threshold:
                        signals["policy_tightening_risk"] = min(
                            1.0, inflation_yoy / high_inflation_threshold
                        )
            negative_signal_score = (
                signals["recession_risk"] * 0.4
                + signals["crisis_warning"] * 0.4
                + signals["policy_tightening_risk"] * 0.2
            )
            signals["macro_confidence_adj"] = -0.2 * negative_signal_score
            return signals
        except Exception as e:
            self.logger.warning(f"Macro analysis failed: {e}")
            return self._default_signals()

    def _default_signals(self) -> Dict[str, float]:
        """ """
        return {
            "recession_risk": 0.0,
            "policy_tightening_risk": 0.0,
            "crisis_warning": 0.0,
            "macro_confidence_adj": 0.0,
        }


class MarketDetector:
    """ """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        get_param = partial(config.get_param_with_override, kwargs, "market_detector")
        self.lookback_short = get_param("lookback_short", 20)
        self.lookback_medium = get_param("lookback_medium", 50)
        self.lookback_long = get_param("lookback_long", 200)
        self.bull_threshold = get_param("bull_threshold", 0.03)
        self.strong_bull_threshold = get_param("strong_bull_threshold", 0.10)
        self.bear_threshold = get_param("bear_threshold", -0.05)
        self.crisis_threshold = get_param("crisis_threshold", -0.15)
        self.breadth_bull_threshold = get_param("breadth_bull_threshold", 0.65)
        self.breadth_bear_threshold = get_param("breadth_bear_threshold", 0.35)

        # --- NEW: Prototype Analysis Setup (replaces Similarity Analysis) ---
        proto_config = kwargs.get("prototype_analysis", {})
        self.prototype_enabled = proto_config.get("enabled", True)
        self.prototype_lookback = proto_config.get("lookback", 252)
        self.min_history_for_prototype = proto_config.get("min_history", 100)
        self.transition_lambda = proto_config.get(
            "transition_lambda", 1.5
        )  # Controls sensitivity of the adjustment factor

        # History for prototype analysis
        self.prototype_vector_history = deque(maxlen=self.prototype_lookback)

        sideways_config = config.get_trading_param(
            "RISK_PARAMS", "sideways_detection", default={}
        )
        self.sideways_detection_enabled = sideways_config.get("enabled", True)
        self.sideways_lookback = sideways_config.get("lookback", 30)
        self.sideways_low_trend_threshold = sideways_config.get(
            "low_trend_threshold", 0.05
        )
        self.sideways_vol_min = sideways_config.get("vol_min", 0.008)
        self.sideways_vol_max = sideways_config.get("vol_max", 0.025)

        self.garch_detector = MsGarchDetector(**kwargs)

        macro_config = kwargs.get("market_detector_params", {}).get(
            "macro_enhancement", {}
        )
        self.macro_enhancer = MacroEnhancer(macro_config)

        md_const = config.get_trading_param("MARKET_DETECTOR_CONSTANTS", default={})
        self.regime_confidence = md_const.get("regime_confidence", {})
        self.trend_analysis_weights = md_const.get("trend_analysis", {})

        self.macro_regime_history = deque(maxlen=3)
        self.state_cache_duration = get_param("cache_duration_seconds", 300)
        self.last_detection_time = None
        self.last_market_state = None
        self.garch_state_cache = {}
        self.garch_cache_days = 15

        self.logger.info(
            "MarketDetector initialized with MS-GARCH and Macro Enhancement"
        )

    def _vectorize_state(
        self,
        index_analysis: Dict,
        market_breadth: float,
        garch_vol_state: str,
        macro_signals: Dict,
    ) -> np.ndarray:
        """MODIFIED: Encodes market state into a de-trended, standardized numerical vector.
        This captures relative market conditions rather than absolute levels.

        Args:
          index_analysis: Dict:
          market_breadth: float:
          garch_vol_state: str:
          macro_signals: Dict:

        Returns:

        """
        vector = [
            # De-trended momentum (medium-term vs long-term)
            index_analysis.get("return_50d", 0) - index_analysis.get("return_200d", 0),
            # Price deviation from long-term trend
            index_analysis.get("ma_200_deviation", 0),
            # Other relative indicators
            index_analysis.get("trend_strength", 0.5),
            index_analysis.get("drawdown", 0),
            market_breadth,
            1.0 if garch_vol_state == "high_vol" else 0.0,
            macro_signals.get("recession_risk", 0.0),
            macro_signals.get("crisis_warning", 0.0),
        ]
        vec = np.array(vector, dtype=np.float32)
        # Standardize the vector to have zero mean and unit variance for consistent distance metrics.
        if vec.std() > 1e-6:
            vec = (vec - vec.mean()) / vec.std()
        return vec

    def _update_and_calculate_prototype_factors(
        self, current_vector: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """NEW: Maintains a history of state vectors to compute a dynamic "prototype"
        of the current market regime and calculates deviation from it.

        Args:
          current_vector: np.ndarray:

        Returns:

        """
        self.prototype_vector_history.append(current_vector)

        # Check if we have enough data to perform the analysis
        if len(self.prototype_vector_history) < self.min_history_for_prototype:
            # Not enough history, return neutral adjustment
            return 1.0, np.zeros_like(current_vector)

        # --- Calculate the sliding prototype and deviation ---
        historical_vectors = np.array(self.prototype_vector_history, dtype=np.float32)
        prototype_vector = historical_vectors.mean(axis=0)

        # Calculate deviation vector (micro-level alpha adjustment)
        deviation_vector = current_vector - prototype_vector

        # Calculate Euclidean distance to the prototype
        distance = np.linalg.norm(deviation_vector)

        # Calculate transition adjustment factor (macro-level risk scaling)
        # This factor approaches 1 when the market is "normal" (close to prototype)
        # and approaches 0 as the market deviates significantly.
        transition_adjustment = np.exp(-self.transition_lambda * distance)

        return transition_adjustment, deviation_vector

    def detect_market_state(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        symbols: List[str] = None,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> MarketState:
        """

        Args:
          market_data: pd.DataFrame:
          timestamp: pd.Timestamp:
          symbols: List[str]:  (Default value = None)
          macro_data: Optional[pd.DataFrame]:  (Default value = None)

        Returns:

        """
        # Cache check
        if self.last_detection_time is not None and self.last_market_state is not None:
            time_diff = (timestamp - self.last_detection_time).total_seconds()
            if time_diff < self.state_cache_duration:
                return self.last_market_state

        index_analysis = self._analyze_market_index(market_data, timestamp)
        market_breadth = self._calculate_market_breadth(market_data, timestamp, symbols)

        garch_cache_key = timestamp.normalize().floor(f"{self.garch_cache_days}D")
        if garch_cache_key in self.garch_state_cache:
            garch_vol_state = self.garch_state_cache[garch_cache_key]
        else:
            try:
                spy_data = market_data.xs("SPY", level="symbol").loc[:timestamp]
                spy_returns = spy_data["close"].pct_change()
                garch_vol_state = self.garch_detector.get_volatility_state(
                    spy_returns, timestamp
                )
                self.garch_state_cache[garch_cache_key] = garch_vol_state
            except Exception as e:
                self.logger.warning(f"GARCH detection failed: {e}")
                garch_vol_state = "unknown"

        is_sideways = self._detect_sideways_market(market_data, timestamp)
        macro_signals = self.macro_enhancer.analyze_macro_conditions(
            macro_data, timestamp
        )

        # --- NEW: Prototype Analysis Integration (replaces similarity analysis) ---
        if self.prototype_enabled:
            current_vector = self._vectorize_state(
                index_analysis, market_breadth, garch_vol_state, macro_signals
            )
            transition_adj, deviation_vec = (
                self._update_and_calculate_prototype_factors(current_vector)
            )
        else:
            transition_adj, deviation_vec = 1.0, None

        regime, confidence = self._determine_market_regime(
            index_analysis, market_breadth, garch_vol_state, macro_signals
        )
        macro_regime = self._map_to_macro_regime(regime, garch_vol_state, macro_signals)

        self.macro_regime_history.append(macro_regime)
        if (
            len(self.macro_regime_history) > 1
            and self.macro_regime_history[-1] != self.macro_regime_history[-2]
        ):
            confirmed_macro_regime = self.macro_regime_history[-2]
        else:
            confirmed_macro_regime = macro_regime

        market_state = MarketState(
            regime=regime,
            macro_regime=confirmed_macro_regime,
            confidence=confidence,
            indicators={
                "index_return_50d": index_analysis.get("return_50d", 0),
                "trend_strength": index_analysis.get("trend_strength", 0),
                "market_breadth": market_breadth,
                "garch_state": 1 if garch_vol_state == "high_vol" else 0,
            },
            environment_risk_level=self._assess_risk_level(regime),
            trend_strength=index_analysis.get("trend_strength", 0),
            volatility_regime=garch_vol_state,
            breadth=market_breadth,
            macro_signals=macro_signals,
            is_sideways=is_sideways,
            timestamp=timestamp,
            transition_adjustment=transition_adj,
            deviation_vector=deviation_vec,
        )

        self.last_detection_time = timestamp
        self.last_market_state = market_state
        return market_state

    def _detect_sideways_market(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> bool:
        """

        Args:
          market_data: pd.DataFrame:
          timestamp: pd.Timestamp:

        Returns:

        """
        if not self.sideways_detection_enabled:
            return False
        try:
            spy_data = market_data.xs("SPY", level="symbol").loc[:timestamp]
            if len(spy_data) < self.sideways_lookback:
                return False

            returns = spy_data["close"].pct_change().dropna()
            if len(returns) < self.sideways_lookback:
                return False

            recent_returns = np.array(returns[-self.sideways_lookback :])
            cumulative_return = np.sum(recent_returns)
            volatility = np.std(recent_returns)
            is_low_trend = abs(cumulative_return) < self.sideways_low_trend_threshold
            is_moderate_vol = self.sideways_vol_min < volatility < self.sideways_vol_max
            if is_low_trend and is_moderate_vol:
                self.logger.info(
                    f"Sideways market detected: cum_ret={cumulative_return:.2%}, vol={volatility:.3f}"
                )
                return True
        except Exception as e:
            self.logger.warning(f"Sideways detection failed: {e}")

        return False

    def _map_to_macro_regime(
        self, regime: MarketRegime, garch_state: str, macro_signals: Dict[str, float]
    ) -> MacroRegime:
        """

        Args:
          regime: MarketRegime:
          garch_state: str:
          macro_signals: Dict[str:
          float]:

        Returns:

        """
        if regime in [MarketRegime.CRISIS, MarketRegime.BEAR]:
            return MacroRegime.DEFENSE
        if (
            macro_signals["crisis_warning"] > 0.6
            or macro_signals["recession_risk"] > 0.7
        ):
            return MacroRegime.DEFENSE
        if (
            regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]
            and garch_state == "low_vol"
            and macro_signals["crisis_warning"] < 0.3
            and macro_signals["recession_risk"] < 0.4
        ):
            return MacroRegime.OFFENSE
        return MacroRegime.SIDEWAYS

    def _determine_market_regime(
        self,
        index_analysis: Dict,
        market_breadth: float,
        garch_vol_state: str,
        macro_signals: Dict[str, float],
    ) -> Tuple[MarketRegime, float]:
        """

        Args:
          index_analysis: Dict:
          market_breadth: float:
          garch_vol_state: str:
          macro_signals: Dict[str:
          float]:

        Returns:

        """
        ret_50d = index_analysis["return_50d"]
        drawdown = index_analysis["drawdown"]

        base_confidence = self.regime_confidence.get("normal_base", 0.6)

        if garch_vol_state == "high_vol" and drawdown < self.crisis_threshold:
            base_confidence = self.regime_confidence.get("crisis_base", 0.9)
            regime = MarketRegime.CRISIS
        elif (
            ret_50d < self.bear_threshold
            and market_breadth < self.breadth_bear_threshold
        ) or (index_analysis["trend_strength"] < 0.3 and garch_vol_state == "high_vol"):
            base_confidence = self.regime_confidence.get("bear_base", 0.8)
            regime = MarketRegime.BEAR
        elif (
            ret_50d > self.strong_bull_threshold
            and market_breadth > self.breadth_bull_threshold
            and garch_vol_state == "low_vol"
        ):
            base_confidence = self.regime_confidence.get("strong_bull_base", 0.85)
            regime = MarketRegime.STRONG_BULL
        elif (
            ret_50d > self.bull_threshold
            and market_breadth > self.breadth_bull_threshold * 0.9
            and garch_vol_state == "low_vol"
        ):
            base_confidence = self.regime_confidence.get("bull_base", 0.75)
            regime = MarketRegime.BULL
        else:
            if garch_vol_state == "high_vol":
                if abs(ret_50d) > 0.05 or index_analysis["trend_strength"] > 0.5:
                    base_confidence = self.regime_confidence.get("normal_base", 0.6)
                    regime = MarketRegime.NORMAL
                else:
                    base_confidence = self.regime_confidence.get("volatile_base", 0.7)
                    regime = MarketRegime.VOLATILE
            else:
                base_confidence = self.regime_confidence.get("normal_base", 0.6)
                regime = MarketRegime.NORMAL

        adjusted_confidence = base_confidence + macro_signals["macro_confidence_adj"]
        clip_min = self.regime_confidence.get("confidence_clip_min", 0.3)
        clip_max = self.regime_confidence.get("confidence_clip_max", 0.95)
        adjusted_confidence = max(clip_min, min(clip_max, adjusted_confidence))

        return regime, adjusted_confidence

    def _assess_risk_level(self, regime: MarketRegime) -> str:
        """

        Args:
          regime: MarketRegime:

        Returns:

        """
        if regime in [MarketRegime.CRISIS, MarketRegime.BEAR]:
            return "high"
        if regime == MarketRegime.VOLATILE:
            return "medium"
        if regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]:
            return "low"
        return "medium"

    def _analyze_market_index(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """

        Args:
          market_data: pd.DataFrame:
          timestamp: pd.Timestamp:

        Returns:

        """
        index_symbol = "SPY"
        try:
            if index_symbol in market_data.index.get_level_values("symbol"):
                index_data = market_data.xs(index_symbol, level="symbol").loc[
                    :timestamp
                ]
                if len(index_data) < self.lookback_long:
                    return self._default_index_analysis()

                prices = index_data["close"].values
                current_price = prices[-1]

                ret_50d = (
                    ((current_price / prices[-self.lookback_medium]) - 1)
                    if len(prices) >= self.lookback_medium
                    else 0
                )
                ret_200d = (
                    ((current_price / prices[-self.lookback_long]) - 1)
                    if len(prices) >= self.lookback_long
                    else 0
                )
                ma_50 = (
                    np.mean(prices[-self.lookback_medium :])
                    if len(prices) >= self.lookback_medium
                    else current_price
                )
                ma_200 = np.mean(prices[-self.lookback_long :])

                trend_strength = 0.0
                if current_price > ma_200:
                    trend_strength += self.trend_analysis_weights.get(
                        "above_ma200_weight", 0.5
                    )
                if current_price > ma_50:
                    trend_strength += self.trend_analysis_weights.get(
                        "above_ma50_weight", 0.3
                    )
                if ma_50 > ma_200:
                    trend_strength += self.trend_analysis_weights.get(
                        "ma50_above_ma200_weight", 0.2
                    )

                recent_high = (
                    np.max(prices[-self.lookback_medium :])
                    if len(prices) >= self.lookback_medium
                    else current_price
                )
                drawdown = (current_price - recent_high) / recent_high
                ma_200_deviation = (
                    (current_price - ma_200) / ma_200 if ma_200 > 0 else 0
                )

                return {
                    "return_20d": (
                        ((current_price / prices[-self.lookback_short]) - 1)
                        if len(prices) >= self.lookback_short
                        else 0
                    ),
                    "return_50d": ret_50d,
                    "return_200d": ret_200d,
                    "trend_strength": trend_strength,
                    "drawdown": drawdown,
                    "ma_200_deviation": ma_200_deviation,
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
        """

        Args:
          market_data: pd.DataFrame:
          timestamp: pd.Timestamp:
          symbols: List[str]:  (Default value = None)

        Returns:

        """
        if symbols is None:
            symbols = RISK_ON_SYMBOLS
        try:
            start_date = timestamp - pd.Timedelta(days=self.lookback_short * 2)
            sliced_data = market_data[
                (market_data.index.get_level_values("timestamp") >= start_date)
                & (market_data.index.get_level_values("timestamp") <= timestamp)
            ]
            if sliced_data.empty:
                return 0.5

            prices = sliced_data["close"].unstack(level="symbol")
            if prices.empty or timestamp not in prices.index:
                return 0.5

            available_symbols = prices.columns.intersection(symbols)
            if len(available_symbols) < 20:
                return 0.5

            relevant_prices = prices[available_symbols]
            returns = (relevant_prices / relevant_prices.shift(self.lookback_short)) - 1
            returns_today = returns.loc[timestamp].dropna()

            if returns_today.empty:
                return 0.5

            advancing = (returns_today > 0.02).sum()
            declining = (returns_today < -0.02).sum()
            total = advancing + declining
            return advancing / total if total > 0 else 0.5
        except Exception as e:
            self.logger.warning(f"Market breadth calculation failed: {e}")
            return 0.5

    def _default_index_analysis(self) -> Dict[str, float]:
        """ """
        return {
            "return_20d": 0,
            "return_50d": 0,
            "return_200d": 0,
            "trend_strength": 0.5,
            "drawdown": 0,
            "ma_200_deviation": 0,
        }
