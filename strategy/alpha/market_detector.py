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


class MarketRegime(Enum):
    """Fine-grained market regime classifications (intermediate states)."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NORMAL = "normal"
    BEAR = "bear"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    VOLATILE = "volatile"


class MacroRegime(Enum):
    """High-level strategic states for portfolio allocation."""

    OFFENSE = "offense"
    DEFENSE = "defense"
    SIDEWAYS = "sideways"


@dataclass
class MarketState:
    """Complete market state snapshot at a given time."""

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


class MacroEnhancer:
    """
    Analyzes macro indicators to enhance regime detection.
    Provides early warning signals and regime confidence adjustments.
    """

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
        Analyze macro indicators and return warning signals.

        Returns:
            Dict with keys:
            - recession_risk: [0, 1] probability
            - policy_tightening_risk: [0, 1] probability
            - crisis_warning: [0, 1] severity
            - macro_confidence_adj: [-0.2, +0.2] adjustment to regime confidence
        """
        if not self.enabled or macro_data is None or macro_data.empty:
            return self._default_signals()

        try:
            # Get latest macro values at or before timestamp
            macro_snapshot = macro_data.loc[:timestamp].iloc[-1]

            signals = {
                "recession_risk": 0.0,
                "policy_tightening_risk": 0.0,
                "crisis_warning": 0.0,
                "macro_confidence_adj": 0.0,
            }

            # === 1. Yield Curve Analysis (Recession Predictor) ===
            if "T10Y2Y" in macro_snapshot and not pd.isna(macro_snapshot["T10Y2Y"]):
                yield_spread = macro_snapshot["T10Y2Y"]
                inversion_threshold = self.thresholds.get(
                    "yield_curve_inversion", -0.20
                )

                if yield_spread < inversion_threshold:
                    # Severe inversion = high recession risk
                    signals["recession_risk"] = min(
                        1.0, abs(yield_spread) / abs(inversion_threshold)
                    )
                    self.logger.debug(
                        f"Yield curve inverted: {yield_spread:.2f}% "
                        f"(recession_risk={signals['recession_risk']:.2f})"
                    )

            # === 2. VIX Fear Gauge (Crisis Detection) ===
            if "VIXCLS" in macro_snapshot and not pd.isna(macro_snapshot["VIXCLS"]):
                vix_level = macro_snapshot["VIXCLS"]
                crisis_threshold = self.thresholds.get("vix_crisis_level", 35)

                if vix_level > crisis_threshold:
                    # Elevated VIX = crisis warning
                    signals["crisis_warning"] = min(
                        1.0, (vix_level - 20) / (crisis_threshold - 20)
                    )
                    self.logger.debug(
                        f"VIX elevated: {vix_level:.1f} "
                        f"(crisis_warning={signals['crisis_warning']:.2f})"
                    )

            # === 3. Labor Market Health (Economic Momentum) ===
            if "UNRATE" in macro_snapshot and not pd.isna(macro_snapshot["UNRATE"]):
                # Check if unemployment is rising (compare to 3-month ago)
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
                        # Rising unemployment = weakening economy
                        signals["recession_risk"] = max(
                            signals["recession_risk"],
                            min(1.0, unemployment_change / spike_threshold),
                        )
                        self.logger.debug(
                            f"Unemployment rising: +{unemployment_change:.2f}% "
                            f"(recession_risk={signals['recession_risk']:.2f})"
                        )

            # === 4. Inflation Pressure (Policy Tightening Risk) ===
            if "CPIAUCSL" in macro_snapshot and not pd.isna(macro_snapshot["CPIAUCSL"]):
                # Calculate YoY inflation if we have 12 months of data
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
                        # High inflation = potential Fed tightening
                        signals["policy_tightening_risk"] = min(
                            1.0, inflation_yoy / high_inflation_threshold
                        )
                        self.logger.debug(
                            f"Inflation elevated: {inflation_yoy:.2%} YoY "
                            f"(tightening_risk={signals['policy_tightening_risk']:.2f})"
                        )

            # === 5. Composite Macro Confidence Adjustment ===
            # Negative signals reduce confidence in bullish regimes
            negative_signal_score = (
                signals["recession_risk"] * 0.4
                + signals["crisis_warning"] * 0.4
                + signals["policy_tightening_risk"] * 0.2
            )

            # Adjustment range: -0.2 (very negative) to 0 (neutral)
            signals["macro_confidence_adj"] = -0.2 * negative_signal_score

            self.logger.debug(
                f"Macro signals: recession={signals['recession_risk']:.2f}, "
                f"crisis={signals['crisis_warning']:.2f}, "
                f"tightening={signals['policy_tightening_risk']:.2f}, "
                f"conf_adj={signals['macro_confidence_adj']:.2f}"
            )

            return signals

        except Exception as e:
            self.logger.warning(f"Macro analysis failed: {e}")
            return self._default_signals()

    def _default_signals(self) -> Dict[str, float]:
        """Return neutral signals when macro data unavailable."""
        return {
            "recession_risk": 0.0,
            "policy_tightening_risk": 0.0,
            "crisis_warning": 0.0,
            "macro_confidence_adj": 0.0,
        }


class MarketDetector:
    """
    Market regime detector with MS-GARCH volatility model and macro enhancement.

    NEW: Integrates MacroEnhancer for early warning signals
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Load parameters from config
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

        # Load sideways detection config
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

        # Initialize GARCH detector
        self.garch_detector = MsGarchDetector()

        # Initialize macro enhancer
        macro_config = {
            "enable_macro_enhancement": get_param("enable_macro_enhancement", False),
            "macro_indicators_subset": get_param("macro_indicators_subset", []),
            "macro_thresholds": get_param("macro_thresholds", {}),
        }
        self.macro_enhancer = MacroEnhancer(macro_config)

        # Regime smoothing
        self.macro_regime_history = deque(maxlen=3)
        # Caching
        self.state_cache_duration = get_param("cache_duration_seconds", 300)
        self.last_detection_time = None
        self.last_market_state = None
        self.garch_state_cache = {}
        self.garch_cache_days = 15

        self.logger.info(
            "MarketDetector initialized with MS-GARCH and Macro Enhancement"
        )

    def detect_market_state(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        symbols: List[str] = None,
        macro_data: Optional[pd.DataFrame] = None,  # NEW parameter
    ) -> MarketState:
        """
        Detect current market regime with macro enhancement.

        NEW: Accepts optional macro_data for enhanced detection
        """
        # Check cache
        if self.last_detection_time is not None and self.last_market_state is not None:
            time_diff = (timestamp - self.last_detection_time).total_seconds()
            if time_diff < self.state_cache_duration:
                return self.last_market_state

            # --- Step 1: Analyze market index (fast) and market breadth (now vectorized and fast) ---
        index_analysis = self._analyze_market_index(market_data, timestamp)
        market_breadth = self._calculate_market_breadth(market_data, timestamp, symbols)

        # --- Step 2: Get GARCH volatility state using the new intelligent cache ---
        garch_cache_key = timestamp.normalize().floor(f"{self.garch_cache_days}D")

        if garch_cache_key in self.garch_state_cache:
            garch_vol_state = self.garch_state_cache[garch_cache_key]
            self.logger.debug(
                f"Using cached GARCH state for {timestamp.date()}: {garch_vol_state}"
            )
        else:
            self.logger.info(
                f"Recalculating GARCH state for cache key: {garch_cache_key.date()}"
            )
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

        # If GARCH fails, we need a valid spy_returns series for sideways detection
        try:
            spy_returns = (
                market_data.xs("SPY", level="symbol")
                .loc[:timestamp]["close"]
                .pct_change()
            )
        except Exception:
            spy_returns = pd.Series()

        # --- The rest of the logic remains the same ---
        is_sideways = self._detect_sideways_market(spy_returns.dropna().tolist())
        macro_signals = self.macro_enhancer.analyze_macro_conditions(
            macro_data, timestamp
        )
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
        )

        self.last_detection_time = timestamp
        self.last_market_state = market_state
        return market_state

    def _detect_sideways_market(self, returns: List[float]) -> bool:
        """
        Detects a choppy, trendless market based on recent returns.
        This is a market condition analysis, so it belongs in the MarketDetector.
        """
        if not self.sideways_detection_enabled or len(returns) < self.sideways_lookback:
            return False

        recent_returns = np.array(returns[-self.sideways_lookback :])

        # Low cumulative return and moderate volatility are signs of a sideways market
        cumulative_return = np.sum(recent_returns)
        volatility = np.std(recent_returns)

        # Check against thresholds loaded in __init__
        is_low_trend = abs(cumulative_return) < self.sideways_low_trend_threshold
        is_moderate_vol = self.sideways_vol_min < volatility < self.sideways_vol_max

        if is_low_trend and is_moderate_vol:
            self.logger.info(
                f"Sideways market detected: cum_ret={cumulative_return:.2%}, vol={volatility:.3f}"
            )
            return True

        return False

    def _map_to_macro_regime(
        self, regime: MarketRegime, garch_state: str, macro_signals: Dict[str, float]
    ) -> MacroRegime:
        """
        Map fine-grained regime to macro regime with macro enhancement.

        NEW: Uses macro signals for additional validation
        """
        # DEFENSE: Crisis, Bear, or macro warnings
        if regime in [MarketRegime.CRISIS, MarketRegime.BEAR]:
            return MacroRegime.DEFENSE

        # NEW: Macro warning override - even if price looks ok, macro says be cautious
        if (
            macro_signals["crisis_warning"] > 0.6
            or macro_signals["recession_risk"] > 0.7
        ):
            self.logger.info(
                f"Macro override to DEFENSE: crisis_warning={macro_signals['crisis_warning']:.2f}, "
                f"recession_risk={macro_signals['recession_risk']:.2f}"
            )
            return MacroRegime.DEFENSE

        # OFFENSE: Strong bull + low vol + no macro warnings
        if (
            regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]
            and garch_state == "low_vol"
            and macro_signals["crisis_warning"] < 0.3
            and macro_signals["recession_risk"] < 0.4
        ):
            return MacroRegime.OFFENSE

        # SIDEWAYS: Everything else (uncertain conditions)
        return MacroRegime.SIDEWAYS

    def _determine_market_regime(
        self,
        index_analysis: Dict,
        market_breadth: float,
        garch_vol_state: str,
        macro_signals: Dict[str, float],
    ) -> Tuple[MarketRegime, float]:
        """
        Determine fine-grained regime with macro-adjusted confidence.

        NEW: Confidence is adjusted by macro signals
        """
        ret_50d = index_analysis["return_50d"]
        drawdown = index_analysis["drawdown"]

        # Base confidence before macro adjustment
        base_confidence = 0.5

        # === Crisis Detection ===
        if garch_vol_state == "high_vol" and drawdown < self.crisis_threshold:
            base_confidence = 0.9
            regime = MarketRegime.CRISIS

        # === Bear Market ===
        elif (
            ret_50d < self.bear_threshold
            and market_breadth < self.breadth_bear_threshold
        ) or (index_analysis["trend_strength"] < 0.3 and garch_vol_state == "high_vol"):
            base_confidence = 0.8
            regime = MarketRegime.BEAR

        # === Strong Bull ===
        elif (
            ret_50d > self.strong_bull_threshold
            and market_breadth > self.breadth_bull_threshold
            and garch_vol_state == "low_vol"
        ):
            base_confidence = 0.85
            regime = MarketRegime.STRONG_BULL

        # === Bull ===
        elif (
            ret_50d > self.bull_threshold
            and market_breadth > self.breadth_bull_threshold * 0.9
            and garch_vol_state == "low_vol"
        ):
            base_confidence = 0.75
            regime = MarketRegime.BULL

        # === Normal / Volatile ===
        else:
            if garch_vol_state == "high_vol":
                if abs(ret_50d) > 0.05 or index_analysis["trend_strength"] > 0.5:
                    base_confidence = 0.6
                    regime = MarketRegime.NORMAL
                else:
                    base_confidence = 0.7
                    regime = MarketRegime.VOLATILE
            else:
                base_confidence = 0.6
                regime = MarketRegime.NORMAL

        # NEW: Apply macro confidence adjustment
        adjusted_confidence = base_confidence + macro_signals["macro_confidence_adj"]
        adjusted_confidence = max(
            0.3, min(0.95, adjusted_confidence)
        )  # Clamp to [0.3, 0.95]

        if abs(macro_signals["macro_confidence_adj"]) > 0.05:
            self.logger.debug(
                f"Regime {regime.value}: confidence {base_confidence:.2f} → "
                f"{adjusted_confidence:.2f} (macro_adj={macro_signals['macro_confidence_adj']:.2f})"
            )

        return regime, adjusted_confidence

    def _assess_risk_level(self, regime: MarketRegime) -> str:
        """Assess risk level from regime."""
        if regime in [MarketRegime.CRISIS, MarketRegime.BEAR]:
            return "high"
        if regime == MarketRegime.VOLATILE:
            return "medium"
        if regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]:
            return "low"
        return "medium"

    # === Price-based Analysis Methods (Unchanged) ===

    def _analyze_market_index(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """Analyze SPY for trend detection."""
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

                ret_20d = (
                    ((current_price / prices[-self.lookback_short]) - 1)
                    if len(prices) >= self.lookback_short
                    else 0
                )
                ret_50d = (
                    ((current_price / prices[-self.lookback_medium]) - 1)
                    if len(prices) >= self.lookback_medium
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
                    trend_strength += 0.5
                if current_price > ma_50:
                    trend_strength += 0.3
                if ma_50 > ma_200:
                    trend_strength += 0.2

                recent_high = (
                    np.max(prices[-self.lookback_medium :])
                    if len(prices) >= self.lookback_medium
                    else current_price
                )
                drawdown = (current_price - recent_high) / recent_high

                return {
                    "return_20d": ret_20d,
                    "return_50d": ret_50d,
                    "ma_50": ma_50,
                    "ma_200": ma_200,
                    "trend_strength": trend_strength,
                    "drawdown": drawdown,
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
        if symbols is None:

            symbols = RISK_ON_SYMBOLS

        try:
            # 1. Define the date range needed for the calculation
            start_date = timestamp - pd.Timedelta(
                days=self.lookback_short * 2
            )  # Get a wider slice for safety

            # 2. Slice the main market_data DataFrame once for efficiency
            sliced_data = market_data[
                (market_data.index.get_level_values("timestamp") >= start_date)
                & (market_data.index.get_level_values("timestamp") <= timestamp)
            ]

            if sliced_data.empty:
                return 0.5  # Return neutral breadth if no data is available

            # 3. Unstack to get prices with symbols as columns
            prices = sliced_data["close"].unstack(level="symbol")
            if prices.empty or timestamp not in prices.index:
                return 0.5
            # 4. Find which of the requested symbols are *actually available* in the historical data
            available_symbols = prices.columns.intersection(symbols)

            # If there are too few symbols with data, the breadth measure is meaningless
            if len(available_symbols) < 20:  # Use a threshold like 20 symbols
                self.logger.debug(
                    f"Too few symbols ({len(available_symbols)}) for breadth calculation at {timestamp.date()}."
                )
                return 0.5  # Fallback to neutral
            # 5. Calculate returns ONLY for the symbols that were available at that time
            relevant_prices = prices[available_symbols]
            returns = (relevant_prices / relevant_prices.shift(self.lookback_short)) - 1

            # 6. Get the returns for the current timestamp for the valid symbols
            returns_today = returns.loc[timestamp].dropna()

            if returns_today.empty:
                return 0.5

            # 7. Count advancing and declining stocks
            advancing = (returns_today > 0.02).sum()
            declining = (returns_today < -0.02).sum()

            total = advancing + declining
            return advancing / total if total > 0 else 0.5

        except Exception as e:
            # This will now catch other, unexpected errors, making debugging easier.
            self.logger.warning(
                f"Market breadth calculation failed unexpectedly for {timestamp.date()}: {e}"
            )
            return 0.5

    def _default_index_analysis(self) -> Dict[str, float]:
        """Default values when data insufficient."""
        return {
            "return_20d": 0,
            "return_50d": 0,
            "ma_50": 0,
            "ma_200": 0,
            "trend_strength": 0.5,
            "drawdown": 0,
        }
