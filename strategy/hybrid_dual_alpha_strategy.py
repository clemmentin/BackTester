import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import config
import config.trading_parameters as tp
from backtester.events import SignalEvent
from config import general_config as gc

from strategy.alpha import (AlphaEngine, CompositeAlphaSignal, MarketDetector,
                            MarketRegime)
from strategy.base_strategy import BaseStrategy
from strategy.risk import PositionTracker, RiskManager, RiskMode, StopManager


@dataclass
class StrategyAllocation:
    """Strategy capital allocation state"""

    conservative_ratio: float
    aggressive_ratio: float
    cash_ratio: float
    timestamp: datetime
    reason: str


class EnhancedConservativeModule:
    def __init__(self, alpha_engine: AlphaEngine, **kwargs):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.alpha_engine = alpha_engine

        # --- Core conservative parameters (from config) ---
        self.momentum_windows = kwargs.get(
            "conservative_momentum_windows", [20, 60, 120]
        )
        self.max_positions = kwargs.get("conservative_max_positions", 3)
        self.position_weights = kwargs.get(
            "conservative_position_weights", [0.45, 0.35, 0.20]
        )
        self.momentum_weights = kwargs.get(
            "conservative_momentum_weights", [0.2, 0.3, 0.5]
        )

        # --- Alpha enhancement parameters (from config) ---
        self.alpha_enhancement_weight = kwargs.get("conservative_alpha_weight", 0.25)
        self.momentum_weight = 1 - self.alpha_enhancement_weight

        # --- Filter criteria (from config) ---
        self.min_alpha_confidence = kwargs.get("conservative_min_confidence", 0.4)
        self.require_positive_medium_term = kwargs.get("require_positive_medium", True)

        # --- Asset preferences (from general config) ---
        self.preferred_symbols = set(
            kwargs.get("risk_off_symbols", getattr(gc, "RISK_OFF_SYMBOLS", []))
        )
        self.risk_on_symbols = set(
            kwargs.get("risk_on_symbols", getattr(gc, "RISK_ON_SYMBOLS", []))
        )

        # --- Early warning system (from config) ---
        self.warning_activation_threshold = kwargs.get(
            "conservative_warning_threshold", 2
        )
        self.warning_position_multiplier = kwargs.get(
            "conservative_warning_multiplier", 0.5
        )
        self.warning_cooldown_days = kwargs.get("conservative_warning_cooldown_days", 5)

        # --- Risk adjustments (from config) ---
        self.risk_regime_multipliers = kwargs.get(
            "conservative_risk_multipliers",
            {"low": 1.1, "medium": 1.0, "high": 0.7, "extreme": 0.4},
        )
        self.asset_preference_multipliers = kwargs.get(
            "conservative_asset_preference_multipliers",
            {
                "risk_off_normal": 1.1,
                "risk_off_crisis": 1.5,
                "risk_on_normal": 0.9,
                "risk_on_crisis": 0.6,
            },
        )

        # --- Internal state ---
        self.early_warning_active = False
        self.last_warning_date = None

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_state,
        alpha_signals: Dict[str, CompositeAlphaSignal],
    ) -> Dict[str, float]:
        """Generate conservative strategy signals with Alpha enhancement"""

        # Step 1: Traditional momentum screening (unchanged core logic)
        momentum_scores = self._calculate_momentum_scores(market_data, timestamp)
        if not momentum_scores:
            return {}

        # Step 2: Alpha enhancement filtering
        alpha_enhanced_scores = self._apply_alpha_enhancement(
            momentum_scores, alpha_signals, market_state
        )

        # Step 3: Early warning system check
        warning_adjustment = self._check_early_warning(market_state, timestamp)

        # Step 4: Asset preference weighting (favor risk-off in uncertain times)
        preference_adjusted_scores = self._apply_asset_preferences(
            alpha_enhanced_scores, market_state, warning_adjustment
        )

        # Step 5: Select top positions with original weight allocation
        final_positions = self._select_final_positions(preference_adjusted_scores)

        if final_positions:
            self.logger.info(
                f"Conservative module selected {len(final_positions)} positions"
            )

        return final_positions

    def _calculate_momentum_scores(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate traditional momentum scores (original logic preserved)"""
        scores = {}

        # Get all available symbols
        available_symbols = sorted(
            market_data.index.get_level_values("symbol").unique()
        )

        for symbol in available_symbols:
            try:
                symbol_data = market_data.xs(symbol, level="symbol")
                symbol_data = symbol_data[symbol_data.index <= timestamp]

                if len(symbol_data) < max(self.momentum_windows):
                    continue

                prices = symbol_data["close"].values

                # Calculate momentum for each timeframe
                momentum_values = []
                for window in self.momentum_windows:
                    if len(prices) >= window:
                        momentum = (prices[-1] / prices[-window]) - 1
                        momentum_values.append(momentum)

                if len(momentum_values) == len(self.momentum_windows):
                    # Weighted momentum score (favor longer timeframes for conservative)
                    composite_momentum = sum(
                        m * w for m, w in zip(momentum_values, self.momentum_weights)
                    )
                    scores[symbol] = composite_momentum

            except Exception as e:
                self.logger.debug(f"Error calculating momentum for {symbol}: {e}")

        return scores

    def _apply_alpha_enhancement(
        self,
        momentum_scores: Dict[str, float],
        alpha_signals: Dict[str, CompositeAlphaSignal],
        market_state,
    ) -> Dict[str, float]:
        """Apply Alpha signal enhancement to momentum scores"""
        enhanced_scores = {}

        for symbol, momentum_score in momentum_scores.items():
            alpha_signal = alpha_signals.get(symbol)

            if (
                alpha_signal
                and alpha_signal.final_confidence >= self.min_alpha_confidence
            ):
                # Blend momentum and alpha scores
                alpha_component = (
                    alpha_signal.final_score * self.alpha_enhancement_weight
                )
                momentum_component = momentum_score * self.momentum_weight

                # Additional alpha confidence boost for high-quality signals
                confidence_boost = 1.0 + (alpha_signal.final_confidence - 0.5) * 0.2

                enhanced_score = (
                    momentum_component + alpha_component
                ) * confidence_boost
                enhanced_scores[symbol] = enhanced_score
            else:
                # Use pure momentum if no alpha signal or low confidence
                enhanced_scores[symbol] = (
                    momentum_score * 0.8
                )  # Slight penalty for no alpha

        return enhanced_scores

    def _check_early_warning(self, market_state, timestamp: pd.Timestamp) -> float:
        """Early warning system (adapted from original conservative strategy)"""
        warning_multiplier = 1.0

        if not market_state:
            return warning_multiplier

        # Check for crisis conditions
        crisis_conditions = 0
        if market_state.regime.value in ["crisis", "bear"]:
            crisis_conditions += 1
        if market_state.risk_level in ["high", "extreme"]:
            crisis_conditions += 1
        if market_state.volatility_regime == "high":
            crisis_conditions += 1

        # Activate early warning if multiple conditions met
        if crisis_conditions >= self.warning_activation_threshold:
            if not self.early_warning_active:
                self.logger.warning("Conservative early warning activated")
                self.early_warning_active = True
                self.last_warning_date = timestamp.date()
            warning_multiplier = self.warning_position_multiplier
        else:
            # Check cooldown period
            if self.early_warning_active and self.last_warning_date:
                days_since = (timestamp.date() - self.last_warning_date).days
                if days_since > self.warning_cooldown_days:
                    self.logger.info("Conservative early warning deactivated")
                    self.early_warning_active = False

        if self.early_warning_active:
            warning_multiplier = self.warning_position_multiplier

        return warning_multiplier

    def _apply_asset_preferences(
        self, scores: Dict[str, float], market_state, warning_adjustment: float
    ) -> Dict[str, float]:
        """Apply asset class preferences based on market conditions"""
        adjusted_scores = {}

        for symbol, score in scores.items():
            multiplier = 1.0

            # Favor defensive assets in uncertain markets
            if symbol in self.preferred_symbols:  # Risk-off assets
                if market_state and market_state.risk_level in ["high", "extreme"]:
                    multiplier = self.asset_preference_multipliers.get(
                        "risk_off_crisis", 1.5
                    )
                else:
                    multiplier = self.asset_preference_multipliers.get(
                        "risk_off_normal", 1.1
                    )
            elif symbol in self.risk_on_symbols:  # Risk-on assets
                if market_state and market_state.risk_level in ["high", "extreme"]:
                    multiplier = self.asset_preference_multipliers.get(
                        "risk_on_crisis", 0.6
                    )
                else:
                    multiplier = self.asset_preference_multipliers.get(
                        "risk_on_normal", 0.9
                    )

            adjusted_scores[symbol] = score * multiplier * warning_adjustment

        return adjusted_scores

    def _select_final_positions(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Select final positions with original weight allocation logic"""
        if not scores:
            return {}

        # Sort by score (descending)
        sorted_positions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Take top positions up to max_positions
        selected_positions = sorted_positions[: self.max_positions]

        # Apply original weight allocation
        final_positions = {}
        for i, (symbol, score) in enumerate(selected_positions):
            if i < len(self.position_weights):
                final_positions[symbol] = self.position_weights[i]
            else:
                # If we have more positions than weights, use smallest weight
                final_positions[symbol] = min(self.position_weights)

        total_weight = sum(final_positions.values())
        if total_weight > 0:
            final_positions = {s: w / total_weight for s, w in final_positions.items()}

        return final_positions


class EnhancedAggressiveModule:
    def __init__(self, alpha_engine: AlphaEngine, **kwargs):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.alpha_engine = alpha_engine

        # --- Core aggressive parameters (from config) ---
        self.momentum_windows = kwargs.get("aggressive_momentum_windows", [10, 25, 50])
        self.momentum_weights = kwargs.get(
            "aggressive_momentum_weights", [0.5, 0.3, 0.2]
        )
        self.max_positions = kwargs.get("aggressive_max_positions", 12)
        self.max_single_position = kwargs.get("aggressive_max_single_position", 0.12)

        self.equal_weight_blend = kwargs.get("aggressive_equal_weight_blend", 0.5)
        self.score_weight_blend = kwargs.get("aggressive_score_weight_blend", 0.5)

        # --- Alpha enhancement parameters (from config) ---
        self.alpha_enhancement_weight = kwargs.get("aggressive_alpha_weight", 0.4)
        self.momentum_weight = 1 - self.alpha_enhancement_weight

        # --- Filter criteria (from config) ---
        self.min_alpha_confidence = kwargs.get("aggressive_min_confidence", 0.25)
        self.min_momentum_threshold = kwargs.get("min_momentum_threshold", -0.15)

        # --- Market state adjustments (from config) ---
        self.market_regime_adjustments = kwargs.get(
            "aggressive_regime_multipliers",
            {
                "bull": 1.2,
                "strong_bull": 1.3,
                "normal": 1.0,
                "bear": 0.6,
                "crisis": 0.4,
                "volatile": 0.7,
                "recovery": 1.1,
            },
        )
        self.market_risk_adjustments = kwargs.get(
            "aggressive_risk_multipliers",
            {"low": 1.1, "medium": 1.0, "high": 0.7, "extreme": 0.4},
        )

        # --- Position sizing parameters (from config) ---
        self.equal_weight_blend = kwargs.get("aggressive_equal_weight_blend", 0.5)
        self.score_weight_blend = kwargs.get("aggressive_score_weight_blend", 0.5)
        self.acceleration_bonus_factor = kwargs.get(
            "aggressive_acceleration_bonus_factor", 0.1
        )

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_state,
        alpha_signals: Dict[str, CompositeAlphaSignal],
    ) -> Dict[str, float]:
        """Generate aggressive strategy signals with Alpha enhancement"""

        # Step 1: Traditional momentum screening (unchanged core logic)
        momentum_scores = self._calculate_momentum_scores(market_data, timestamp)
        if not momentum_scores:
            return {}

        # Step 2: Alpha enhancement filtering
        alpha_enhanced_scores = self._apply_alpha_enhancement(
            momentum_scores, alpha_signals, market_state
        )

        # Step 3: Market state adjustments
        state_adjusted_scores = self._apply_market_state_adjustments(
            alpha_enhanced_scores, market_state
        )

        # Step 4: Volatility-based position sizing
        final_positions = self._calculate_position_sizes(
            state_adjusted_scores, market_data
        )

        if final_positions:
            self.logger.info(
                f"Aggressive module selected {len(final_positions)} positions"
            )

        return final_positions

    def _calculate_momentum_scores(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate momentum scores with aggressive timeframes"""
        scores = {}

        available_symbols = sorted(
            market_data.index.get_level_values("symbol").unique()
        )

        for symbol in available_symbols:
            try:
                symbol_data = market_data.xs(symbol, level="symbol")
                symbol_data = symbol_data[symbol_data.index <= timestamp]

                if len(symbol_data) < max(self.momentum_windows):
                    continue

                prices = symbol_data["close"].values

                # Calculate momentum for each timeframe
                momentum_values = []
                for window in self.momentum_windows:
                    if len(prices) >= window:
                        momentum = (prices[-1] / prices[-window]) - 1
                        momentum_values.append(momentum)

                if len(momentum_values) == len(self.momentum_windows):
                    # Weighted momentum (favor shorter timeframes for aggressive)
                    composite_momentum = sum(
                        m * w for m, w in zip(momentum_values, self.momentum_weights)
                    )

                    # Apply minimum threshold filter
                    if composite_momentum > self.min_momentum_threshold:
                        scores[symbol] = composite_momentum

            except Exception as e:
                self.logger.debug(f"Error calculating momentum for {symbol}: {e}")

        return scores

    def _apply_alpha_enhancement(
        self,
        momentum_scores: Dict[str, float],
        alpha_signals: Dict[str, CompositeAlphaSignal],
        market_state,
    ) -> Dict[str, float]:
        """Apply Alpha enhancement (more aggressive than conservative module)"""
        enhanced_scores = {}

        for symbol, momentum_score in momentum_scores.items():
            alpha_signal = alpha_signals.get(symbol)

            if (
                alpha_signal
                and alpha_signal.final_confidence >= self.min_alpha_confidence
            ):
                # Higher alpha weight for aggressive strategy
                alpha_component = (
                    alpha_signal.final_score * self.alpha_enhancement_weight
                )
                momentum_component = momentum_score * self.momentum_weight

                # Momentum acceleration bonus (aggressive likes accelerating trends)
                if hasattr(alpha_signal, "sources"):
                    momentum_source = alpha_signal.sources.get("momentum")
                    if momentum_source:
                        accel_component = momentum_source.components.get(
                            "acceleration", 1.0
                        )
                        acceleration_bonus = (
                            accel_component - 1.0
                        ) * self.acceleration_bonus_factor
                    else:
                        acceleration_bonus = 0.0
                else:
                    acceleration_bonus = 0.0

                enhanced_score = (
                    momentum_component + alpha_component + acceleration_bonus
                )
                enhanced_scores[symbol] = enhanced_score
            else:
                # More forgiving than conservative for missing alpha signals
                enhanced_scores[symbol] = momentum_score * 0.9

        return enhanced_scores

    def _apply_market_state_adjustments(
        self, scores: Dict[str, float], market_state
    ) -> Dict[str, float]:
        """Adjust scores based on market regime (aggressive adapts more)"""
        if not market_state:
            return scores

        regime = market_state.regime.value
        multiplier = self.market_regime_adjustments.get(regime, 1.0)

        # Additional risk level adjustment
        risk_multiplier = self.market_risk_adjustments.get(market_state.risk_level, 1.0)

        final_multiplier = multiplier * risk_multiplier

        adjusted_scores = {
            symbol: score * final_multiplier for symbol, score in scores.items()
        }

        return adjusted_scores

    def _calculate_position_sizes(
        self, scores: Dict[str, float], market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate position sizes with volatility adjustment"""
        if not scores:
            return {}

            # Sort and filter top candidates
        sorted_candidates = sorted(
            scores.items(), key=lambda x: (x[1], x[0]), reverse=True
        )
        top_candidates = sorted_candidates[: self.max_positions]

        if not top_candidates:
            return {}

        # Calculate base equal weight
        base_weight = 1.0 / len(top_candidates)

        # Apply score-based weighting
        total_score = sum(abs(score) for _, score in top_candidates)
        final_positions = {}

        for symbol, score in top_candidates:
            if total_score > 0:
                score_weight = abs(score) / total_score
                blended_weight = (base_weight * self.equal_weight_blend) + (
                    score_weight * self.score_weight_blend
                )
            else:
                blended_weight = base_weight

            # Apply single position limit
            final_weight = min(blended_weight, self.max_single_position)
            final_positions[symbol] = final_weight

        return final_positions


class HybridDualAlphaStrategy(BaseStrategy):
    def __init__(self, events_queue, symbol_list, **kwargs):
        super().__init__(events_queue, symbol_list, **kwargs)

        self.initial_capital = kwargs.get(
            "initial_capital", getattr(config, "INITIAL_CAPITAL", 1000000)
        )

        self.alpha_engine = AlphaEngine(**kwargs)

        self.conservative_module = EnhancedConservativeModule(
            self.alpha_engine, **kwargs
        )
        self.aggressive_module = EnhancedAggressiveModule(self.alpha_engine, **kwargs)

        # --- Dynamic allocation controller (from config) ---
        self.base_conservative_allocation = kwargs.get(
            "base_conservative_allocation", 0.65
        )
        self.base_aggressive_allocation = kwargs.get("base_aggressive_allocation", 0.35)
        self.allocation_adaptation_enabled = kwargs.get(
            "allocation_adaptation_enabled", True
        )
        self.dynamic_allocation_map = kwargs.get(
            "dynamic_allocation_map",
            {
                "crisis": {"conservative": 0.85, "aggressive": 0.15},
                "bear": {"conservative": 0.75, "aggressive": 0.25},
                "volatile": {"conservative": 0.70, "aggressive": 0.30},
                "recovery": {"conservative": 0.60, "aggressive": 0.40},
                "normal": {"conservative": 0.65, "aggressive": 0.35},
                "bull": {"conservative": 0.55, "aggressive": 0.45},
                "strong_bull": {"conservative": 0.50, "aggressive": 0.50},
            },
        )

        # --- Execution parameters (from config) ---
        self.target_max_utilization = kwargs.get("target_max_utilization", 0.95)
        self.min_target_weight = kwargs.get("min_target_weight", 0.01)
        self.weight_to_score_multiplier = kwargs.get("weight_to_score_multiplier", 10)

        # --- Rebalancing parameters (from config) ---
        self.rebalance_frequency = kwargs.get("rebalance_frequency", "weekly")
        self.allocation_change_threshold = kwargs.get(
            "allocation_change_threshold", 0.05
        )

        # --- Risk management integration ---
        risk_params = {
            "max_drawdown_threshold": getattr(tp, "RISK_PARAMS", {})
            .get("drawdown_control", {})
            .get("max_drawdown_threshold", 0.15),
            "emergency_drawdown": getattr(tp, "RISK_PARAMS", {})
            .get("drawdown_control", {})
            .get("emergency_drawdown_threshold", 0.25),
            "max_positions": getattr(tp, "TRADING_PARAMS", {})
            .get("position_sizing", {})
            .get("max_positions", 15),
            "max_single_position": getattr(tp, "RISK_PARAMS", {})
            .get("portfolio_limits", {})
            .get("max_single_position", 0.15),
        }
        self.risk_manager = RiskManager(self.initial_capital, **risk_params)

        stop_params = {
            "enabled": getattr(tp, "RISK_PARAMS", {})
            .get("stop_loss", {})
            .get("enabled", True),
            "fixed_stop_loss": getattr(tp, "RISK_PARAMS", {})
            .get("stop_loss", {})
            .get("fixed_stop_loss", 0.08),
            "trailing_activation": getattr(tp, "RISK_PARAMS", {})
            .get("stop_loss", {})
            .get("trailing_stop_activation", 0.05),
            "trailing_distance": getattr(tp, "RISK_PARAMS", {})
            .get("stop_loss", {})
            .get("trailing_stop_distance", 0.03),
        }
        self.stop_manager = StopManager(**stop_params)

        # --- Performance tracking parameters (from config) ---
        self.cash_buffer_percentage = kwargs.get(
            "cash_buffer_percentage", getattr(tp, "CASH_BUFFER_PERCENTAGE", 0.05)
        )
        self.normalization_factor = kwargs.get(
            "normalization_factor", getattr(tp, "NORMALIZATION_FACTOR", 0.95)
        )

        # --- ATR estimation parameters (from config) ---
        self.default_atr_percentage = kwargs.get(
            "default_atr_percentage", getattr(tp, "DEFAULT_ATR_PERCENTAGE", 0.02)
        )

        # --- State tracking ---
        self.last_rebalance_date = None
        self.market_data_cache = {}
        self.current_allocation = None
        self.last_alpha_signals = {}

        # --- Performance tracking ---
        self.rebalance_count = 0
        self.signal_count = 0
        self.allocation_changes = 0

        self.logger.info(
            "HybridDualAlphaStrategy initialized successfully from config parameters."
        )

    def set_portfolio_reference(self, portfolio):
        """Set portfolio reference for data access"""
        self.portfolio_ref = portfolio
        if hasattr(portfolio, "data_handler"):
            self.data_handler = portfolio.data_handler
            self.logger.info("Portfolio reference set successfully")

    def calculate_signal(self, event):
        """Main signal calculation entry point"""
        if event.type != "Market":
            return

        # Update market data cache
        self._update_market_data_cache(event)

        if hasattr(self, "_market_event_count"):
            self._market_event_count += 1
        else:
            self._market_event_count = 0

        if self._market_event_count % 100 == 0:
            self._validate_portfolio_allocation(event.timestamp)

        # Check rebalancing schedule
        current_date = event.timestamp.date()
        if self._should_rebalance(current_date):
            self._execute_hybrid_rebalance(event.timestamp)

        # Monitor stop losses
        if self.stop_manager.enabled:
            self._check_stop_losses(event.timestamp)

    def _update_market_data_cache(self, event):
        """Update internal market data cache"""
        symbol = event.symbol
        self.market_data_cache[symbol] = {
            "timestamp": event.timestamp,
            "close": event.data["close"],
            "volume": event.data.get("volume", 0),
            "high": event.data.get("high", event.data["close"]),
            "low": event.data.get("low", event.data["close"]),
        }

    def _should_rebalance(self, current_date) -> bool:
        """Determine if rebalancing is needed"""
        if self.last_rebalance_date is None:
            return True

        days_since = (current_date - self.last_rebalance_date).days

        # Get rebalance frequency from config
        freq = self.rebalance_frequency

        if freq == "daily":
            return days_since >= 1
        elif freq == "weekly":
            return days_since >= 7
        else:  # monthly
            return days_since >= 30

    def _execute_hybrid_rebalance(self, timestamp: pd.Timestamp):
        """Execute hybrid dual strategy rebalancing"""
        self.logger.info(f"=== Executing Hybrid Rebalance at {timestamp.date()} ===")

        # Step 1: Prepare market data
        market_data = self._prepare_market_data(timestamp)
        if market_data is None or market_data.empty:
            self.logger.warning("Insufficient market data for rebalancing")
            return

        # Step 2: Generate Alpha signals for both modules
        alpha_signals = self.alpha_engine.generate_alpha_signals(
            market_data, self.symbol_list, timestamp
        )
        self.last_alpha_signals = alpha_signals

        market_state = self.alpha_engine.market_detector.last_market_state

        # Step 3: Calculate dynamic allocation
        current_allocation = self._calculate_dynamic_allocation(market_state, timestamp)

        # Step 4: Generate signals from both modules
        conservative_signals = self.conservative_module.generate_signals(
            market_data, timestamp, market_state, alpha_signals
        )

        aggressive_signals = self.aggressive_module.generate_signals(
            market_data, timestamp, market_state, alpha_signals
        )

        # Step 5: Risk assessment and filtering
        portfolio_value = self._get_portfolio_value()
        position_values = self._get_position_values()

        risk_assessment = self.risk_manager.assess_portfolio_risk(
            portfolio_value, position_values, market_state, timestamp
        )

        if risk_assessment.mode in [RiskMode.RISK_OFF, RiskMode.EMERGENCY]:
            self._validate_portfolio_allocation(timestamp)

        # Step 6: Combine and scale signals
        final_signals = self._combine_module_signals(
            conservative_signals,
            aggressive_signals,
            current_allocation,
            risk_assessment,
        )

        # Step 7: Generate trading orders
        self._generate_trading_orders(timestamp, final_signals)

        self._validate_portfolio_allocation(timestamp)

        # Update state
        self.last_rebalance_date = timestamp.date()
        self.current_allocation = current_allocation
        self.rebalance_count += 1

    def _prepare_market_data(self, timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Prepare market data DataFrame for Alpha engine"""
        if not hasattr(self, "data_handler") or not hasattr(
            self.data_handler, "all_data"
        ):
            self.logger.error(
                "Data handler or its 'all_data' attribute is not available."
            )
            return None

        all_data = self.data_handler.all_data
        if all_data is None or all_data.empty:
            self.logger.error("Data handler's 'all_data' is empty.")
            return None

        lookback_days = 252
        start_date = timestamp - pd.Timedelta(days=lookback_days)

        timestamps = all_data.index.get_level_values("timestamp")
        time_mask = (timestamps >= start_date) & (timestamps <= timestamp)
        historical_data = all_data[time_mask].copy()

        if historical_data.empty:
            self.logger.warning(
                f"No historical data in lookback window ending at {timestamp}"
            )
            return None

        # Ensure required columns exist
        required_cols = ["close", "volume", "high", "low"]
        for col in required_cols:
            if col not in historical_data.columns:
                if col == "volume":
                    historical_data[col] = 1000000
                else:
                    historical_data[col] = historical_data["close"]

        return historical_data

    def _calculate_dynamic_allocation(
        self, market_state, timestamp: pd.Timestamp
    ) -> StrategyAllocation:
        base_conservative = self.base_conservative_allocation
        base_aggressive = self.base_aggressive_allocation

        if not self.allocation_adaptation_enabled or not market_state:
            return StrategyAllocation(
                conservative_ratio=base_conservative,
                aggressive_ratio=base_aggressive,
                cash_ratio=0.0,
                timestamp=timestamp,
                reason="static_allocation",
            )

        # Market regime adjustments
        regime = market_state.regime.value
        if regime in self.dynamic_allocation_map:
            target_conservative = self.dynamic_allocation_map[regime]["conservative"]
            target_aggressive = self.dynamic_allocation_map[regime]["aggressive"]
            reason = f"regime_{regime}"
        else:
            target_conservative = base_conservative
            target_aggressive = base_aggressive
            reason = "default_allocation"

        # Risk level adjustments
        if market_state.risk_level == "extreme":
            target_conservative = min(target_conservative + 0.15, 0.90)
            target_aggressive = max(target_aggressive - 0.15, 0.10)
            reason += "_risk_extreme"
        elif market_state.risk_level == "high":
            target_conservative = min(target_conservative + 0.08, 0.85)
            target_aggressive = max(target_aggressive - 0.08, 0.15)
            reason += "_risk_high"

        # Log allocation changes
        if (
            self.current_allocation is None
            or abs(target_conservative - self.current_allocation.conservative_ratio)
            > self.allocation_change_threshold
        ):
            self.logger.info(
                f"Allocation change: Conservative {target_conservative:.1%}, "
                f"Aggressive {target_aggressive:.1%} ({reason})"
            )
            self.allocation_changes += 1

        return StrategyAllocation(
            conservative_ratio=target_conservative,
            aggressive_ratio=target_aggressive,
            cash_ratio=0.0,
            timestamp=timestamp,
            reason=reason,
        )

    def _combine_module_signals(
        self,
        conservative_signals: Dict[str, float],
        aggressive_signals: Dict[str, float],
        allocation: StrategyAllocation,
        risk_assessment,
    ) -> Dict[str, float]:
        scaled_conservative = {
            s: w * allocation.conservative_ratio
            for s, w in conservative_signals.items()
        }
        scaled_aggressive = {
            s: w * allocation.aggressive_ratio for s, w in aggressive_signals.items()
        }

        combined_weights = defaultdict(float)
        for symbol in set(scaled_conservative.keys()) | set(scaled_aggressive.keys()):
            combined_weights[symbol] = scaled_conservative.get(
                symbol, 0.0
            ) + scaled_aggressive.get(symbol, 0.0)

        risk_multiplier = risk_assessment.position_size_multiplier
        if risk_multiplier < 1.0:
            logging.info(
                f"Applying global risk multiplier of {risk_multiplier:.2f} to all target weights."
            )
            for symbol in combined_weights:
                combined_weights[symbol] *= risk_multiplier

        position_limit = risk_assessment.position_limit
        if len(combined_weights) > position_limit:
            logging.warning(
                f"Strategy proposed {len(combined_weights)} positions, but risk limit is {position_limit}. "
                f"Trimming to the top {position_limit} positions by weight."
            )
            # Sort by weight and take the top N
            top_positions = sorted(
                combined_weights.items(), key=lambda item: item[1], reverse=True
            )[:position_limit]
            combined_weights = dict(top_positions)

        total_weight = sum(combined_weights.values())
        if total_weight > self.target_max_utilization:
            normalization_factor = self.target_max_utilization / total_weight
            logging.info(
                f"Normalizing total portfolio weight from {total_weight:.2%} to {self.target_max_utilization:.2%}"
            )
            for symbol in combined_weights:
                combined_weights[symbol] *= normalization_factor

        # Filter out any symbols that are on the forced exit list
        final_signals = {
            s: w
            for s, w in combined_weights.items()
            if s not in risk_assessment.force_exit_symbols
        }
        if risk_assessment.force_exit_symbols:
            logging.warning(
                f"Forced exit for symbols: {risk_assessment.force_exit_symbols}. They will be removed from targets."
            )

        return final_signals

    def _generate_trading_orders(
        self, timestamp: pd.Timestamp, target_signals: Dict[str, float]
    ):
        """Generate actual trading orders"""
        current_weights = self._get_current_position_weights()
        all_symbols = set(current_weights.keys()) | set(target_signals.keys())

        for symbol in sorted(list(all_symbols)):
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_signals.get(symbol, 0.0)

            # Define a dead zone to prevent tiny, frequent rebalances
            rebalance_threshold = 0.005  # 0.5% weight change

            # Case 1: Target weight is effectively zero, and we hold a position -> EXIT
            if target_weight < self.min_target_weight and current_weight > 0:
                signal = SignalEvent(timestamp, symbol, "Exit")
                self.events.put(signal)
                self.logger.info(
                    f"HYBRID EXIT: {symbol} (target weight is zero, current was {current_weight:.2%})"
                )
            # Case 2: Target weight is positive and significantly different from current weight
            elif (
                target_weight >= self.min_target_weight
                and abs(target_weight - current_weight) > rebalance_threshold
            ):
                if target_weight > current_weight:
                    signal = SignalEvent(timestamp, symbol, "Long", score=target_weight)
                    self.events.put(signal)
                    self.logger.info(
                        f"HYBRID UPSIZE: {symbol} (current={current_weight:.2%}, target={target_weight:.2%})"
                    )
                else:
                    reduction_ratio = (current_weight - target_weight) / current_weight
                    signal = SignalEvent(
                        timestamp, symbol, "Reduce", score=reduction_ratio
                    )
                    self.events.put(signal)
                    self.logger.info(
                        f"HYBRID REDUCE: {symbol} (current={current_weight:.2%}, target={target_weight:.2%})"
                    )

    def _get_current_position_weights(self) -> Dict[str, float]:
        portfolio_value = self._get_portfolio_value()
        position_values = self._get_position_values()

        if portfolio_value <= 0:
            return {}

        return {
            symbol: value / portfolio_value for symbol, value in position_values.items()
        }

    def _check_stop_losses(self, timestamp: pd.Timestamp):
        """Monitor and trigger stop losses"""
        if not self.stop_manager.enabled:
            return

        # Prepare market data for stops
        market_data_for_stops = {}
        for symbol in self._get_current_positions():
            if symbol in self.market_data_cache:
                cache_data = self.market_data_cache[symbol]
                market_data_for_stops[symbol] = {
                    "price": cache_data["close"],
                    "atr": self._estimate_atr(symbol),
                    "volatility": 0.20,  # Default volatility estimate
                }

        # Check stops
        market_state = self.alpha_engine.market_detector.last_market_state
        stop_actions = self.stop_manager.update_stops(
            market_data_for_stops, market_state, timestamp
        )

        # Process stop triggers
        for action in stop_actions:
            if action.action == "exit":
                signal = SignalEvent(timestamp, action.symbol, "Exit")
                self.events.put(signal)
                self.logger.warning(f"STOP LOSS: {action.symbol} - {action.reason}")
                self.stop_manager.remove_position(action.symbol)

    def _estimate_atr(self, symbol: str) -> float:
        if symbol in self.market_data_cache:
            current_price = self.market_data_cache[symbol]["close"]
            return current_price * self.default_atr_percentage
        return 0.0

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        if self.portfolio_ref:
            return self.portfolio_ref.current_holdings.get(
                "total", self.initial_capital
            )
        return self.initial_capital

    def _get_current_positions(self) -> Set[str]:
        """Get currently held position symbols"""
        if self.portfolio_ref:
            positions = set()
            for symbol, quantity in self.portfolio_ref.current_positions.items():
                if quantity > 0:
                    positions.add(symbol)
            return positions
        return set()

    def _get_position_values(self) -> Dict[str, float]:
        """Get current position values"""
        if self.portfolio_ref:
            values = {}
            for symbol, quantity in self.portfolio_ref.current_positions.items():
                if quantity > 0 and symbol in self.market_data_cache:
                    price = self.market_data_cache[symbol]["close"]
                    values[symbol] = quantity * price
            return values
        return {}

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status"""
        status = {
            "strategy_type": "hybrid_dual_alpha",
            "last_rebalance": self.last_rebalance_date,
            "rebalance_count": self.rebalance_count,
            "signal_count": self.signal_count,
            "allocation_changes": self.allocation_changes,
            "current_positions": len(self._get_current_positions()),
            "portfolio_value": self._get_portfolio_value(),
        }

        # Add current allocation
        if self.current_allocation:
            status["current_allocation"] = {
                "conservative": f"{self.current_allocation.conservative_ratio:.1%}",
                "aggressive": f"{self.current_allocation.aggressive_ratio:.1%}",
                "reason": self.current_allocation.reason,
            }

        # Add risk status
        if hasattr(self.risk_manager, "current_mode"):
            status["risk_mode"] = self.risk_manager.current_mode.value
            status["drawdown"] = f"{self.risk_manager.current_drawdown:.1%}"

        # Add market regime
        if hasattr(self.alpha_engine.market_detector, "last_market_state"):
            market_state = self.alpha_engine.market_detector.last_market_state
            if market_state:
                status["market_regime"] = market_state.regime.value
                status["risk_level"] = market_state.risk_level

        # Add alpha signal stats
        if self.last_alpha_signals:
            status["alpha_signals"] = {
                "total": len(self.last_alpha_signals),
                "avg_confidence": np.mean(
                    [s.final_confidence for s in self.last_alpha_signals.values()]
                ),
                "long_signals": sum(
                    1 for s in self.last_alpha_signals.values() if s.action == "long"
                ),
            }

        return status

    def _validate_portfolio_allocation(self, timestamp: pd.Timestamp):
        """Emergency portfolio allocation check and correction"""
        portfolio_value = self._get_portfolio_value()
        position_values = self._get_position_values()

        if portfolio_value <= 0:
            return

        # Check total exposure
        total_exposure = sum(position_values.values())
        exposure_ratio = total_exposure / portfolio_value

        # Log current allocation status
        if exposure_ratio > 0.95:
            self.logger.warning(
                f"High exposure warning: {exposure_ratio:.1%} of portfolio invested"
            )

        # Critical check: Over-invested (shouldn't happen but safety check)
        if exposure_ratio > 1.01:  # Allow 1% margin for rounding
            self.logger.error(f"CRITICAL: Over-invested at {exposure_ratio:.1%}")

            # Force reduce largest positions
            sorted_positions = sorted(
                position_values.items(), key=lambda x: (x[1], x[0]), reverse=True
            )

            excess_value = total_exposure - (portfolio_value * 0.95)  # Target 95% max

            for symbol, value in sorted_positions[:3]:  # Check top 3 positions
                position_weight = value / portfolio_value

                # If single position exceeds limit
                if position_weight > self.risk_manager.max_single_position:
                    reduction_ratio = 1 - (
                        self.risk_manager.max_single_position / position_weight
                    )
                    signal = SignalEvent(
                        timestamp, symbol, "Reduce", score=reduction_ratio
                    )
                    self.events.put(signal)
                    self.logger.warning(
                        f"Emergency reduction: {symbol} from {position_weight:.1%} to {self.risk_manager.max_single_position:.1%}"
                    )

        # Check individual position limits
        for symbol, value in position_values.items():
            position_weight = value / portfolio_value

            # Warn if approaching single position limit
            if position_weight > self.risk_manager.max_single_position * 0.9:
                self.logger.warning(
                    f"{symbol} approaching position limit: {position_weight:.1%}"
                )
