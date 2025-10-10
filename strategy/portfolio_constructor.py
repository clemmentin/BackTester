import logging
import numpy as np
from typing import Dict, Set
import config
import pandas as pd
from functools import partial
from strategy.contracts import (
    RawAlphaSignal,
    RawAlphaSignalDict,
    StratifiedCandidatePool,
    RiskBudget,
    FinalTargetPortfolio,
)


class PortfolioConstructor:
    """
    Acts as Layer 4: The Portfolio Construction Layer.
    FOCUS: Pure portfolio weight calculation and constraint application
    """

    def __init__(self, **kwargs):
        """Initialize portfolio construction parameters"""
        self.logger = logging.getLogger(__name__)

        get_sizing_param = partial(
            config.get_param_with_override, kwargs, "position_sizing"
        )
        get_limit_param = partial(
            config.get_param_with_override, kwargs, "portfolio_limits"
        )
        get_exec_param = partial(
            config.get_param_with_override, kwargs, "strategy_execution"
        )

        # Weighting scheme configuration
        self.weighting_scheme = get_sizing_param(
            "weighting_scheme", "stratified_score_weighted"
        )

        # Position limits
        self.max_single_position_weight = get_limit_param("max_single_position", 0.12)
        self.min_position_weight = get_sizing_param("min_position_weight", 0.01)

        # Minimum trade size (NEW)
        self.min_trade_size_dollars = get_sizing_param("min_trade_size_dollars", 500)

        # Stratification bonuses
        self.consensus_multiplier = get_sizing_param("consensus_multiplier", 1.2)
        self.conservative_only_multiplier = get_sizing_param(
            "conservative_only_multiplier", 1.0
        )
        self.aggressive_only_multiplier = get_sizing_param(
            "aggressive_only_multiplier", 0.9
        )
        self.consensus_max_share = get_sizing_param("consensus_max_share", 0.4)

        # Rebalancing parameters
        self.rebalance_threshold = get_sizing_param("rebalance_threshold", 0.10)
        self.min_holding_period = get_exec_param("min_holding_days", 7)
        self.position_entry_history = {}

        self.logger.info(
            f"PortfolioConstructor initialized: "
            f"scheme={self.weighting_scheme}, "
            f"max_single={self.max_single_position_weight:.1%}, "
            f"min_single={self.min_position_weight:.1%}, "
            f"min_trade=${self.min_trade_size_dollars:,.0f}"
        )

    def build_target_portfolio(
        self,
        raw_alpha_signals: RawAlphaSignalDict,
        stratified_pool: StratifiedCandidatePool,
        risk_budget: RiskBudget,
        market_state=None,
        current_positions: Dict[str, float] = None,
        timestamp: pd.Timestamp = None,
    ) -> FinalTargetPortfolio:
        """
        Constructs the final target portfolio using stratified candidate selection.

        ASSUMPTION: Signals have already been filtered by Layer 2
        """
        self.logger.info("--- [Layer 4] Starting Portfolio Construction ---")

        # Step 1: Filter to only pool candidates (basic validation only)
        valid_signals = self._filter_to_pool_candidates(
            raw_alpha_signals, stratified_pool
        )

        if not valid_signals:
            self.logger.warning(
                f"No valid candidates in stratified pool (pool size: {len(stratified_pool.all_candidates)})"
            )
            return {}

        # Step 2: Sort and select top candidates
        sorted_candidates = sorted(
            valid_signals.items(), key=lambda x: x[1].score, reverse=True
        )

        max_positions = min(risk_budget.max_position_count, len(sorted_candidates))
        min_positions = max(3, max_positions // 2)
        target_positions = max(min_positions, max_positions)

        selected_candidates = dict(sorted_candidates[:target_positions])

        self.logger.info(f"Selected top {len(selected_candidates)} candidates")

        # Step 3: Calculate weights
        try:
            weights = self._calculate_stratified_weights(
                selected_candidates, stratified_pool, risk_budget
            )

            if not weights:
                self.logger.warning("Final portfolio is empty after all steps.")
                return {}

            total_value = sum(weights.values())
            max_value = max(weights.values())

            self.logger.info(
                f"Portfolio construction complete: {len(weights)} positions, "
                f"total value: ${total_value:,.0f}, max single: ${max_value:,.0f}"
            )

            return weights

        except Exception as e:
            self.logger.error(f"Error calculating weights: {e}", exc_info=True)
            return {}

    def _filter_to_pool_candidates(
        self,
        raw_alpha_signals: RawAlphaSignalDict,
        stratified_pool: StratifiedCandidatePool,
    ) -> Dict[str, RawAlphaSignal]:
        """
        Basic filter: only include symbols in the stratified pool.

        NOTE: Advanced quality filtering should happen in Layer 2
        """
        valid = {}

        for symbol in stratified_pool.all_candidates:
            if symbol in raw_alpha_signals:
                signal = raw_alpha_signals[symbol]
                # Only basic validation
                if signal.score != 0:  # Non-zero score
                    valid[symbol] = signal

        self.logger.info(
            f"Pool filter: {len(valid)}/{len(stratified_pool.all_candidates)} candidates"
        )

        return valid

    def _apply_rebalance_buffer(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, float],
        timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        """Apply rebalancing buffer to reduce unnecessary turnover."""
        adjusted_weights = {}
        trades_avoided = 0

        all_symbols = set(target_weights.keys()) | set(current_positions.keys())

        for symbol in all_symbols:
            target_w = target_weights.get(symbol, 0.0)
            current_w = current_positions.get(symbol, 0.0)

            # Currently holding position
            if current_w > 0:
                # Check holding period
                if symbol in self.position_entry_history:
                    days_held = (timestamp - self.position_entry_history[symbol]).days

                    if days_held < self.min_holding_period and target_w > 0:
                        adjusted_weights[symbol] = current_w
                        trades_avoided += 1
                        continue

                # Check weight change threshold
                if target_w > 0:
                    weight_change_pct = abs(target_w - current_w) / current_w

                    if weight_change_pct < self.rebalance_threshold:
                        adjusted_weights[symbol] = current_w
                        trades_avoided += 1
                    else:
                        adjusted_weights[symbol] = target_w

            # New position
            elif target_w > 0:
                adjusted_weights[symbol] = target_w
                self.position_entry_history[symbol] = timestamp

        # Renormalize
        current_total = sum(adjusted_weights.values())
        target_total = sum(target_weights.values())

        if current_total > 0 and abs(current_total - target_total) > 0.01:
            scaling_factor = target_total / current_total
            adjusted_weights = {
                s: w * scaling_factor for s, w in adjusted_weights.items()
            }

        self.logger.info(f"Rebalancing: {trades_avoided} positions held stable")

        return adjusted_weights

    def _calculate_stratified_weights(
        self,
        selected_signals: Dict[str, RawAlphaSignal],
        stratified_pool: StratifiedCandidatePool,
        risk_budget: RiskBudget,
    ) -> FinalTargetPortfolio:
        """Calculate position weights using stratified, score-weighted method."""
        if not selected_signals:
            return {}

        signal_list = list(selected_signals.items())
        signal_list.sort(key=lambda x: x[1].score, reverse=True)

        weights = {}
        total_rank_score = 0.0

        # Calculate adjusted score for each candidate
        for rank, (symbol, signal) in enumerate(signal_list, start=1):
            rank_score = np.exp(-0.1 * (rank - 1))

            # Apply stratification multiplier
            if symbol in stratified_pool.consensus_candidates:
                multiplier = self.consensus_multiplier
            elif symbol in stratified_pool.conservative_only:
                multiplier = self.conservative_only_multiplier
            else:
                multiplier = self.aggressive_only_multiplier

            adjusted_score = rank_score * multiplier
            weights[symbol] = adjusted_score
            total_rank_score += adjusted_score

        if total_rank_score <= 0:
            return {}

        target_gross_exposure = risk_budget.target_portfolio_exposure

        final_position_values = {
            symbol: (score / total_rank_score) * target_gross_exposure
            for symbol, score in weights.items()
        }

        # Apply weight constraints (includes min trade size filter)
        final_position_values = self._apply_weight_constraints(
            final_position_values, target_gross_exposure
        )

        return final_position_values

    def _apply_weight_constraints(
        self, position_values: Dict[str, float], target_exposure: float
    ) -> Dict[str, float]:
        """
        Apply min/max weight constraints and minimum trade size filter.

        Steps:
        1. Cap oversized positions
        2. Filter positions below minimum dollar amount
        3. Filter positions below minimum percentage
        4. Redistribute freed capital
        """
        max_single_position_value = self.max_single_position_weight * target_exposure

        # Step 1: Cap oversized positions
        capped_values = {}
        excess_value = 0.0

        for symbol, value in position_values.items():
            if value > max_single_position_value:
                excess_value += value - max_single_position_value
                capped_values[symbol] = max_single_position_value
            else:
                capped_values[symbol] = value

        # Redistribute excess from capped positions
        if excess_value > 0:
            uncapped_sum = sum(
                v for s, v in capped_values.items() if v < max_single_position_value
            )
            if uncapped_sum > 0:
                for symbol, value in capped_values.items():
                    if value < max_single_position_value:
                        capped_values[symbol] += excess_value * (value / uncapped_sum)

        # Step 2: Filter by minimum dollar amount (NEW)
        dollar_filtered = {}
        filtered_by_dollar = []

        for symbol, value in capped_values.items():
            if value >= self.min_trade_size_dollars:
                dollar_filtered[symbol] = value
            else:
                filtered_by_dollar.append((symbol, value))

        if filtered_by_dollar:
            self.logger.info(
                f"Filtered {len(filtered_by_dollar)} positions below ${self.min_trade_size_dollars:,.0f}: "
                f"{', '.join([f'{s}(${v:.0f})' for s, v in filtered_by_dollar[:5]])}"
                f"{' ...' if len(filtered_by_dollar) > 5 else ''}"
            )

        # Step 3: Filter by minimum percentage
        min_position_value = self.min_position_weight * target_exposure
        pct_filtered = {}
        filtered_by_pct = []

        for symbol, value in dollar_filtered.items():
            if value >= min_position_value:
                pct_filtered[symbol] = value
            else:
                filtered_by_pct.append((symbol, value))

        if filtered_by_pct:
            self.logger.debug(
                f"Filtered {len(filtered_by_pct)} positions below {self.min_position_weight:.1%}: "
                f"{', '.join([f'{s}(${v:.0f})' for s, v in filtered_by_pct[:3]])}"
            )

        final_values = pct_filtered

        # Step 4: Renormalize to target exposure
        current_total = sum(final_values.values())
        if (
            current_total > 0
            and abs(current_total - target_exposure) > 0.01 * target_exposure
        ):
            rescale_factor = target_exposure / current_total
            final_values = {s: v * rescale_factor for s, v in final_values.items()}

            self.logger.debug(
                f"Renormalized portfolio from ${current_total:,.0f} to ${target_exposure:,.0f} "
                f"(factor={rescale_factor:.4f})"
            )

        return final_values
