import logging
import numpy as np
from typing import Dict, Set
import config
import pandas as pd
from functools import partial
from strategy.contracts import (
    RawAlphaSignal,
    RawAlphaSignalDict,
    StratifiedCandidatePool,  # Kept for type hinting, but logic is removed.
    RiskBudget,
    FinalTargetPortfolio,
)


class PortfolioConstructor:
    """
    Acts as Layer 4: Portfolio Construction.
    Constructs a portfolio for a SINGLE alpha sleeve, with weights driven purely by Bayesian EV.
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        get_sizing_param = partial(
            config.get_param_with_override, kwargs, "position_sizing"
        )
        get_limit_param = partial(
            config.get_param_with_override, kwargs, "portfolio_limits"
        )

        # MODIFICATION: Removed stratified/consensus parameters as they don't apply within a single sleeve.
        self.weighting_scheme = get_sizing_param("weighting_scheme", "ev_proportional")
        self.max_single_position_weight = get_limit_param("max_single_position", 0.10)
        self.min_position_weight = get_sizing_param("min_position_weight", 0.015)
        self.min_trade_size_dollars = get_sizing_param("min_trade_size_dollars", 500)

        self.logger.info(
            f"PortfolioConstructor Initialized: scheme={self.weighting_scheme}, "
            f"max_single={self.max_single_position_weight:.1%}, min_single={self.min_position_weight:.1%}"
        )

    def build_target_portfolio(
        self,
        alpha_signals: RawAlphaSignalDict,
        risk_budget: RiskBudget,
        **kwargs,  # Accept other potential args but ignore them.
    ) -> FinalTargetPortfolio:
        """
        Constructs the final target portfolio for a single alpha sleeve.
        MODIFIED: Simplified to remove stratified pool logic.
        """
        if not alpha_signals:
            return {}

        # Step 1: Sort candidates strictly by Bayesian EV.
        # Signals without positive EV are implicitly filtered by the quality filter,
        # but we ensure sorting is correct here.
        sorted_candidates = sorted(
            alpha_signals.items(),
            key=lambda x: x[1].expected_value,
            reverse=True,
        )

        # Step 2: Select top candidates based on the sleeve's position count budget.
        max_positions = min(risk_budget.max_position_count, len(sorted_candidates))
        selected_candidates = dict(sorted_candidates[:max_positions])

        if not selected_candidates:
            return {}

        # Step 3: Calculate weights based on EV.
        weights = self._calculate_ev_proportional_weights(
            selected_candidates, risk_budget
        )

        if not weights:
            self.logger.warning("Sleeve portfolio is empty after weight calculation.")
            return {}

        return weights

    def _calculate_ev_proportional_weights(
        self,
        selected_signals: Dict[str, RawAlphaSignal],
        risk_budget: RiskBudget,
    ) -> FinalTargetPortfolio:
        """
        Calculates position weights strictly proportional to their positive Expected Value.
        MODIFIED: This is now the core weighting logic, replacing the stratified approach.
        """
        if not selected_signals:
            return {}

        # Use positive EV as the basis for weighting.
        # A small epsilon ensures that signals with zero EV don't cause division errors.
        weights = {
            symbol: max(signal.expected_value, 1e-9)
            for symbol, signal in selected_signals.items()
        }

        total_ev = sum(weights.values())
        if total_ev <= 0:
            return {}

        target_gross_exposure = risk_budget.target_portfolio_exposure

        # Allocate capital proportional to each signal's share of the total EV.
        final_position_values = {
            symbol: (ev / total_ev) * target_gross_exposure
            for symbol, ev in weights.items()
        }

        # Apply final weight constraints (min/max position size).
        # This logic is now self-contained within the constructor for the sleeve.
        final_position_values = self._apply_weight_constraints(
            final_position_values, target_gross_exposure
        )
        return final_position_values

    def _apply_weight_constraints(
        self, position_values: Dict[str, float], target_exposure: float
    ) -> Dict[str, float]:
        """Apply min/max weight constraints for the sleeve."""
        if not position_values:
            return {}

        max_single_position_value = self.max_single_position_weight * target_exposure

        # Cap oversized positions
        capped_values = {}
        excess_value = 0.0
        for symbol, value in position_values.items():
            if value > max_single_position_value:
                excess_value += value - max_single_position_value
                capped_values[symbol] = max_single_position_value
            else:
                capped_values[symbol] = value

        # Redistribute excess value from capped positions
        if excess_value > 0:
            uncapped_sum = sum(
                v for s, v in capped_values.items() if v < max_single_position_value
            )
            if uncapped_sum > 0:
                for symbol, value in capped_values.items():
                    if value < max_single_position_value:
                        capped_values[symbol] += excess_value * (value / uncapped_sum)

        # Filter by minimum position size
        min_position_value = self.min_position_weight * target_exposure
        final_values = {
            s: v for s, v in capped_values.items() if v >= min_position_value
        }

        if not final_values:
            return {}

        # Renormalize to ensure the sleeve's total exposure matches its budget
        current_total = sum(final_values.values())
        if current_total > 0:
            rescale_factor = target_exposure / current_total
            final_values = {s: v * rescale_factor for s, v in final_values.items()}

        return final_values
