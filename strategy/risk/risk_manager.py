import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from collections import deque
from functools import partial
import numpy as np
import config
from config import get_param_with_override
from strategy.alpha.market_detector import MarketRegime, MarketState
from strategy.contracts import RiskBudget


class RiskLevel(Enum):
    """Internal risk levels used to determine the final budget."""

    NORMAL = "normal"
    CAUTIOUS = "cautious"
    RECOVERY = "recovery"
    SIDEWAYS = "sideways"
    DEFENSIVE = "defensive"
    EMERGENCY = "emergency"


class RiskManager:
    """
    Acts as Layer 3: The Risk Budgeting Layer.

    Its sole responsibility is to analyze the market and portfolio state to
    produce a clear, actionable RiskBudget. It is completely independent of
    any alpha signals or strategy-specific logic.
    """

    def __init__(self, initial_capital: float, config_dict: Dict):
        """
        Constructor remains largely the same, loading risk parameters.
        """
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        get_dd_param = partial(get_param_with_override, config_dict, "drawdown_control")
        get_exposure_param = partial(
            get_param_with_override, config_dict, "exposure_management"
        )
        get_pos_mgmt_param = partial(
            get_param_with_override, config_dict, "position_management"
        )
        get_vol_param = partial(
            get_param_with_override, config_dict, "portfolio_volatility_control"
        )
        get_scaling_param = partial(
            get_param_with_override, config_dict, "continuous_risk_scaling"
        )
        get_sizing_param = partial(
            get_param_with_override, config_dict, "position_sizing"
        )

        # --- Drawdown Control Parameters ---
        self.max_drawdown = get_dd_param("max_drawdown_threshold", 0.25)
        self.emergency_drawdown = get_dd_param("emergency_drawdown_threshold", 0.30)
        self.recovery_threshold = get_dd_param("recovery_threshold", 0.90)
        self.cooldown_days = get_dd_param("cooldown_days", 3)
        self.drawdown_lookback_days = get_dd_param("drawdown_lookback_days", 252)

        # --- Portfolio Volatility Control Parameters ---
        self.volatility_control_enabled = get_vol_param("enabled", True)
        self.volatility_target = get_vol_param("portfolio_volatility_target", 0.15)
        self.volatility_lookback = get_vol_param("portfolio_volatility_lookback", 60)
        self.volatility_max_scaling = get_vol_param(
            "portfolio_volatility_max_scaling", 1.5
        )
        self.volatility_min_scaling = get_vol_param(
            "portfolio_volatility_min_scaling", 0.5
        )

        # --- Continuous Risk Scaling Parameters ---
        self.continuous_scaling_enabled = get_scaling_param("enabled", True)
        self.drawdown_weight = get_scaling_param("drawdown_weight", 0.50)
        self.market_regime_weight = get_scaling_param("market_regime_weight", 0.30)
        self.portfolio_volatility_weight = get_scaling_param(
            "portfolio_volatility_weight", 0.20
        )

        # --- Position Sizing Limits ---
        self.max_positions = get_sizing_param("max_positions", 15)
        self.min_positions = get_sizing_param("min_positions", 3)

        # --- State Initialization ---
        self.peak_value = initial_capital
        self.risk_budget_levels = config.get_trading_param(
            "RISK_PARAMS", "risk_budget_levels", default={}
        )
        self.current_level = RiskLevel.NORMAL
        self.current_drawdown = 0.0
        self.in_recovery_mode = False
        self.recovery_start_time = None

        # --- History Tracking ---
        max_lookback = max(self.volatility_lookback, self.drawdown_lookback_days) + 5
        self.portfolio_value_history = deque(maxlen=max_lookback)
        self.portfolio_value_history.append(initial_capital)

        # --- Leverage Limits ---
        limits_config = config.get_trading_param(
            "RISK_PARAMS", "portfolio_limits", default={}
        )
        self.max_leverage = config_dict.get(
            "max_leverage", limits_config.get("max_leverage", 1.0)
        )

        self.min_exposure_pct = get_exposure_param("min_exposure_pct", 0.30)
        self.max_exposure_pct = get_exposure_param("max_exposure_pct", 0.98)
        self.risk_score_history = deque(maxlen=5)

        self.min_positions = get_pos_mgmt_param("min_total_positions", 5)
        self.max_positions = get_pos_mgmt_param("max_total_positions", 20)
        self.logger.info(
            f"RiskManager initialized: max_dd={self.max_drawdown:.1%}, vol_target={self.volatility_target:.1%}"
            f"max_leverage={self.max_leverage:.2f}x, "
            f"positions=[{self.min_positions}-{self.max_positions}]"
        )

    def determine_risk_budget(
        self,
        portfolio_value: float,
        timestamp: datetime,
        market_state: MarketState,
    ) -> RiskBudget:

        self._update_drawdown(portfolio_value, timestamp)
        self.portfolio_value_history.append(portfolio_value)

        # Step 1: Calculate a continuous risk score from 0 (safest) to 1 (riskiest)
        raw_risk_score = self._calculate_continuous_risk_score(market_state)

        self.risk_score_history.append(raw_risk_score)
        smoothed_risk_score = np.mean(self.risk_score_history)

        risk_score = smoothed_risk_score

        # Step 2: Determine dynamic leverage based on the risk score
        target_exposure_pct = self.min_exposure_pct + risk_score * (
            self.max_exposure_pct - self.min_exposure_pct
        )

        # Step 3: Map the risk score to a target *gross* exposure
        target_gross_exposure_value = portfolio_value * target_exposure_pct

        # Step 4: Map the risk score to the number of positions
        min_pos = self.min_positions
        max_pos = self.max_positions
        target_positions = int(
            self.min_positions + risk_score * (self.max_positions - self.min_positions)
        )
        target_positions = np.clip(
            target_positions, self.min_positions, self.max_positions
        )

        # Step 5: Update the RiskBudget object
        final_budget = RiskBudget(
            target_portfolio_exposure=target_gross_exposure_value,
            max_position_count=target_positions,
        )

        self.logger.info(
            f"Risk Budget: Score={risk_score:.2f}, Target Exposure={target_exposure_pct:.1%}, "
            f"Target Value=${final_budget.target_portfolio_exposure:,.0f}, "
            f"TargetPos={final_budget.max_position_count}"
        )

        return final_budget

    def _update_drawdown(self, current_value: float, timestamp: datetime):
        """Update drawdown calculation."""
        # Get the history of portfolio values within the lookback window
        history_array = np.array(self.portfolio_value_history)

        # Find the peak value within that recent window
        rolling_peak = np.max(history_array)

        if rolling_peak > 0:
            self.current_drawdown = (rolling_peak - current_value) / rolling_peak
        else:
            self.current_drawdown = 0.0

    def _calculate_continuous_risk_score(self, market_state: MarketState) -> float:
        """
        Calculates a single risk score from 0 (max defense) to 1 (max offense).
        This score combines drawdown, market regime, and portfolio volatility.
        """
        # 1. Drawdown Component
        drawdown_score = 1.0 - (self.current_drawdown / self.max_drawdown)
        drawdown_score = np.clip(drawdown_score, 0, 1)

        # 2. Market Regime Component
        regime_scores = {
            MarketRegime.STRONG_BULL: 1.0,
            MarketRegime.BULL: 0.9,
            MarketRegime.NORMAL: 0.7,
            MarketRegime.RECOVERY: 0.6,
            MarketRegime.VOLATILE: 0.4,
            MarketRegime.BEAR: 0.2,
            MarketRegime.CRISIS: 0.0,
        }
        market_score = regime_scores.get(market_state.regime, 0.5)

        # 3. Portfolio Volatility Component
        # Use the inverse of the scaler we developed in Improvement 1
        volatility_scaler = self._calculate_volatility_scaler()
        # We can normalize it to a 0-1 range.
        volatility_score = (volatility_scaler - self.volatility_min_scaling) / (
            self.volatility_max_scaling - self.volatility_min_scaling
        )
        volatility_score = np.clip(volatility_score, 0, 1)

        # Combine with weights
        final_score = (
            (drawdown_score * 0.35) + (market_score * 0.45) + (volatility_score * 0.20)
        )

        return np.clip(final_score, 0, 1)

    def _check_recovery_complete(
        self, portfolio_value: float, timestamp: datetime
    ) -> bool:
        """Check if recovery conditions are met."""
        # Condition 1: Drawdown has recovered significantly
        recovered_drawdown = self.current_drawdown < self.max_drawdown * (
            1 - self.recovery_threshold
        )

        # Condition 2: Cooldown period has passed
        cooldown_passed = False
        if self.recovery_start_time:
            days_elapsed = (timestamp - self.recovery_start_time).days
            cooldown_passed = days_elapsed >= self.cooldown_days

        return recovered_drawdown and cooldown_passed

    def _calculate_volatility_scaler(self) -> float:
        # Check if we have enough historical data to calculate volatility
        if len(self.portfolio_value_history) < self.volatility_lookback:
            return 1.0  # Not enough data, do not scale

        # Calculate daily returns from the stored portfolio values
        values = np.array(self.portfolio_value_history)
        returns = (values[1:] - values[:-1]) / values[:-1]

        # Calculate annualized volatility over the lookback period
        realized_volatility = np.std(returns) * np.sqrt(252)

        # If volatility is zero (e.g., all cash), avoid division by zero
        if realized_volatility < 1e-6:
            return self.volatility_max_scaling  # If no risk, we can take max risk

        # The core scaling logic: (Target Vol / Realized Vol)
        scaler = self.volatility_target / realized_volatility

        # Clip the scaler to prevent extreme adjustments
        scaler = np.clip(
            scaler, self.volatility_min_scaling, self.volatility_max_scaling
        )

        self.logger.debug(
            f"Volatility Scaler: Realized Vol={realized_volatility:.2%}, "
            f"Target Vol={self.volatility_target:.2%}, Scaler={scaler:.2f}"
        )

        return scaler
