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
    """ """
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    RECOVERY = "recovery"
    SIDEWAYS = "sideways"
    DEFENSIVE = "defensive"
    EMERGENCY = "emergency"


class RiskManager:
    """ """
    def __init__(self, initial_capital: float, config_dict: Dict):
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
        get_cvar_param = partial(get_param_with_override, config_dict, "cvar_control")

        self.max_drawdown = get_dd_param("max_drawdown_threshold", 0.22)
        self.emergency_drawdown = get_dd_param("emergency_drawdown_threshold", 0.28)
        self.recovery_threshold = get_dd_param("recovery_threshold", 0.90)
        self.cooldown_days = get_dd_param("cooldown_days", 5)
        self.drawdown_lookback_days = get_dd_param("drawdown_lookback_days", 252)

        self.cvar_control_enabled = get_cvar_param("enabled", True)
        self.cvar_target = get_cvar_param("cvar_target", 0.02)
        self.cvar_lookback = get_cvar_param("cvar_lookback", 60)
        self.cvar_confidence_level = get_cvar_param("cvar_confidence_level", 0.95)
        self.cvar_min_scaling = get_cvar_param("cvar_min_scaling", 0.3)
        self.cvar_max_scaling = get_cvar_param("cvar_max_scaling", 1.2)

        self.volatility_control_enabled = get_vol_param("enabled", True)
        self.volatility_target = get_vol_param("portfolio_volatility_target", 0.14)
        self.volatility_lookback = get_vol_param("portfolio_volatility_lookback", 50)
        self.volatility_max_scaling = get_vol_param(
            "portfolio_volatility_max_scaling", 1.4
        )
        self.volatility_min_scaling = get_vol_param(
            "portfolio_volatility_min_scaling", 0.45
        )

        self.continuous_scaling_enabled = get_scaling_param("enabled", True)
        self.drawdown_weight = get_scaling_param("drawdown_weight", 0.50)
        self.market_regime_weight = get_scaling_param("market_regime_weight", 0.30)
        self.portfolio_volatility_weight = get_scaling_param(
            "portfolio_volatility_weight", 0.20
        )

        self.max_positions = get_sizing_param("max_positions", 15)
        self.min_positions = get_sizing_param("min_positions", 3)

        self.peak_value = initial_capital
        self.risk_budget_levels = config.get_trading_param(
            "RISK_PARAMS", "risk_budget_levels", default={}
        )
        self.current_level = RiskLevel.NORMAL
        self.current_drawdown = 0.0
        self.in_recovery_mode = False
        self.recovery_start_time = None

        max_lookback = (
            max(
                self.volatility_lookback,
                self.drawdown_lookback_days,
                self.cvar_lookback,
            )
            + 5
        )
        self.portfolio_value_history = deque(maxlen=max_lookback)
        self.portfolio_value_history.append(initial_capital)

        limits_config = config.get_trading_param(
            "RISK_PARAMS", "portfolio_limits", default={}
        )
        self.max_leverage = config_dict.get(
            "max_leverage", limits_config.get("max_leverage", 1.0)
        )

        self.min_exposure_pct = get_exposure_param("min_exposure_pct", 0.25)
        self.max_exposure_pct = get_exposure_param("max_exposure_pct", 0.95)
        self.risk_score_history = deque(maxlen=5)

        self.min_positions = get_pos_mgmt_param("min_total_positions", 6)
        self.max_positions = get_pos_mgmt_param("max_total_positions", 18)
        self.logger.info(
            f"RiskManager initialized: max_dd={self.max_drawdown:.1%}, vol_target={self.volatility_target:.1%}, "
            f"cvar_target={self.cvar_target:.1%}"
        )

    def determine_risk_budget(
        self,
        portfolio_value: float,
        timestamp: datetime,
        market_state: MarketState,
    ) -> RiskBudget:
        """

        Args:
          portfolio_value: float: 
          timestamp: datetime: 
          market_state: MarketState: 

        Returns:

        """

        self._update_drawdown(portfolio_value, timestamp)
        self.portfolio_value_history.append(portfolio_value)

        # Step 1: Calculate a continuous risk score (base reactive score)
        raw_risk_score = self._calculate_continuous_risk_score(market_state)
        smoothed_risk_score = np.mean(list(self.risk_score_history) + [raw_risk_score])
        risk_score = smoothed_risk_score

        # Step 2: Determine base target exposure from the reactive score
        base_target_exposure_pct = self.min_exposure_pct + risk_score * (
            self.max_exposure_pct - self.min_exposure_pct
        )

        # Step 3: Get dynamic risk overlays
        cvar_scaler = self._calculate_cvar_scaler()
        # MODIFIED: Replaced similarity_scaler with transition_adjustment for forward-looking risk management.
        transition_scaler = market_state.transition_adjustment

        # Step 4: Apply all scalers to the base exposure
        target_exposure_pct = np.clip(
            base_target_exposure_pct * cvar_scaler * transition_scaler,
            self.min_exposure_pct,
            self.max_exposure_pct,
        )

        target_gross_exposure_value = portfolio_value * target_exposure_pct

        target_positions = int(
            self.min_positions + risk_score * (self.max_positions - self.min_positions)
        )
        target_positions = np.clip(
            target_positions, self.min_positions, self.max_positions
        )

        final_budget = RiskBudget(
            target_portfolio_exposure=target_gross_exposure_value,
            max_position_count=target_positions,
        )

        self.logger.info(
            f"Risk Budget: Score={risk_score:.2f}, CVaR_Scale={cvar_scaler:.2f}, "
            f"Transition_Scale={transition_scaler:.2f}, Final Exp={target_exposure_pct:.1%}, "
            f"Value=${final_budget.target_portfolio_exposure:,.0f}, "
            f"Pos={final_budget.max_position_count}"
        )

        return final_budget

    def _update_drawdown(self, current_value: float, timestamp: datetime):
        """

        Args:
          current_value: float: 
          timestamp: datetime: 

        Returns:

        """
        history_array = np.array(self.portfolio_value_history)
        rolling_peak = np.max(history_array)
        if rolling_peak > 0:
            self.current_drawdown = (rolling_peak - current_value) / rolling_peak
        else:
            self.current_drawdown = 0.0

    def _calculate_cvar_scaler(self) -> float:
        """ """
        if (
            not self.cvar_control_enabled
            or len(self.portfolio_value_history) < self.cvar_lookback
        ):
            return 1.0

        values = np.array(list(self.portfolio_value_history)[-self.cvar_lookback :])
        returns = (values[1:] - values[:-1]) / values[:-1]

        if len(returns) < 10 or np.std(returns) == 0:
            return 1.0

        var_threshold = np.percentile(returns, (1 - self.cvar_confidence_level) * 100)
        tail_losses = returns[returns <= var_threshold]

        if len(tail_losses) == 0:
            return self.cvar_max_scaling

        realized_cvar_abs = abs(np.mean(tail_losses))

        if realized_cvar_abs < 1e-6:
            return self.cvar_max_scaling

        scaler = self.cvar_target / realized_cvar_abs
        scaler = np.clip(scaler, self.cvar_min_scaling, self.cvar_max_scaling)

        self.logger.debug(
            f"CVaR Scaler: Realized={realized_cvar_abs:.2%}, Target={self.cvar_target:.2%}, Scaler={scaler:.2f}"
        )
        return scaler

    def _calculate_continuous_risk_score(self, market_state: MarketState) -> float:
        """

        Args:
          market_state: MarketState: 

        Returns:

        """
        drawdown_score = np.clip(
            1.0 - (self.current_drawdown / self.max_drawdown), 0, 1
        )

        regime_scores = {
            MarketRegime.STRONG_BULL: 1.0,
            MarketRegime.BULL: 0.90,
            MarketRegime.NORMAL: 0.70,
            MarketRegime.RECOVERY: 0.55,
            MarketRegime.VOLATILE: 0.35,
            MarketRegime.BEAR: 0.20,
            MarketRegime.CRISIS: 0.10,
        }
        market_score = regime_scores.get(market_state.regime, 0.5)

        volatility_scaler = self._calculate_volatility_scaler()
        volatility_score = np.clip(
            (volatility_scaler - self.volatility_min_scaling)
            / (self.volatility_max_scaling - self.volatility_min_scaling),
            0,
            1,
        )

        final_score = (
            (drawdown_score * 0.40) + (market_score * 0.42) + (volatility_score * 0.18)
        )
        return np.clip(final_score, 0, 1)

    def _check_recovery_complete(
        self, portfolio_value: float, timestamp: datetime
    ) -> bool:
        """

        Args:
          portfolio_value: float: 
          timestamp: datetime: 

        Returns:

        """
        recovered_drawdown = self.current_drawdown < self.max_drawdown * (
            1 - self.recovery_threshold
        )
        cooldown_passed = False
        if self.recovery_start_time:
            cooldown_passed = (
                timestamp - self.recovery_start_time
            ).days >= self.cooldown_days
        return recovered_drawdown and cooldown_passed

    def _calculate_volatility_scaler(self) -> float:
        """ """
        if len(self.portfolio_value_history) < self.volatility_lookback:
            return 1.0

        values = np.array(self.portfolio_value_history)
        returns = (values[1:] - values[:-1]) / values[:-1]
        realized_volatility = np.std(returns) * np.sqrt(252)

        if realized_volatility < 1e-6:
            return self.volatility_max_scaling

        scaler = self.volatility_target / realized_volatility
        scaler = np.clip(
            scaler, self.volatility_min_scaling, self.volatility_max_scaling
        )

        self.logger.debug(
            f"Volatility Scaler: Realized Vol={realized_volatility:.2%}, "
            f"Target Vol={self.volatility_target:.2%}, Scaler={scaler:.2f}"
        )
        return scaler
