# strategies/risk/integrated_risk_manager.py

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy.alpha.market_detector import MarketRegime, MarketState


class RiskMode(Enum):
    """Risk management modes"""

    NORMAL = "normal"
    CAUTIOUS = "cautious"
    DEFENSIVE = "defensive"
    RISK_OFF = "risk_off"
    EMERGENCY = "emergency"


@dataclass
class RiskAssessment:
    """Risk assessment result"""

    mode: RiskMode
    position_limit: int
    position_size_multiplier: float
    allow_new_positions: bool
    force_exit_symbols: List[str]
    reduce_position_symbols: List[str]
    risk_score: float
    warnings: List[str]
    timestamp: datetime


class RiskManager:
    """
    Strategy-integrated risk management module
    Enhanced from backtester/risk_manager.py for internal strategy use
    """

    def __init__(self, initial_capital: float, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital

        # Core risk parameters
        self.max_drawdown_threshold = kwargs.get("max_drawdown_threshold", 0.20)
        self.emergency_drawdown = kwargs.get("emergency_drawdown", 0.30)
        self.drawdown_recovery_threshold = kwargs.get("recovery_threshold", 0.90)
        self.risk_off_cooldown_days = kwargs.get("risk_off_cooldown", 5)

        # Position limits
        self.base_max_positions = kwargs.get("max_positions", 8)
        self.min_positions = kwargs.get("min_positions", 2)
        self.max_single_position = kwargs.get("max_single_position", 0.35)
        self.max_sector_exposure = kwargs.get("max_sector_exposure", 0.40)

        # Volatility management
        self.target_portfolio_volatility = kwargs.get("target_volatility", 0.15)
        self.max_portfolio_volatility = kwargs.get("max_volatility", 0.25)
        self.volatility_lookback = kwargs.get("volatility_lookback", 20)

        # Risk scoring thresholds
        self.risk_score_thresholds = {
            "low": 0.2,
            "medium": 0.4,
            "high": 0.6,
            "extreme": 0.8,
        }

        # State tracking
        self.current_drawdown = 0.0
        self.peak_value = initial_capital
        self.current_mode = RiskMode.NORMAL
        self.risk_off_entry_time = None
        self.position_values = {}  # Track individual position values
        self.portfolio_volatility_history = []

        # Risk metrics history
        self.risk_history = []
        self.last_assessment = None

        self.is_initialized = False

    def assess_portfolio_risk(
        self,
        portfolio_value: float,
        positions: Dict[str, float],
        market_state: "MarketState",
        timestamp: datetime,
    ) -> RiskAssessment:
        """
        Comprehensive portfolio risk assessment

        Args:
            portfolio_value: Current total portfolio value
            positions: Dictionary of symbol -> position value
            market_state: Current market state from MarketDetector
            timestamp: Current timestamp

        Returns:
            RiskAssessment with recommendations
        """
        if not self.is_initialized:
            self.is_initialized = True
            self.peak_value = portfolio_value
            self.current_mode = RiskMode.NORMAL

        self.update_drawdown_status(portfolio_value, timestamp)

        # Calculate risk components
        drawdown_risk = self._assess_drawdown_risk()
        volatility_risk = self._assess_volatility_risk(positions)
        concentration_risk = self._assess_concentration_risk(positions, portfolio_value)
        market_risk = self._assess_market_risk(market_state)

        # Calculate composite risk score
        risk_score = self._calculate_composite_risk_score(
            drawdown_risk, volatility_risk, concentration_risk, market_risk
        )

        # Determine risk mode
        risk_mode = self._determine_risk_mode(risk_score, market_state)

        # Generate risk-based recommendations
        recommendations = self._generate_recommendations(
            risk_mode, risk_score, positions, portfolio_value, market_state
        )

        # Create assessment
        assessment = RiskAssessment(
            mode=risk_mode,
            position_limit=recommendations["position_limit"],
            position_size_multiplier=recommendations["size_multiplier"],
            allow_new_positions=recommendations["allow_new"],
            force_exit_symbols=recommendations["force_exits"],
            reduce_position_symbols=recommendations["reduce_positions"],
            risk_score=risk_score,
            warnings=recommendations["warnings"],
            timestamp=timestamp,
        )

        # Update history
        self.last_assessment = assessment
        self._update_risk_history(assessment)

        return assessment

    def update_drawdown_status(self, current_value: float, timestamp: datetime):
        """Update drawdown tracking"""
        if self.peak_value == 0 or self.peak_value < self.initial_capital * 0.5:
            self.peak_value = max(current_value, self.initial_capital)
        # Check cooldown period
        if self.risk_off_entry_time:
            days_since = (timestamp - self.risk_off_entry_time).days
            if days_since > self.risk_off_cooldown_days:
                self.logger.info("Risk-off cooldown period ended")
                self.risk_off_entry_time = None

        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
            if self.current_mode == RiskMode.RISK_OFF:
                self.logger.info("New peak reached, exiting risk-off mode")
                self.current_mode = RiskMode.NORMAL
                self.risk_off_entry_time = None

        # Calculate drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

    def _assess_drawdown_risk(self) -> float:
        """Assess risk from drawdown perspective"""
        if self.current_drawdown >= self.emergency_drawdown:
            return 1.0  # Maximum risk
        elif self.current_drawdown >= self.max_drawdown_threshold:
            return 0.8
        elif self.current_drawdown >= self.max_drawdown_threshold * 0.75:
            return 0.6
        elif self.current_drawdown >= self.max_drawdown_threshold * 0.5:
            return 0.4
        else:
            # Scale linearly for smaller drawdowns
            return (self.current_drawdown / self.max_drawdown_threshold) * 0.4

    def _assess_volatility_risk(self, positions: Dict[str, float]) -> float:
        """Assess portfolio volatility risk"""
        if not positions:
            return 0.0

        # This would need actual price history in production
        # For now, use a simplified calculation
        position_count = len(positions)
        concentration = (
            max(positions.values()) / sum(positions.values()) if positions else 0
        )

        # Higher concentration and fewer positions = higher volatility risk
        vol_risk = (
            concentration * 0.5
            + (1 - min(position_count / self.base_max_positions, 1)) * 0.5
        )

        return vol_risk

    def _assess_concentration_risk(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> float:
        """Assess position concentration risk"""
        if not positions or portfolio_value <= 0:
            return 0.0

        risks = []

        # Check individual position concentration
        for symbol, value in positions.items():
            position_weight = value / portfolio_value
            if position_weight > self.max_single_position:
                risks.append(1.0)
            elif position_weight > self.max_single_position * 0.8:
                risks.append(0.7)
            elif position_weight > self.max_single_position * 0.6:
                risks.append(0.4)

        # Check overall concentration (HHI)
        weights = [v / portfolio_value for v in positions.values()]
        hhi = sum(w**2 for w in weights)

        # HHI ranges from 1/n (equal weight) to 1 (single position)
        min_hhi = 1 / max(len(positions), 1)
        normalized_hhi = (hhi - min_hhi) / (1 - min_hhi) if min_hhi < 1 else 0
        risks.append(normalized_hhi)

        return np.mean(risks) if risks else 0.0

    def _assess_market_risk(self, market_state: "MarketState") -> float:
        """Assess market environment risk"""
        # Map market regimes to risk levels
        regime_risk = {
            "crisis": 1.0,
            "bear": 0.7,
            "volatile": 0.6,
            "recovery": 0.4,
            "normal": 0.3,
            "bull": 0.2,
            "strong_bull": 0.1,
        }

        base_risk = regime_risk.get(market_state.regime.value, 0.5)

        # Adjust for market risk level
        risk_adjustment = {"extreme": 1.2, "high": 1.1, "medium": 1.0, "low": 0.9}

        adjusted_risk = base_risk * risk_adjustment.get(market_state.risk_level, 1.0)

        return min(adjusted_risk, 1.0)

    def _calculate_composite_risk_score(
        self, drawdown: float, volatility: float, concentration: float, market: float
    ) -> float:
        """Calculate weighted composite risk score"""
        weights = {
            "drawdown": 0.35,
            "volatility": 0.20,
            "concentration": 0.20,
            "market": 0.25,
        }

        composite = (
            drawdown * weights["drawdown"]
            + volatility * weights["volatility"]
            + concentration * weights["concentration"]
            + market * weights["market"]
        )

        return min(composite, 1.0)

    def _determine_risk_mode(
        self, risk_score: float, market_state: "MarketState"
    ) -> RiskMode:
        """Determine appropriate risk management mode"""
        # Emergency conditions
        if self.risk_off_entry_time:
            # During cooldown, be more conservative
            min_mode = RiskMode.CAUTIOUS
        else:
            min_mode = RiskMode.NORMAL

            # Determine mode based on conditions
        determined_mode = RiskMode.NORMAL  # Default

        # Emergency conditions
        if self.current_drawdown >= self.emergency_drawdown:
            determined_mode = RiskMode.EMERGENCY
            # Record entry time for risk-off
            if not self.risk_off_entry_time:
                self.risk_off_entry_time = datetime.now()

        # Risk-off conditions
        elif (
            self.current_drawdown >= self.max_drawdown_threshold
            or risk_score > self.risk_score_thresholds["extreme"]
        ):
            determined_mode = RiskMode.RISK_OFF
            if not self.risk_off_entry_time:
                self.risk_off_entry_time = datetime.now()

        elif risk_score > self.risk_score_thresholds[
            "high"
        ] or market_state.regime.value in ["crisis", "bear"]:
            determined_mode = RiskMode.DEFENSIVE

        # Cautious conditions
        elif (
            risk_score > self.risk_score_thresholds["medium"]
            or market_state.regime.value == "volatile"
        ):
            determined_mode = RiskMode.CAUTIOUS

        # Apply minimum mode from cooldown (don't allow going below CAUTIOUS during cooldown)
        if min_mode == RiskMode.CAUTIOUS and determined_mode == RiskMode.NORMAL:
            return RiskMode.CAUTIOUS

        return determined_mode

    def _generate_recommendations(
        self,
        mode: RiskMode,
        risk_score: float,
        positions: Dict[str, float],
        portfolio_value: float,
        market_state: "MarketState",
    ) -> Dict:
        """Generate specific risk management recommendations"""
        recommendations = {
            "position_limit": self.base_max_positions,
            "size_multiplier": 1.0,
            "allow_new": True,
            "force_exits": [],
            "reduce_positions": [],
            "warnings": [],
        }

        # Mode-specific adjustments
        if mode == RiskMode.EMERGENCY:
            recommendations["position_limit"] = 0.1
            recommendations["size_multiplier"] = 0.1
            recommendations["allow_new"] = False
            recommendations["force_exits"] = list(positions.keys())
            recommendations["warnings"].append(
                "EMERGENCY: Exit all positions immediately"
            )

        elif mode == RiskMode.RISK_OFF:
            recommendations["position_limit"] = self.min_positions
            recommendations["size_multiplier"] = 0.3
            recommendations["allow_new"] = False
            # Exit weakest positions
            if positions:
                sorted_positions = sorted(positions.items(), key=lambda x: x[1])
                exit_count = max(len(positions) - self.min_positions, 0)
                recommendations["force_exits"] = [
                    s for s, _ in sorted_positions[:exit_count]
                ]
            recommendations["warnings"].append(
                f"RISK-OFF: Drawdown {self.current_drawdown:.1%}"
            )

        elif mode == RiskMode.DEFENSIVE:
            recommendations["position_limit"] = max(self.min_positions + 1, 3)
            recommendations["size_multiplier"] = 0.6
            recommendations["allow_new"] = risk_score < 0.7
            # Reduce large positions
            for symbol, value in positions.items():
                if value / portfolio_value > self.max_single_position * 0.8:
                    recommendations["reduce_positions"].append(symbol)
            recommendations["warnings"].append("DEFENSIVE: High risk environment")

        elif mode == RiskMode.CAUTIOUS:
            recommendations["position_limit"] = self.base_max_positions - 2
            recommendations["size_multiplier"] = 0.8
            recommendations["allow_new"] = True
            recommendations["warnings"].append("CAUTIOUS: Elevated risk levels")

        # Check position concentration regardless of mode
        for symbol, value in positions.items():
            weight = value / portfolio_value if portfolio_value > 0 else 0
            if weight > self.max_single_position:
                if symbol not in recommendations["reduce_positions"]:
                    recommendations["reduce_positions"].append(symbol)
                recommendations["warnings"].append(
                    f"{symbol} exceeds max position size ({weight:.1%})"
                )

        return recommendations

    def _update_risk_history(self, assessment: RiskAssessment):
        """Update risk assessment history"""
        self.risk_history.append(
            {
                "timestamp": assessment.timestamp,
                "mode": assessment.mode.value,
                "risk_score": assessment.risk_score,
                "drawdown": self.current_drawdown,
                "position_limit": assessment.position_limit,
            }
        )

        # Keep only recent history
        if len(self.risk_history) > 100:
            self.risk_history = self.risk_history[-50:]

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics for monitoring"""
        return {
            "current_mode": self.current_mode.value,
            "current_drawdown": self.current_drawdown,
            "peak_value": self.peak_value,
            "risk_score": (
                self.last_assessment.risk_score if self.last_assessment else 0
            ),
            "position_limit": (
                self.last_assessment.position_limit
                if self.last_assessment
                else self.base_max_positions
            ),
            "allow_new_positions": (
                self.last_assessment.allow_new_positions
                if self.last_assessment
                else True
            ),
            "warnings": self.last_assessment.warnings if self.last_assessment else [],
        }

    def should_emergency_exit(self) -> bool:
        """Check if emergency exit conditions are met"""
        return (
            self.current_mode == RiskMode.EMERGENCY
            or self.current_drawdown >= self.emergency_drawdown
        )
