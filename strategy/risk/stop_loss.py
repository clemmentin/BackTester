#
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class StopType(Enum):
    """Stop loss types"""

    FIXED = "fixed"
    TRAILING = "trailing"
    ATR = "atr"
    TIME = "time"
    VOLATILITY = "volatility"
    PROFIT_PROTECTION = "profit_protection"


@dataclass
class PositionStop:
    """Stop loss information for a position"""

    symbol: str
    entry_price: float
    entry_date: datetime
    current_price: float
    stop_price: float
    stop_type: StopType
    highest_price: float
    lowest_price: float
    position_pnl_pct: float
    days_held: int
    trailing_activated: bool
    profit_protection_level: float


@dataclass
class StopAction:
    """Stop loss action to take"""

    symbol: str
    action: str  # 'exit', 'reduce', 'tighten', 'none'
    reason: str
    stop_price: float
    current_price: float
    pnl_pct: float
    urgency: str  # 'immediate', 'normal', 'monitor'


class StopManager:
    """
    Strategy-integrated stop loss management
    Enhanced from backtester/stop_loss_manager.py
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.enabled = kwargs.get("enabled", True)

        # Core stop parameters
        self.fixed_stop_pct = kwargs.get("fixed_stop_loss", 0.08)
        self.trailing_activation = kwargs.get("trailing_activation", 0.10)
        self.trailing_distance = kwargs.get("trailing_distance", 0.05)
        self.time_stop_days = kwargs.get("time_stop_days", 60)

        # ATR-based stops
        self.use_atr_stops = kwargs.get("use_atr_stops", True)
        self.atr_multiplier = kwargs.get("atr_multiplier", 2.0)
        self.atr_period = kwargs.get("atr_period", 14)

        # Profit protection levels
        self.profit_levels = kwargs.get(
            "profit_levels",
            {
                "level1": {"profit": 0.10, "protect": 0.05},
                "level2": {"profit": 0.20, "protect": 0.12},
                "level3": {"profit": 0.30, "protect": 0.20},
            },
        )

        # Volatility adjustment
        self.volatility_adjusted = kwargs.get("volatility_adjusted", True)
        self.vol_adjustment_factor = kwargs.get("vol_factor", 1.5)

        # Market regime adjustments
        self.regime_multipliers = kwargs.get(
            "regime_multipliers",
            {
                "crisis": 0.5,  # Tighter stops in crisis
                "bear": 0.7,
                "volatile": 0.8,
                "normal": 1.0,
                "bull": 1.2,
                "strong_bull": 1.3,
            },
        )

        # Position tracking
        self.position_stops = {}  # symbol -> PositionStop
        self.stop_history = []
        self.triggered_stops = {}  # Track recently triggered stops

        # Performance metrics
        self.stop_effectiveness = {
            "total_stops": 0,
            "profitable_stops": 0,
            "loss_prevented": 0.0,
            "premature_stops": 0,
        }

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        entry_date: datetime,
        initial_stop: Optional[float] = None,
    ):
        """Add a new position to stop tracking"""
        if not self.enabled:
            return

        # Calculate initial stop price
        if initial_stop is None:
            initial_stop = entry_price * (1 - self.fixed_stop_pct)

        self.position_stops[symbol] = PositionStop(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            current_price=entry_price,
            stop_price=initial_stop,
            stop_type=StopType.FIXED,
            highest_price=entry_price,
            lowest_price=entry_price,
            position_pnl_pct=0.0,
            days_held=0,
            trailing_activated=False,
            profit_protection_level=0.0,
        )

        self.logger.info(
            f"Added stop tracking for {symbol}: entry=${entry_price:.2f}, stop=${initial_stop:.2f}"
        )

    def update_stops(
        self,
        market_data: Dict[str, Dict],
        market_state: "MarketState",
        timestamp: datetime,
    ) -> List[StopAction]:
        """
        Update all position stops and check for triggers

        Args:
            market_data: Dict of symbol -> {'price': float, 'atr': float, 'volatility': float}
            market_state: Current market state
            timestamp: Current timestamp

        Returns:
            List of StopAction recommendations
        """
        if not self.enabled or not self.position_stops:
            return []

        stop_actions = []

        for symbol, position_stop in list(self.position_stops.items()):
            if symbol not in market_data:
                continue

            # Update position data
            current_data = market_data[symbol]
            current_price = current_data["price"]

            # Update position stop info
            self._update_position_data(position_stop, current_price, timestamp)

            # Calculate dynamic stop levels
            new_stop = self._calculate_dynamic_stop(
                position_stop, current_data, market_state
            )

            # Update stop price (only move up for longs)
            if new_stop > position_stop.stop_price:
                position_stop.stop_price = new_stop
                position_stop.stop_type = self._determine_stop_type(position_stop)

            # Check if stop is triggered
            action = self._check_stop_trigger(position_stop, current_price, timestamp)
            if action.action != "none":
                stop_actions.append(action)

        return stop_actions

    def _update_position_data(
        self, position: PositionStop, current_price: float, timestamp: datetime
    ):
        """Update position tracking data"""
        position.current_price = current_price
        position.highest_price = max(position.highest_price, current_price)
        position.lowest_price = min(position.lowest_price, current_price)
        position.position_pnl_pct = (current_price / position.entry_price) - 1
        position.days_held = (timestamp - position.entry_date).days

        # Check for trailing stop activation
        if not position.trailing_activated:
            if position.position_pnl_pct >= self.trailing_activation:
                position.trailing_activated = True
                self.logger.info(
                    f"{position.symbol}: Trailing stop activated at {position.position_pnl_pct:.1%} profit"
                )

    def _calculate_dynamic_stop(
        self, position: PositionStop, market_data: Dict, market_state: "MarketState"
    ) -> float:
        """Calculate dynamic stop price based on multiple factors"""
        stops = []

        # 1. Fixed stop
        fixed_stop = position.entry_price * (1 - self.fixed_stop_pct)
        stops.append(fixed_stop)

        # 2. Trailing stop (if activated)
        if position.trailing_activated:
            trailing_stop = position.highest_price * (1 - self.trailing_distance)
            stops.append(trailing_stop)

        # 3. ATR-based stop
        if self.use_atr_stops and "atr" in market_data:
            atr = market_data["atr"]
            if atr > 0:
                atr_stop = position.current_price - (atr * self.atr_multiplier)
                # Adjust for market regime
                regime = market_state.regime.value if market_state else "normal"
                regime_mult = self.regime_multipliers.get(regime, 1.0)
                atr_stop = position.current_price - (
                    atr * self.atr_multiplier / regime_mult
                )
                stops.append(atr_stop)

        # 4. Profit protection stops
        for level_name, level_config in self.profit_levels.items():
            if position.position_pnl_pct >= level_config["profit"]:
                protect_stop = position.entry_price * (1 + level_config["protect"])
                stops.append(protect_stop)
                position.profit_protection_level = max(
                    position.profit_protection_level, level_config["protect"]
                )

        # 5. Volatility-adjusted stop
        if self.volatility_adjusted and "volatility" in market_data:
            vol = market_data.get("volatility", 0.15)
            if vol > 0:
                # Widen stops in high volatility
                vol_multiplier = 1 + (vol - 0.15) * self.vol_adjustment_factor
                vol_multiplier = np.clip(vol_multiplier, 0.5, 2.0)
                vol_stop = position.entry_price * (
                    1 - self.fixed_stop_pct * vol_multiplier
                )
                stops.append(vol_stop)

        # Return the highest stop (tightest)
        return max(stops) if stops else position.stop_price

    def _determine_stop_type(self, position: PositionStop) -> StopType:
        """Determine the current stop type"""
        if position.profit_protection_level > 0:
            return StopType.PROFIT_PROTECTION
        elif position.trailing_activated:
            return StopType.TRAILING
        elif position.days_held > self.time_stop_days * 0.8:
            return StopType.TIME
        else:
            return StopType.FIXED

    def _check_stop_trigger(
        self, position: PositionStop, current_price: float, timestamp: datetime
    ) -> StopAction:
        """Check if stop is triggered and determine action"""

        # Main stop trigger
        if current_price <= position.stop_price:
            reason = (
                f"{position.stop_type.value} stop hit at ${position.stop_price:.2f}"
            )
            urgency = "immediate"

            # Record stop effectiveness
            self._record_stop_effectiveness(position)

            return StopAction(
                symbol=position.symbol,
                action="exit",
                reason=reason,
                stop_price=position.stop_price,
                current_price=current_price,
                pnl_pct=position.position_pnl_pct,
                urgency=urgency,
            )

        # Time-based stop for underperforming positions
        if (
            position.days_held > self.time_stop_days
            and position.position_pnl_pct < 0.05
        ):  # Less than 5% gain
            return StopAction(
                symbol=position.symbol,
                action="exit",
                reason=f"Time stop: held {position.days_held} days with poor performance",
                stop_price=position.stop_price,
                current_price=current_price,
                pnl_pct=position.position_pnl_pct,
                urgency="normal",
            )

        # Partial exit for extreme profits
        if position.position_pnl_pct > 0.50:  # 50% profit
            return StopAction(
                symbol=position.symbol,
                action="reduce",
                reason=f"Profit taking at {position.position_pnl_pct:.1%} gain",
                stop_price=position.stop_price,
                current_price=current_price,
                pnl_pct=position.position_pnl_pct,
                urgency="normal",
            )

        # Monitor positions approaching stop
        stop_distance = (current_price - position.stop_price) / current_price
        if stop_distance < 0.02:  # Within 2% of stop
            return StopAction(
                symbol=position.symbol,
                action="none",
                reason=f"Monitoring: close to stop (distance: {stop_distance:.1%})",
                stop_price=position.stop_price,
                current_price=current_price,
                pnl_pct=position.position_pnl_pct,
                urgency="monitor",
            )

        return StopAction(
            symbol=position.symbol,
            action="none",
            reason="",
            stop_price=position.stop_price,
            current_price=current_price,
            pnl_pct=position.position_pnl_pct,
            urgency="normal",
        )

    def remove_position(self, symbol: str):
        """Remove position from stop tracking"""
        if symbol in self.position_stops:
            position = self.position_stops[symbol]

            # Record final statistics
            self.stop_history.append(
                {
                    "symbol": symbol,
                    "entry_date": position.entry_date,
                    "exit_date": datetime.now(),
                    "entry_price": position.entry_price,
                    "exit_price": position.current_price,
                    "pnl_pct": position.position_pnl_pct,
                    "stop_type": position.stop_type.value,
                    "days_held": position.days_held,
                }
            )

            del self.position_stops[symbol]
            self.logger.info(f"Removed stop tracking for {symbol}")

    def _record_stop_effectiveness(self, position: PositionStop):
        """Record stop loss effectiveness metrics"""
        self.stop_effectiveness["total_stops"] += 1

        if position.position_pnl_pct > 0:
            self.stop_effectiveness["profitable_stops"] += 1
        else:
            # Calculate loss prevented (could have been worse)
            potential_further_loss = (
                position.stop_price * 0.1
            )  # Assume 10% further decline
            self.stop_effectiveness["loss_prevented"] += abs(potential_further_loss)

        # Check if stop was premature (price recovered quickly)
        # This would need future price data in production
        # For now, just track the metric

    def get_stop_summary(self) -> Dict[str, Any]:
        """Get summary of all current stops"""
        if not self.position_stops:
            return {"active_stops": 0}

        summary = {
            "active_stops": len(self.position_stops),
            "trailing_activated": sum(
                1 for p in self.position_stops.values() if p.trailing_activated
            ),
            "profit_protected": sum(
                1 for p in self.position_stops.values() if p.profit_protection_level > 0
            ),
            "avg_pnl": np.mean(
                [p.position_pnl_pct for p in self.position_stops.values()]
            ),
            "at_risk": [],  # Positions close to stop
        }

        # Identify at-risk positions
        for symbol, position in self.position_stops.items():
            stop_distance = (
                position.current_price - position.stop_price
            ) / position.current_price
            if stop_distance < 0.03:  # Within 3% of stop
                summary["at_risk"].append(
                    {
                        "symbol": symbol,
                        "distance": stop_distance,
                        "pnl": position.position_pnl_pct,
                    }
                )

        return summary

    def get_effectiveness_metrics(self) -> Dict[str, Any]:
        """Get stop loss effectiveness metrics"""
        if self.stop_effectiveness["total_stops"] == 0:
            return {"message": "No stops triggered yet"}

        return {
            "total_stops": self.stop_effectiveness["total_stops"],
            "profitable_stops": self.stop_effectiveness["profitable_stops"],
            "win_rate": self.stop_effectiveness["profitable_stops"]
            / self.stop_effectiveness["total_stops"],
            "avg_loss_prevented": self.stop_effectiveness["loss_prevented"]
            / self.stop_effectiveness["total_stops"],
            "recent_stops": self.stop_history[-10:] if self.stop_history else [],
        }

    def adjust_stops_for_volatility(self, current_volatility: float):
        """Globally adjust stops based on market volatility"""
        if not self.volatility_adjusted:
            return

        # Calculate adjustment factor
        base_volatility = 0.15  # 15% annual volatility as baseline
        vol_ratio = current_volatility / base_volatility

        # Adjust all stops
        for position in self.position_stops.values():
            if vol_ratio > 1.5:  # High volatility
                # Widen stops to avoid whipsaws
                adjustment = 1.2
            elif vol_ratio < 0.7:  # Low volatility
                # Tighten stops
                adjustment = 0.9
            else:
                adjustment = 1.0

            # Apply adjustment to trailing distance
            if position.trailing_activated:
                adjusted_distance = self.trailing_distance * adjustment
                new_stop = position.highest_price * (1 - adjusted_distance)
                position.stop_price = max(position.stop_price, new_stop)
