# strategies/risk/position_tracker.py

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class Position:
    """Detailed position information"""

    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float  # Position weight in portfolio
    days_held: int
    sector: str = None
    asset_class: str = None  # 'risk_on' or 'risk_off'

    def update_price(self, new_price: float, portfolio_value: float):
        """Update position with new price"""
        self.current_price = new_price
        self.current_value = self.quantity * new_price
        self.unrealized_pnl = self.current_value - (self.quantity * self.entry_price)
        self.unrealized_pnl_pct = (
            (new_price / self.entry_price - 1) if self.entry_price > 0 else 0
        )
        self.weight = self.current_value / portfolio_value if portfolio_value > 0 else 0


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot"""

    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    num_positions: int
    position_weights: Dict[str, float]
    sector_weights: Dict[str, float]
    asset_class_weights: Dict[str, float]
    largest_position: str
    smallest_position: str
    total_unrealized_pnl: float
    average_position_age: float


class PositionTracker:
    """
    Advanced position tracking for strategy internal use
    Provides detailed position analytics and portfolio composition tracking
    """

    def __init__(self, initial_capital: float, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital

        # Position storage
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital

        # Classification
        self.risk_on_symbols = set(kwargs.get("risk_on_symbols", []))
        self.risk_off_symbols = set(kwargs.get("risk_off_symbols", []))
        self.sector_map = kwargs.get("sector_map", {})

        # Tracking limits
        self.max_positions = kwargs.get("max_positions", 10)
        self.max_sector_exposure = kwargs.get("max_sector_exposure", 0.40)
        self.max_single_position = kwargs.get("max_single_position", 0.25)

        # Historical tracking
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.trade_history = []
        self.closed_positions = []

        # Performance tracking
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def add_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
    ) -> bool:
        """
        Add or increase a position

        Returns:
            bool: Success status
        """
        if quantity <= 0:
            self.logger.error(f"Invalid quantity {quantity} for {symbol}")
            return False

        # Check position limits
        if symbol not in self.positions and len(self.positions) >= self.max_positions:
            self.logger.warning(
                f"Cannot add {symbol}: max positions ({self.max_positions}) reached"
            )
            return False

        # Check cash availability
        total_cost = quantity * price + commission
        if total_cost > self.cash:
            self.logger.warning(
                f"Insufficient cash for {symbol}: need ${total_cost:.2f}, have ${self.cash:.2f}"
            )
            return False

        # Add or update position
        if symbol in self.positions:
            # Average up/down existing position
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            avg_price = (
                (existing.quantity * existing.entry_price) + (quantity * price)
            ) / total_quantity

            existing.quantity = total_quantity
            existing.entry_price = avg_price
            self.logger.info(
                f"Increased {symbol} position to {total_quantity} @ avg ${avg_price:.2f}"
            )
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_date=timestamp,
                current_price=price,
                current_value=quantity * price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                weight=0.0,
                days_held=0,
                sector=self.sector_map.get(symbol, "unknown"),
                asset_class=self._classify_symbol(symbol),
            )
            self.logger.info(f"Opened {symbol} position: {quantity} @ ${price:.2f}")

        # Update cash and tracking
        self.cash -= total_cost
        self.total_trades += 1

        # Record trade
        self.trade_history.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "cash_after": self.cash,
            }
        )

        self._update_portfolio_value()
        return True

    def reduce_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
    ) -> bool:
        """
        Reduce or close a position

        Returns:
            bool: Success status
        """
        if symbol not in self.positions:
            self.logger.error(f"No position in {symbol} to reduce")
            return False

        position = self.positions[symbol]

        if quantity > position.quantity:
            self.logger.warning(
                f"Reducing quantity {quantity} exceeds position {position.quantity}"
            )
            quantity = position.quantity

        # Calculate realized P&L
        realized_pnl = (price - position.entry_price) * quantity - commission
        self.realized_pnl += realized_pnl

        # Update trade statistics
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update position or close
        if quantity >= position.quantity:
            # Close position
            self.closed_positions.append(
                {
                    "symbol": symbol,
                    "entry_date": position.entry_date,
                    "exit_date": timestamp,
                    "entry_price": position.entry_price,
                    "exit_price": price,
                    "quantity": position.quantity,
                    "realized_pnl": realized_pnl,
                    "return_pct": (price / position.entry_price - 1) * 100,
                    "days_held": (timestamp - position.entry_date).days,
                }
            )
            del self.positions[symbol]
            self.logger.info(f"Closed {symbol} position: PnL ${realized_pnl:.2f}")
        else:
            # Partial close
            position.quantity -= quantity
            position.current_value = position.quantity * price
            self.logger.info(
                f"Reduced {symbol} position by {quantity}: PnL ${realized_pnl:.2f}"
            )

        # Update cash
        proceeds = quantity * price - commission
        self.cash += proceeds
        self.total_trades += 1

        # Record trade
        self.trade_history.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "realized_pnl": realized_pnl,
                "cash_after": self.cash,
            }
        )

        self._update_portfolio_value()
        return True

    def update_prices(self, price_data: Dict[str, float], timestamp: datetime):
        """Update all position prices"""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                new_price = price_data[symbol]
                position.current_price = new_price
                position.current_value = position.quantity * new_price
                position.unrealized_pnl = position.current_value - (
                    position.quantity * position.entry_price
                )
                position.unrealized_pnl_pct = new_price / position.entry_price - 1
                position.days_held = (timestamp - position.entry_date).days

        self._update_portfolio_value()
        self._update_weights()

    def _update_portfolio_value(self):
        """Update total portfolio value"""
        positions_value = sum(p.current_value for p in self.positions.values())
        self.portfolio_value = self.cash + positions_value

    def _update_weights(self):
        """Update position and sector weights"""
        if self.portfolio_value <= 0:
            return

        for position in self.positions.values():
            position.weight = position.current_value / self.portfolio_value

    def _classify_symbol(self, symbol: str) -> str:
        """Classify symbol as risk_on or risk_off"""
        if symbol in self.risk_on_symbols:
            return "risk_on"
        elif symbol in self.risk_off_symbols:
            return "risk_off"
        else:
            return "unknown"

    def get_portfolio_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Get current portfolio snapshot"""
        position_weights = {s: p.weight for s, p in self.positions.items()}

        # Calculate sector weights
        sector_weights = defaultdict(float)
        for position in self.positions.values():
            sector = position.sector or "unknown"
            sector_weights[sector] += position.weight

        # Calculate asset class weights
        asset_class_weights = defaultdict(float)
        for position in self.positions.values():
            asset_class_weights[position.asset_class] += position.weight

        # Find largest and smallest positions
        if self.positions:
            sorted_positions = sorted(
                self.positions.items(), key=lambda x: x[1].current_value
            )
            largest = sorted_positions[-1][0] if sorted_positions else None
            smallest = sorted_positions[0][0] if sorted_positions else None
        else:
            largest = smallest = None

        # Calculate average position age
        if self.positions:
            avg_age = np.mean([p.days_held for p in self.positions.values()])
        else:
            avg_age = 0

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=self.portfolio_value,
            cash=self.cash,
            positions_value=self.portfolio_value - self.cash,
            num_positions=len(self.positions),
            position_weights=dict(position_weights),
            sector_weights=dict(sector_weights),
            asset_class_weights=dict(asset_class_weights),
            largest_position=largest,
            smallest_position=smallest,
            total_unrealized_pnl=sum(p.unrealized_pnl for p in self.positions.values()),
            average_position_age=avg_age,
        )

        self.portfolio_history.append(snapshot)
        return snapshot

    def check_constraints(self) -> Dict[str, List[str]]:
        """Check portfolio constraints and return violations"""
        violations = defaultdict(list)

        # Check position concentration
        for symbol, position in self.positions.items():
            if position.weight > self.max_single_position:
                violations["position_concentration"].append(
                    f"{symbol}: {position.weight:.1%} > {self.max_single_position:.1%}"
                )

        # Check sector concentration
        sector_weights = defaultdict(float)
        for position in self.positions.values():
            if position.sector:
                sector_weights[position.sector] += position.weight

        for sector, weight in sector_weights.items():
            if weight > self.max_sector_exposure:
                violations["sector_concentration"].append(
                    f"{sector}: {weight:.1%} > {self.max_sector_exposure:.1%}"
                )

        # Check position count
        if len(self.positions) > self.max_positions:
            violations["position_count"].append(
                f"Positions: {len(self.positions)} > {self.max_positions}"
            )

        return dict(violations)

    def get_position_recommendations(
        self, target_positions: Set[str]
    ) -> Dict[str, str]:
        """
        Get recommendations for position adjustments

        Args:
            target_positions: Set of symbols that should be held

        Returns:
            Dict of symbol -> action (e.g., 'add', 'remove', 'hold', 'reduce')
        """
        recommendations = {}

        current_symbols = set(self.positions.keys())

        # Positions to add
        for symbol in target_positions - current_symbols:
            if len(self.positions) < self.max_positions:
                recommendations[symbol] = "add"
            else:
                recommendations[symbol] = "skip"  # At position limit

        # Positions to remove
        for symbol in current_symbols - target_positions:
            recommendations[symbol] = "remove"

        # Check existing positions
        for symbol in current_symbols & target_positions:
            position = self.positions[symbol]

            # Check if position is too large
            if position.weight > self.max_single_position:
                recommendations[symbol] = "reduce"
            # Check if position has been held too long with poor performance
            elif position.days_held > 60 and position.unrealized_pnl_pct < -0.10:
                recommendations[symbol] = "remove"
            else:
                recommendations[symbol] = "hold"

        return recommendations

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        total_trades = self.winning_trades + self.losing_trades

        metrics = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions_value": self.portfolio_value - self.cash,
            "num_positions": len(self.positions),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values()),
            "total_pnl": self.realized_pnl
            + sum(p.unrealized_pnl for p in self.positions.values()),
            "total_return_pct": ((self.portfolio_value / self.initial_capital) - 1)
            * 100,
            "total_trades": total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / total_trades if total_trades > 0 else 0,
            "avg_position_size": (
                np.mean([p.current_value for p in self.positions.values()])
                if self.positions
                else 0
            ),
            "cash_utilization": (
                1 - (self.cash / self.portfolio_value)
                if self.portfolio_value > 0
                else 0
            ),
        }

        # Add profit factor if we have wins and losses
        if self.winning_trades > 0 and self.losing_trades > 0:
            wins = [
                t["realized_pnl"]
                for t in self.trade_history
                if t.get("realized_pnl", 0) > 0
            ]
            losses = [
                abs(t["realized_pnl"])
                for t in self.trade_history
                if t.get("realized_pnl", 0) < 0
            ]

            if wins and losses:
                metrics["profit_factor"] = sum(wins) / sum(losses)
                metrics["avg_win"] = np.mean(wins)
                metrics["avg_loss"] = np.mean(losses)

        return metrics

    def get_position_details(self, symbol: str = None) -> Dict:
        """Get detailed position information"""
        if symbol:
            if symbol in self.positions:
                pos = self.positions[symbol]
                return {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "current_value": pos.current_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct * 100,
                    "weight": pos.weight * 100,
                    "days_held": pos.days_held,
                    "asset_class": pos.asset_class,
                }
            else:
                return {}
        else:
            # Return all positions
            return {
                symbol: {
                    "value": pos.current_value,
                    "weight": pos.weight * 100,
                    "pnl_pct": pos.unrealized_pnl_pct * 100,
                    "days_held": pos.days_held,
                }
                for symbol, pos in self.positions.items()
            }
