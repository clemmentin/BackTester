import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


@dataclass
class StopInfo:
    """Stores all stop-loss related information for a single position."""

    symbol: str
    entry_price: float
    stop_price: float
    highest_price: float
    trailing_activated: bool
    # MODIFIED: Added fields to track profit protection status.
    profit_level_achieved: int
    current_trail_distance: float


@dataclass
class StopSignal:
    """Represents a signal to exit a position due to a stop being triggered."""

    symbol: str
    action: str
    reason: str
    current_price: float
    stop_price: float


class StopManager:
    """Manages all stop-loss and profit protection logic for active positions."""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)

        self.enabled = config.get("enabled", True)
        self.fixed_stop_pct = config.get("fixed_stop_loss", 0.10)
        self.trailing_activation_pct = config.get("trailing_stop_activation", 0.10)
        self.base_trailing_distance = config.get("trailing_stop_distance", 0.08)

        # NEW: Load profit protection configuration
        self.pp_config = config.get("profit_protection", {})
        self.pp_enabled = self.pp_config.get("enabled", False)
        # Sort levels by profit threshold to ensure they are checked in order.
        self.pp_levels = sorted(
            self.pp_config.get("levels", []), key=lambda x: x["profit"]
        )

        self.stops: Dict[str, StopInfo] = {}

        self.logger.info(
            f"StopManager Initialized. Profit Protection Enabled: {self.pp_enabled} with {len(self.pp_levels)} levels."
        )

    def add_position(
        self, symbol: str, entry_price: float, stop_loss_pct: Optional[float] = None
    ):
        """Adds a new position to be tracked, using a dynamic or fixed stop loss."""
        if not self.enabled:
            return

        # Use the dynamic stop_loss_pct if provided, otherwise fall back to the fixed one.
        stop_pct_to_use = (
            stop_loss_pct
            if stop_loss_pct is not None and stop_loss_pct > 0
            else self.fixed_stop_pct
        )

        initial_stop = entry_price * (1 - stop_pct_to_use)

        self.stops[symbol] = StopInfo(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=initial_stop,
            highest_price=entry_price,
            trailing_activated=False,
            profit_level_achieved=-1,  # Start at -1, meaning no level achieved
            current_trail_distance=self.base_trailing_distance,
        )

        self.logger.info(
            f"Stop added: {symbol} entry=${entry_price:.2f}, stop=${initial_stop:.2f} ({stop_pct_to_use:.2%})"
        )

    def update_stops(
        self, price_data: Dict[str, float], timestamp: datetime
    ) -> List[StopSignal]:
        """Updates all stops based on the latest price data and generates exit signals."""
        if not self.enabled:
            return []

        signals = []
        symbols_to_check = list(self.stops.keys())

        for symbol in symbols_to_check:
            if symbol not in price_data or symbol not in self.stops:
                continue

            stop_info = self.stops[symbol]
            current_price = price_data[symbol]

            current_gain = (current_price / stop_info.entry_price) - 1

            if current_price > stop_info.highest_price:
                stop_info.highest_price = current_price

            # Check and apply tiered profit protection
            if self.pp_enabled:
                for i, level in enumerate(self.pp_levels):
                    if (
                        current_gain >= level["profit"]
                        and i > stop_info.profit_level_achieved
                    ):
                        self.logger.info(
                            f"Profit Protection Level {i + 1} activated for {symbol} at {current_gain:.1%} gain."
                        )
                        stop_info.profit_level_achieved = i
                        action = level["action"]

                        if action == "lock_in":
                            lock_in_price = stop_info.entry_price * (
                                1 + level["lock_in_pct"]
                            )
                            stop_info.stop_price = max(
                                stop_info.stop_price, lock_in_price
                            )
                            self.logger.info(
                                f"  -> Action: Lock-in. New stop for {symbol} set to ${stop_info.stop_price:.2f}"
                            )

                        elif action == "trail":
                            stop_info.current_trail_distance = level["trail_pct"]
                            stop_info.trailing_activated = True
                            self.logger.info(
                                f"  -> Action: Trail. Trail distance for {symbol} updated to {stop_info.current_trail_distance:.1%}"
                            )

            # Check and apply the base trailing stop if not already activated
            if not stop_info.trailing_activated:
                gain_for_trailing = (
                    stop_info.highest_price / stop_info.entry_price
                ) - 1
                if gain_for_trailing >= self.trailing_activation_pct:
                    stop_info.trailing_activated = True
                    self.logger.info(
                        f"{symbol}: Base trailing stop activated at {gain_for_trailing:.1%} gain"
                    )

            # Update the stop price based on current trailing logic
            if stop_info.trailing_activated:
                new_stop_price = stop_info.highest_price * (
                    1 - stop_info.current_trail_distance
                )
                stop_info.stop_price = max(stop_info.stop_price, new_stop_price)

            # Final check to see if the current price has breached the stop price
            if current_price <= stop_info.stop_price:
                reason = (
                    "Profit protection"
                    if stop_info.trailing_activated
                    or stop_info.profit_level_achieved > -1
                    else "Initial stop"
                )
                signals.append(
                    StopSignal(
                        symbol=symbol,
                        action="exit",
                        reason=f"{reason} triggered at ${stop_info.stop_price:.2f}",
                        current_price=current_price,
                        stop_price=stop_info.stop_price,
                    )
                )
        return signals

    def remove_position(self, symbol: str):
        """Removes a position from the stop tracker after it's sold."""
        if symbol in self.stops:
            del self.stops[symbol]
            self.logger.debug(f"Stop tracker removed for: {symbol}")

    def get_stop_info(self, symbol: str) -> Optional[StopInfo]:
        """Retrieves stop information for a single symbol."""
        return self.stops.get(symbol)

    def get_all_stops(self) -> Dict[str, float]:
        """Returns a dictionary of all tracked symbols and their stop prices."""
        return {s: info.stop_price for s, info in self.stops.items()}
