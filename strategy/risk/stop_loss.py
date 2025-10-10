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
    # NEW: Add fields to track profit protection status
    profit_level_achieved: int  # Tracks the highest profit level achieved (by its index in the config list)
    current_trail_distance: (
        float  # The currently active trailing stop distance percentage
    )


@dataclass
class StopSignal:
    """Represents a signal to exit a position due to a stop being triggered."""

    symbol: str
    action: str  # e.g., 'exit'
    reason: str
    current_price: float
    stop_price: float


class StopManager:
    """Manages all stop-loss and profit protection logic for active positions."""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)

        # Load base configuration
        self.enabled = config.get("enabled", True)
        self.fixed_stop_pct = config.get("fixed_stop_loss", 0.10)
        self.trailing_activation_pct = config.get("trailing_stop_activation", 0.10)
        self.base_trailing_distance = config.get("trailing_stop_distance", 0.08)

        # NEW: Load profit protection configuration
        self.pp_config = config.get("profit_protection", {})
        self.pp_enabled = self.pp_config.get("enabled", False)
        # Sort levels by the 'profit' threshold to ensure they are checked in the correct order
        self.pp_levels = sorted(
            self.pp_config.get("levels", []), key=lambda x: x["profit"]
        )

        # In-memory tracking for all active positions
        self.stops: Dict[str, StopInfo] = {}

        self.logger.info(
            f"StopManager Initialized. Profit Protection Enabled: {self.pp_enabled} with {len(self.pp_levels)} levels."
        )

    def add_position(self, symbol: str, entry_price: float):
        """Adds a new position to be tracked by the stop manager."""
        if not self.enabled:
            return

        initial_stop = entry_price * (1 - self.fixed_stop_pct)

        self.stops[symbol] = StopInfo(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=initial_stop,
            highest_price=entry_price,
            trailing_activated=False,
            # NEW: Initialize new fields
            profit_level_achieved=-1,  # -1 means no profit level has been reached yet
            current_trail_distance=self.base_trailing_distance,  # Start with the base trailing distance
        )

        self.logger.debug(
            f"Stop added: {symbol} entry=${entry_price:.2f} stop=${initial_stop:.2f}"
        )

    def update_stops(
        self, price_data: Dict[str, float], timestamp: datetime
    ) -> List[StopSignal]:
        """
        Updates all stops based on the latest price data and generates exit signals if triggered.
        This is the main logic loop, executed on each time step.
        """
        if not self.enabled:
            return []

        signals = []
        # Create a copy of keys to iterate over, as the dictionary may be modified if a stop is hit
        symbols_to_check = list(self.stops.keys())

        for symbol in symbols_to_check:
            if symbol not in price_data or symbol not in self.stops:
                continue

            stop_info = self.stops[symbol]
            current_price = price_data[symbol]

            # --- 1. Calculate current gain percentage ---
            current_gain = (current_price / stop_info.entry_price) - 1

            # --- 2. Update the highest price seen so far for the position ---
            if current_price > stop_info.highest_price:
                stop_info.highest_price = current_price

            # --- 3. (Core Logic) Check and apply tiered profit protection ---
            if self.pp_enabled:
                for i, level in enumerate(self.pp_levels):
                    # Check if we've reached a new profit level that is higher than any previously achieved
                    if (
                        current_gain >= level["profit"]
                        and i > stop_info.profit_level_achieved
                    ):
                        self.logger.info(
                            f"Profit Protection Level {i + 1} activated for {symbol} at {current_gain:.1%} gain."
                        )
                        stop_info.profit_level_achieved = (
                            i  # Update the highest level reached
                        )

                        action = level["action"]
                        if action == "lock_in":
                            lock_in_price = stop_info.entry_price * (
                                1 + level["lock_in_pct"]
                            )
                            # Raise the stop price to the new lock-in level (but never lower it)
                            stop_info.stop_price = max(
                                stop_info.stop_price, lock_in_price
                            )
                            self.logger.info(
                                f"  -> Action: Lock-in. New stop for {symbol} set to ${stop_info.stop_price:.2f}"
                            )

                        elif action == "trail":
                            # Update the trailing distance to the new percentage defined in this level
                            stop_info.current_trail_distance = level["trail_pct"]
                            # Ensure the trailing stop mechanism is activated
                            stop_info.trailing_activated = True
                            self.logger.info(
                                f"  -> Action: Trail. Trail distance for {symbol} updated to {stop_info.current_trail_distance:.1%}"
                            )

            # --- 4. Check and apply the base trailing stop if not already activated by profit protection ---
            if not stop_info.trailing_activated:
                gain_for_trailing = (
                    stop_info.highest_price / stop_info.entry_price
                ) - 1
                if gain_for_trailing >= self.trailing_activation_pct:
                    stop_info.trailing_activated = True
                    self.logger.info(
                        f"{symbol}: Base trailing stop activated at {gain_for_trailing:.1%} gain"
                    )

            # --- 5. Update the stop price based on the current trailing logic ---
            if stop_info.trailing_activated:
                new_stop_price = stop_info.highest_price * (
                    1 - stop_info.current_trail_distance
                )
                # The stop price can only go up, never down
                stop_info.stop_price = max(stop_info.stop_price, new_stop_price)

            # --- 6. Final check to see if the current price has breached the stop price ---
            if current_price <= stop_info.stop_price:
                reason = (
                    f"Profit stop triggered at ${stop_info.stop_price:.2f}"
                    if stop_info.trailing_activated
                    or stop_info.profit_level_achieved > -1
                    else f"Initial stop triggered at ${stop_info.stop_price:.2f}"
                )

                signals.append(
                    StopSignal(
                        symbol=symbol,
                        action="exit",
                        reason=reason,
                        current_price=current_price,
                        stop_price=stop_info.stop_price,
                    )
                )

        return signals

    def remove_position(self, symbol: str):
        """Removes a position from the stop tracker (e.g., after it's sold)."""
        if symbol in self.stops:
            del self.stops[symbol]
            self.logger.debug(f"Stop tracker removed: {symbol}")

    def get_stop_info(self, symbol: str) -> Optional[StopInfo]:
        """Retrieves the current stop information for a single symbol."""
        return self.stops.get(symbol)

    def get_all_stops(self) -> Dict[str, float]:
        """Returns a dictionary of all tracked symbols and their current stop prices."""
        return {s: info.stop_price for s, info in self.stops.items()}
