import logging
from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd

import config.trading_parameters as tp
from backtester.events import OrderEvent, SignalEvent


class Portfolio:
    """ """
    def __init__(
        self,
        events_queue,
        symbol_list,
        data_handler,
        initial_capital=100000.0,
        all_market_data=None,
    ):
        """Initialize portfolio with basic execution capabilities"""
        self.events = events_queue
        self.symbol_list = symbol_list
        self.data_handler = data_handler
        self.initial_capital = initial_capital

        self.logger = logging.getLogger(self.__class__.__name__)
        # === Position Sizing Configuration ===
        sizing_params = tp.TRADING_PARAMS.get("position_sizing", {})
        self.target_utilization = sizing_params.get("target_utilization", 0.98)
        self.position_buffer = sizing_params.get("position_buffer", 0.02)
        pos_mgmt_params = tp.TRADING_PARAMS.get("position_management", {})
        self.max_positions = pos_mgmt_params.get("max_total_positions", 20)
        self.min_positions = pos_mgmt_params.get("min_total_positions", 5)

        # === Transaction Costs ===
        self.commission_rate = tp.TRADING_PARAMS["transaction_costs"]["commission_rate"]
        self.min_commission = tp.TRADING_PARAMS["transaction_costs"]["min_commission"]
        self.slippage_rate = tp.TRADING_PARAMS["transaction_costs"]["slippage_rate"]

        # === Position Limits ===
        self.max_single_position = tp.RISK_PARAMS["portfolio_limits"][
            "max_single_position"
        ]
        self.min_single_position = tp.RISK_PARAMS["portfolio_limits"][
            "min_single_position"
        ]

        # === Current State ===
        self.current_positions = {s: 0 for s in symbol_list}  # Share quantities
        self.current_holdings = {
            "cash": initial_capital,
            "total": initial_capital,
            "timestamp": None,
        }
        for s in symbol_list:
            self.current_holdings[s] = 0.0

        # === Position Tracking (for PnL calculation) ===
        self.position_entries = (
            {}
        )  # {symbol: {'price': avg_entry_price, 'quantity': shares}}

        # === Historical Records ===
        self.all_holdings = []  # List of holdings snapshots
        self.all_positions = []  # List of position snapshots
        self.all_trades = []  # List of executed trades

        # === Trade Statistics ===
        self.trade_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "gross_profits": 0.0,
            "gross_losses": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

        # === Market Data ===
        self.all_market_data = all_market_data
        if self.all_market_data is not None:
            try:
                spy_data = self.all_market_data.xs(
                    "SPY", level="symbol", drop_level=True
                )
                if not spy_data.empty and "close" in spy_data.columns:
                    self.benchmark_prices = spy_data["close"].sort_index()
            except KeyError:
                self.logger.warning(
                    "SPY symbol not found in all_market_data for benchmark."
                )
            if self.benchmark_prices is not None and not self.benchmark_prices.empty:
                self.logger.info("Portfolio successfully loaded SPY benchmark prices.")
            else:
                self.logger.warning(
                    "Failed to load SPY benchmark prices from all_market_data."
                )

        # === Performance Tracking ===
        self.peak_value = initial_capital
        self.current_drawdown = 0.0

        self.logger.info(f"Portfolio initialized:")
        self.logger.info(f"  Initial capital: ${initial_capital:,.0f}")
        self.logger.info(f"  Target utilization: {self.target_utilization:.1%}")
        self.logger.info(
            f"  Position limits: {self.min_positions}-{self.max_positions}"
        )
        self.logger.info(f"  Max single position: {self.max_single_position:.1%}")

    def update_timeindex(self, event):
        """Update portfolio valuation at each market event
        Pure bookkeeping - no risk decisions

        Args:
          event: 

        Returns:

        """
        if event.type != "Market":
            return

        # Calculate total portfolio value
        total_value = self.current_holdings["cash"]

        for symbol in self.symbol_list:
            quantity = self.current_positions[symbol]

            if quantity > 0:
                latest_bar = self.data_handler.get_latest_bars(symbol)
                if latest_bar and latest_bar[0]:
                    current_price = latest_bar[0]["close"]
                    market_value = current_price * quantity
                    self.current_holdings[symbol] = market_value
                    total_value += market_value
                else:
                    # Use previous value if no new data
                    total_value += self.current_holdings.get(symbol, 0.0)
            else:
                self.current_holdings[symbol] = 0.0

        # Update total value
        self.current_holdings["total"] = total_value
        self.current_holdings["timestamp"] = event.timestamp

        # Track peak and drawdown (for reporting only, not for decisions)
        if total_value > self.peak_value:
            self.peak_value = total_value
        self.current_drawdown = (
            (self.peak_value - total_value) / self.peak_value
            if self.peak_value > 0
            else 0
        )

        # Record snapshot
        holdings_snapshot = self.current_holdings.copy()
        self.all_holdings.append(holdings_snapshot)

        positions_snapshot = self.current_positions.copy()
        positions_snapshot["timestamp"] = event.timestamp
        self.all_positions.append(positions_snapshot)

        positions_value = sum(self.current_holdings.get(s, 0) for s in self.symbol_list)
        cash = self.current_holdings["cash"]
        calculated_total = cash + positions_value

        if abs(calculated_total - total_value) > 1:
            self.logger.warning(
                f"Portfolio calculation mismatch: {total_value} vs {calculated_total}"
            )
            total_value = calculated_total

        self.current_holdings["total"] = total_value

    def on_signal(self, event):
        """

        Args:
          event: 

        Returns:

        """
        if event.type != "Signal":
            return
        if event.signal_type == "Long":
            self._adjust_position_to_target_value(event)

        elif event.signal_type == "Exit":
            self._process_sell_signal(event)
        elif event.signal_type == "Reduce":
            self.logger.warning(
                f"Received deprecated 'Reduce' signal for {event.symbol}. "
                f"This should be handled by a 'Long' signal with a new target value."
            )
        elif event.signal_type == "Short":
            logging.info(
                f"Ignoring SHORT signal for {event.symbol} (shorting not supported)"
            )

    def _adjust_position_to_target_value(self, signal: SignalEvent):
        """NEW CORE METHOD: Adjusts a position to a specific target dollar value.
        This handles initial buys, increasing a position, and reducing a position.

        Args:
          signal: SignalEvent: 

        Returns:

        """
        symbol = signal.symbol
        target_value = signal.score  # The 'score' is now the target dollar value

        # --- 1. Get current market price ---
        latest_bar = self.data_handler.get_latest_bars(symbol)
        if not latest_bar or not latest_bar[0] or latest_bar[0]["close"] <= 0:
            self.logger.warning(
                f"Could not get a valid current price for {symbol}. Ignoring adjustment signal."
            )
            return
        current_price = latest_bar[0]["close"]

        # --- 2. Calculate required shares for target value ---
        target_shares = int(target_value / current_price)

        # --- 3. Get current holdings ---
        current_shares = self.current_positions.get(symbol, 0)

        # --- 4. Determine the trade quantity and direction ---
        quantity_to_trade = target_shares - current_shares

        # --- 5. Generate Order Event ---
        if quantity_to_trade > 0:
            # This is a BUY or INCREASE order

            # REMOVED: Cash constraint logic is now handled by the DecisionEngine.
            # The Portfolio layer now assumes incoming signals are fully funded.

            order = OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type="Mkt",
                quantity=quantity_to_trade,
                direction="Buy",
            )
            self.events.put(order)

            action_log = "NEW BUY" if current_shares == 0 else "INCREASE"
            self.logger.info(
                f"PORTFOLIO: {action_log} order for {quantity_to_trade} {symbol} "
                f"to reach target value of ${target_value:,.0f}"
            )

        elif quantity_to_trade < 0:
            # This is a SELL or REDUCE order

            quantity_to_sell = abs(quantity_to_trade)

            # Sanity check: ensure we don't sell more than we own
            if quantity_to_sell > current_shares:
                self.logger.warning(
                    f"Attempted to sell {quantity_to_sell} shares of {symbol}, but only hold {current_shares}. "
                    f"Adjusting to sell all."
                )
                quantity_to_sell = current_shares

            if quantity_to_sell <= 0:
                return

            order = OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type="Mkt",
                quantity=quantity_to_sell,
                direction="Sell",
            )
            self.events.put(order)

            self.logger.info(
                f"PORTFOLIO: REDUCE order, selling {quantity_to_sell} {symbol} "
                f"to reach target value of ${target_value:,.0f}"
            )

    def _process_sell_signal(self, signal):
        """Process sell signal - exit entire position

        Args:
          signal: 

        Returns:

        """
        symbol = signal.symbol
        quantity = self.current_positions.get(symbol, 0)

        if quantity <= 0:
            logging.debug(f"No position in {symbol} to sell")
            return

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            order_type="Mkt",
            quantity=quantity,
            direction="Sell",
        )
        self.events.put(order)
        logging.info(f"SELL ORDER: {quantity} shares of {symbol}")

    def on_fill(self, event):
        """Process fill events - update positions and cash

        Args:
          event: 

        Returns:

        """
        if event.type != "Fill":
            return

        if event.direction == "Buy":
            self._process_buy_fill(event)
        elif event.direction == "Sell":
            self._process_sell_fill(event)

    def _process_buy_fill(self, event):
        """Process executed buy order using Average Cost Method.
        
        When adding to an existing position, the cost basis is updated to the
        weighted average of the old position and new purchase:
        New Average Cost = (Old Cost × Old Qty + New Cost × New Qty) / Total Qty

        Args:
          event: FillEvent containing executed trade details

        Returns:

        """
        symbol = event.symbol
        quantity = event.quantity
        fill_cost = event.fill_cost
        commission = event.commission

        # Calculate price per share
        price_per_share = fill_cost / quantity if quantity > 0 else 0

        # Update positions
        self.current_positions[symbol] = (
            self.current_positions.get(symbol, 0) + quantity
        )

        # Update cash
        total_cost = fill_cost + commission
        self.current_holdings["cash"] -= total_cost

        # Track entry for PnL calculation using Average Cost Method
        if symbol not in self.position_entries:
            # First entry for this symbol
            self.position_entries[symbol] = {
                "price": price_per_share,
                "quantity": quantity,
                "entry_time": event.timestamp,
            }
        else:
            # Average up/down: blend old and new cost basis
            old_entry = self.position_entries[symbol]
            old_cost = old_entry["price"] * old_entry["quantity"]
            new_cost = price_per_share * quantity
            total_quantity = old_entry["quantity"] + quantity
            avg_price = (
                (old_cost + new_cost) / total_quantity if total_quantity > 0 else 0
            )

            self.position_entries[symbol] = {
                "price": avg_price,
                "quantity": total_quantity,
                "entry_time": old_entry["entry_time"],  # Keep original entry time
            }

        # Record trade
        self.all_trades.append(
            {
                "timestamp": event.timestamp,
                "symbol": symbol,
                "direction": "Buy",
                "quantity": quantity,
                "price": price_per_share,
                "commission": commission,
                "total_cost": total_cost,
            }
        )

        logging.info(
            f"FILLED BUY: {quantity} {symbol} @ ${price_per_share:.2f} (commission: ${commission:.2f})"
        )

    def _process_sell_fill(self, event):
        """Process executed sell order using Average Cost Method.
        
        PnL is calculated using the average cost basis:
        Realized PnL = (Exit Price - Average Cost) × Quantity Sold - Commission
        
        For partial exits, the remaining position keeps the same average cost.

        Args:
          event: FillEvent containing executed trade details

        Returns:

        """
        symbol = event.symbol
        quantity = event.quantity
        fill_cost = event.fill_cost
        commission = event.commission
        price_per_share = fill_cost / quantity if quantity > 0 else 0

        self.current_positions[symbol] = max(
            0, self.current_positions.get(symbol, 0) - quantity
        )

        proceeds = fill_cost - commission
        self.current_holdings["cash"] += proceeds

        pnl = 0.0
        entry_date = None
        entry_price = 0.0
        holding_days = 0
        market_return = 0.0

        if symbol in self.position_entries:
            entry_info = self.position_entries[symbol]
            entry_price = entry_info["price"]  # Average cost basis
            entry_date = entry_info.get("entry_time")
            # PnL calculation: (exit price - average cost) * quantity - commission
            pnl = (price_per_share - entry_price) * quantity - commission

            # Calculate holding period
            if entry_date:
                holding_days = (event.timestamp - entry_date).days
            if self.benchmark_prices is not None and entry_date is not None:
                try:
                    start_bm_price = self.benchmark_prices.asof(entry_date)
                    end_bm_price = self.benchmark_prices.asof(event.timestamp)

                    if (
                        pd.notna(start_bm_price)
                        and pd.notna(end_bm_price)
                        and start_bm_price > 0
                    ):
                        market_return = (end_bm_price / start_bm_price) - 1
                except Exception as e:
                    self.logger.warning(f"为 {symbol} 计算 market_return 时出错: {e}")

            self._update_trade_stats(pnl)

            if self.current_positions[symbol] == 0:
                # Full exit: remove position entry
                del self.position_entries[symbol]
            else:
                # Partial exit: reduce quantity but keep same average cost
                self.position_entries[symbol]["quantity"] -= quantity

        # Record trade with MORE details - CRITICAL FIX
        self.all_trades.append(
            {
                "timestamp": event.timestamp,
                "symbol": symbol,
                "direction": "Sell",
                "quantity": quantity,
                "price": price_per_share,
                "commission": commission,
                "proceeds": proceeds,
                "pnl": pnl,
                "entry_date": entry_date,  # ADDED
                "exit_date": event.timestamp,  # ADDED
                "entry_price": entry_price,  # ADDED
                "exit_price": price_per_share,  # ADDED
                "holding_days": holding_days,  # ADDED
                "return_pct": (
                    (pnl / (entry_price * quantity) * 100)
                    if entry_price > 0 and quantity > 0
                    else 0
                ),
                "market_return": market_return,
            }
        )

        logging.info(
            f"FILLED SELL: {quantity} {symbol} @ ${price_per_share:.2f} (PnL: ${pnl:.2f})"
        )

    def _update_trade_stats(self, pnl):
        """Update trade statistics

        Args:
          pnl: 

        Returns:

        """
        self.trade_stats["total_trades"] += 1
        self.trade_stats["total_pnl"] += pnl

        if pnl > 0:
            self.trade_stats["winning_trades"] += 1
            self.trade_stats["gross_profits"] += pnl
            self.trade_stats["largest_win"] = max(self.trade_stats["largest_win"], pnl)
            self.trade_stats["consecutive_wins"] += 1
            self.trade_stats["consecutive_losses"] = 0
            self.trade_stats["max_consecutive_wins"] = max(
                self.trade_stats["max_consecutive_wins"],
                self.trade_stats["consecutive_wins"],
            )
        else:
            self.trade_stats["losing_trades"] += 1
            self.trade_stats["gross_losses"] += abs(pnl)
            self.trade_stats["largest_loss"] = min(
                self.trade_stats["largest_loss"], pnl
            )
            self.trade_stats["consecutive_losses"] += 1
            self.trade_stats["consecutive_wins"] = 0
            self.trade_stats["max_consecutive_losses"] = max(
                self.trade_stats["max_consecutive_losses"],
                self.trade_stats["consecutive_losses"],
            )

    def get_trade_statistics(self):
        """Get comprehensive trade statistics"""
        total = self.trade_stats["total_trades"]

        if total == 0:
            return {"total_trades": 0, "message": "No trades executed yet"}

        win_rate = self.trade_stats["winning_trades"] / total
        avg_win = self.trade_stats["gross_profits"] / max(
            1, self.trade_stats["winning_trades"]
        )
        avg_loss = self.trade_stats["gross_losses"] / max(
            1, self.trade_stats["losing_trades"]
        )
        profit_factor = self.trade_stats["gross_profits"] / max(
            0.01, self.trade_stats["gross_losses"]
        )

        return {
            "total_trades": total,
            "winning_trades": self.trade_stats["winning_trades"],
            "losing_trades": self.trade_stats["losing_trades"],
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": self.trade_stats["total_pnl"],
            "gross_profits": self.trade_stats["gross_profits"],
            "gross_losses": self.trade_stats["gross_losses"],
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": self.trade_stats["largest_win"],
            "largest_loss": self.trade_stats["largest_loss"],
            "max_consecutive_wins": self.trade_stats["max_consecutive_wins"],
            "max_consecutive_losses": self.trade_stats["max_consecutive_losses"],
            "current_drawdown": self.current_drawdown,
            "peak_value": self.peak_value,
        }

    def get_current_state(self):
        """Get current portfolio state summary"""
        positions_value = sum(self.current_holdings.get(s, 0) for s in self.symbol_list)

        return {
            "total_value": self.current_holdings["total"],
            "cash": self.current_holdings["cash"],
            "positions_value": positions_value,
            "num_positions": sum(
                1 for s in self.symbol_list if self.current_positions.get(s, 0) > 0
            ),
            "position_details": {
                s: {
                    "shares": self.current_positions[s],
                    "value": self.current_holdings.get(s, 0),
                    "entry_price": self.position_entries.get(s, {}).get("price", 0),
                }
                for s in self.symbol_list
                if self.current_positions.get(s, 0) > 0
            },
            "drawdown": self.current_drawdown,
            "utilization": (
                positions_value / self.current_holdings["total"]
                if self.current_holdings["total"] > 0
                else 0
            ),
        }

    def get_current_position_values(self) -> Dict[str, float]:
        """Returns a dictionary of current position values."""
        values = {}
        for symbol, quantity in self.current_positions.items():
            if quantity > 0:
                values[symbol] = self.current_holdings.get(symbol, 0.0)
        return values
