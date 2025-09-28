# backtester/portfolio.py
# Clean portfolio implementation - pure execution layer with no risk management

import logging
from datetime import datetime

import numpy as np

import config.trading_parameters as tp
from backtester.events import OrderEvent


class Portfolio:
    """
    Portfolio handles execution and bookkeeping only.
    No risk decisions - those belong in the strategy layer.

    Responsibilities:
    - Execute buy/sell orders
    - Track positions and holdings
    - Calculate position sizes based on configuration
    - Record trade statistics
    - Handle transaction costs
    """

    def __init__(
        self, events_queue, symbol_list, data_handler, initial_capital=100000.0
    ):
        """Initialize portfolio with basic execution capabilities"""
        self.events = events_queue
        self.symbol_list = symbol_list
        self.data_handler = data_handler
        self.initial_capital = initial_capital

        # === Position Sizing Configuration ===
        self.target_utilization = tp.TRADING_PARAMS["position_sizing"][
            "target_utilization"
        ]
        self.max_positions = tp.TRADING_PARAMS["position_sizing"]["max_positions"]
        self.min_positions = tp.TRADING_PARAMS["position_sizing"]["min_positions"]
        self.position_buffer = tp.TRADING_PARAMS["position_sizing"]["position_buffer"]

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

        # === Performance Tracking ===
        self.peak_value = initial_capital
        self.current_drawdown = 0.0

        logging.info(f"Portfolio initialized:")
        logging.info(f"  Initial capital: ${initial_capital:,.0f}")
        logging.info(f"  Target utilization: {self.target_utilization:.1%}")
        logging.info(f"  Position limits: {self.min_positions}-{self.max_positions}")
        logging.info(f"  Max single position: {self.max_single_position:.1%}")

    def update_timeindex(self, event):
        """
        Update portfolio valuation at each market event
        Pure bookkeeping - no risk decisions
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

        if total_value > self.initial_capital * 100:
            logging.error(f"WARNING: Unrealistic portfolio value: ${total_value:,.0f}")

    def on_signal(self, event):
        if event.type != "Signal":
            return

        if event.signal_type == "Long" and event.score > 1.0:
            logging.debug(
                f"Received legacy score format {event.score}, converting to weight."
            )
            event.score = event.score / 10.0

        if event.signal_type == "Long":
            self._process_buy_signal(event)
        elif event.signal_type == "Exit":
            self._process_sell_signal(event)
        elif event.signal_type == "Reduce":
            self._process_reduce_signal(event)
        elif event.signal_type == "Short":
            logging.info(
                f"Ignoring SHORT signal for {event.symbol} (shorting not supported)"
            )

    def _process_buy_signal(self, signal):
        symbol = signal.symbol
        target_weight = signal.score

        # --- Validate target weight ---
        if (
            not 0 < target_weight <= (self.max_single_position + 0.001)
        ):  # Add buffer for float precision
            logging.warning(
                f"Signal for {symbol} has invalid target weight {target_weight:.2%}. "
                f"Must be between 0 and {self.max_single_position:.1%}. Ignoring."
            )
            return

        # --- Get current market price ---
        latest_bar = self.data_handler.get_latest_bars(symbol)
        if not latest_bar or not latest_bar[0] or latest_bar[0]["close"] <= 0:
            logging.warning(
                f"Could not get a valid current price for {symbol}. Ignoring buy signal."
            )
            return
        current_price = latest_bar[0]["close"]

        # --- Core Sizing Logic ---
        total_equity = self.current_holdings["total"]
        target_value = total_equity * target_weight

        current_shares = self.current_positions.get(symbol, 0)
        current_value = current_shares * current_price

        value_to_add = target_value - current_value

        # If we don't need to add more, exit. Reductions are handled by 'Reduce' or 'Exit' signals.
        if value_to_add <= 0:
            return

        quantity_to_buy = int(value_to_add / current_price)

        if quantity_to_buy <= 0:
            return

        # --- Cash Constraint Check ---
        required_cash = quantity_to_buy * current_price
        available_cash = self.current_holdings["cash"] * (1 - self.position_buffer)

        if required_cash > available_cash:
            new_quantity = int(available_cash / current_price)
            logging.warning(
                f"Cash constraint for {symbol}. Reducing order from {quantity_to_buy} to {new_quantity} shares."
            )
            quantity_to_buy = new_quantity

        if quantity_to_buy <= 0:
            logging.info(f"Not enough cash to place any order for {symbol}.")
            return

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type="Mkt",
            quantity=quantity_to_buy,
            direction="Buy",
        )
        self.events.put(order)

        action_log = "ADJUST" if current_shares > 0 else "NEW"
        logging.info(
            f"PORTFOLIO: {action_log} BUY order for {quantity_to_buy} {symbol} to reach target weight of {target_weight:.2%}"
        )

    def _process_sell_signal(self, signal):
        """Process sell signal - exit entire position"""
        symbol = signal.symbol
        quantity = self.current_positions.get(symbol, 0)

        if quantity <= 0:
            logging.debug(f"No position in {symbol} to sell")
            return

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type="Mkt",
            quantity=quantity,
            direction="Sell",
        )
        self.events.put(order)
        logging.info(f"SELL ORDER: {quantity} shares of {symbol}")

    def _process_reduce_signal(self, signal):
        """Process partial position reduction"""
        symbol = signal.symbol
        current_quantity = self.current_positions.get(symbol, 0)

        if current_quantity <= 0:
            logging.debug(f"No position in {symbol} to reduce")
            return

        # Calculate shares to sell based on reduction ratio
        reduction_ratio = signal.score if hasattr(signal, "score") else 0.5
        shares_to_sell = int(current_quantity * reduction_ratio)

        if shares_to_sell > 0:
            order = OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type="Mkt",
                quantity=shares_to_sell,
                direction="Sell",
            )
            self.events.put(order)
            logging.info(
                f"REDUCE ORDER: Selling {shares_to_sell}/{current_quantity} shares of {symbol}"
            )

    def on_fill(self, event):
        """Process fill events - update positions and cash"""
        if event.type != "Fill":
            return

        if event.direction == "Buy":
            self._process_buy_fill(event)
        elif event.direction == "Sell":
            self._process_sell_fill(event)

    def _process_buy_fill(self, event):
        """Process executed buy order"""
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

        # Track entry for PnL calculation
        if symbol not in self.position_entries:
            self.position_entries[symbol] = {
                "price": price_per_share,
                "quantity": quantity,
                "entry_time": event.timestamp,
            }
        else:
            # Average up/down
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
                "entry_time": old_entry["entry_time"],
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
        """Process executed sell order"""
        symbol = event.symbol
        quantity = event.quantity
        fill_cost = event.fill_cost
        commission = event.commission

        # Calculate price per share
        price_per_share = fill_cost / quantity if quantity > 0 else 0

        # Update positions
        self.current_positions[symbol] = max(
            0, self.current_positions.get(symbol, 0) - quantity
        )

        # Update cash
        proceeds = fill_cost - commission
        self.current_holdings["cash"] += proceeds

        # Calculate PnL if we have entry data
        pnl = 0.0
        if symbol in self.position_entries:
            entry_price = self.position_entries[symbol]["price"]
            pnl = (price_per_share - entry_price) * quantity - commission

            # Update trade statistics
            self._update_trade_stats(pnl)

            # Clear or update entry tracking
            if self.current_positions[symbol] == 0:
                # Position fully closed
                del self.position_entries[symbol]
            else:
                # Partial sale
                self.position_entries[symbol]["quantity"] -= quantity

        # Record trade
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
            }
        )

        logging.info(
            f"FILLED SELL: {quantity} {symbol} @ ${price_per_share:.2f} (PnL: ${pnl:.2f})"
        )

    def _update_trade_stats(self, pnl):
        """Update trade statistics"""
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
