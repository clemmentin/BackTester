import logging
import queue
import pandas as pd

from backtester.data import HistoricDBDataHandler
from backtester.execution import SimulatedExecutionHandler
from backtester.portfolio import Portfolio
from backtester.events import SignalEvent

# NEW: Import DecisionEngine instead of strategy config
from strategy.decision_engine import DecisionEngine


class BacktestEngine:
    """
    Modified BacktestEngine that directly integrates the 4-layer DecisionEngine.

    Changes from original:
    - No longer uses strategy abstraction
    - Directly calls DecisionEngine.run_pipeline() at rebalancing intervals
    - Converts target portfolio to signals internally
    """

    def __init__(
        self,
        symbol_list,
        initial_capital,
        all_processed_data,
        start_date=None,
        risk_on_symbols=None,
        risk_off_symbols=None,
        data_manager=None,
        **strategy_kwargs,
    ):
        self.events = queue.Queue()
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital

        # Initialize data handler
        self.data_handler = HistoricDBDataHandler(
            self.events, self.symbol_list, all_processed_data, start_date=start_date
        )

        # Initialize portfolio
        self.portfolio = Portfolio(
            self.events, self.symbol_list, self.data_handler, self.initial_capital
        )

        self.fundamental_data = None
        if data_manager is not None:
            logging.info("Preparing fundamental data for backtest...")
            try:
                individual_data = data_manager.individual_fetcher.fetch(
                    symbol_list, data_types=["fundamentals"], force_refresh=False
                )

                if not individual_data.empty:
                    self.fundamental_data = self._convert_fundamental_data(
                        individual_data
                    )
                    logging.info(
                        f"Loaded fundamental data for {len(self.fundamental_data)} symbols"
                    )
                else:
                    logging.warning("No fundamental data available")

            except Exception as e:
                logging.error(f"Failed to load fundamental data: {e}")

        # Initialize execution handler
        self.execution_handler = SimulatedExecutionHandler(
            self.events, self.data_handler
        )

        self.decision_engine = DecisionEngine(
            initial_capital=initial_capital,
            symbol_list=symbol_list,
            all_processed_data=all_processed_data,
            data_manager=None,
            **strategy_kwargs,
        )

        # Initialize execution handler
        self.execution_handler = SimulatedExecutionHandler(
            self.events, self.data_handler
        )

        # Rebalancing control
        self.rebalance_frequency = strategy_kwargs.get("rebalance_frequency", "weekly")
        self.last_rebalance_date = None
        self.current_targets = {}
        self.min_signal_weight = 0.01

        # Statistics
        self.processed_events = 0
        self.rebalance_count = 0

        logging.info(
            f"BacktestEngine initialized with DecisionEngine "
            f"({self.rebalance_frequency} rebalancing)"
        )

    def run(self):
        """Run the backtest with DecisionEngine integration"""
        logging.info("Starting backtest with DecisionEngine...")

        event_counts = {"Market": 0, "Signal": 0, "Order": 0, "Fill": 0}

        # The main loop is driven by the data_handler's timeline
        while self.data_handler.continue_backtest:
            # 1. Get the current timestamp DIRECTLY from the data handler BEFORE update_bars
            if self.data_handler.current_time_index >= len(self.data_handler.timeline):
                break  # Exit loop if we are at the end of the timeline

            current_timestamp = self.data_handler.timeline[
                self.data_handler.current_time_index
            ]
            current_date = current_timestamp.date()

            # 2. update_bars() will now generate events for `current_timestamp`
            self.data_handler.update_bars()

            # --- Event Processing for the current timestamp ---
            # We collect all market data for this day in a dictionary
            todays_market_data_dict = {}

            while True:
                try:
                    event = self.events.get(block=False)
                except queue.Empty:
                    break

                if event is not None:
                    self.processed_events += 1
                    event_counts[event.type] = event_counts.get(event.type, 0) + 1

                    # Route events to their respective handlers
                    if event.type == "Market":
                        # We just update the portfolio's internal state.
                        # Rebalancing decisions are handled below, once per timestamp.
                        self.portfolio.update_timeindex(event)

                        # Store the data for the decision engine
                        todays_market_data_dict[event.symbol] = event.data

                    elif event.type == "Signal":
                        self.portfolio.on_signal(event)
                    elif event.type == "Order":
                        self.execution_handler.execute_order(event)
                    elif event.type == "Fill":
                        self.portfolio.on_fill(event)

            # 3. Perform rebalancing check ONCE per timestamp, after all events are processed
            if todays_market_data_dict and self._should_rebalance(current_date):
                # Convert the collected daily data into a DataFrame for the DecisionEngine
                market_df_for_day = pd.DataFrame.from_dict(
                    todays_market_data_dict, orient="index"
                )

                self._execute_rebalance(current_timestamp, market_df_for_day)
                self.last_rebalance_date = current_date
                self.rebalance_count += 1

        # Log summary
        logging.info("Backtest completed")
        logging.info(f"Processed events: {self.processed_events}")
        logging.info(f"Rebalances: {self.rebalance_count}")
        for event_type, count in event_counts.items():
            logging.info(f"  {event_type}: {count}")

        # Final results
        final_value = self.portfolio.current_holdings["total"]
        total_return = (final_value / self.initial_capital - 1) * 100
        logging.info(f"Final value: ${final_value:,.2f} (Return: {total_return:.2f}%)")

    def _should_rebalance(self, current_date) -> bool:
        """Determine if rebalancing is needed"""
        if self.last_rebalance_date is None:
            return True

        days_since = (current_date - self.last_rebalance_date).days

        if self.rebalance_frequency == "daily":
            return days_since >= 1
        elif self.rebalance_frequency == "weekly":
            return days_since >= 7
        elif self.rebalance_frequency == "biweekly":
            return days_since >= 14
        elif self.rebalance_frequency == "monthly":
            return days_since >= 30
        else:
            return days_since >= 7

    def _execute_rebalance(self, timestamp, market_data_for_day):
        """
        Execute rebalancing using DecisionEngine.

        This is the core integration point:
        1. Run the 4-layer pipeline
        2. Convert target portfolio to signals
        """
        logging.info(f"\n{'=' * 60}")
        logging.info(f"REBALANCING at {timestamp.date()}")
        logging.info(f"{'=' * 60}")

        # Get current portfolio value
        portfolio_value = self.portfolio.current_holdings["total"]
        current_position_weights = self._get_current_position_weights()

        # Run the 4-layer decision pipeline
        try:
            target_portfolio = self.decision_engine.run_pipeline(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                market_data_for_day=market_data_for_day,
                current_positions=current_position_weights,
            )
        except Exception as e:
            logging.error(f"DecisionEngine pipeline failed: {e}", exc_info=True)
            return

        if not target_portfolio:
            logging.warning("DecisionEngine returned empty target portfolio")
            return

        # Convert target portfolio to trading signals
        self._generate_signals_from_targets(timestamp, target_portfolio)

        # Update tracking
        self.current_targets = target_portfolio.copy()

        target_sum = sum(target_portfolio.values()) if target_portfolio else 0.0

        logging.info(
            f"Rebalancing complete: {len(target_portfolio)} target positions, "
            f"target exposure: ${target_sum:,.0f}"
        )

    def _generate_signals_from_targets(self, timestamp, target_portfolio_values: dict):
        """Convert target portfolio weights to Signal events"""
        current_values = self.portfolio.get_current_position_values()

        all_symbols = set(current_values.keys()) | set(target_portfolio_values.keys())

        for symbol in sorted(all_symbols):
            target_value = target_portfolio_values.get(symbol, 0.0)
            current_value = current_values.get(symbol, 0.0)

            if abs(target_value - current_value) < 100:
                continue

            # Determine signal type
            if target_value <= 0 and current_value > 0:
                signal = SignalEvent(
                    timestamp=timestamp, symbol=symbol, signal_type="Exit"
                )
                self.events.put(signal)
                logging.info(f"  EXIT: {symbol} (current=${current_value:,.0f})")

            elif target_value > 0:
                # We need to send a signal that tells the Portfolio how much *value* to hold.
                # The Portfolio's `on_signal` logic will handle converting this value to shares.
                # We can reuse the 'Long' signal type and use the 'score' to pass the target dollar value.
                signal = SignalEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type="Long",  # Or a new type like 'TargetValue'
                    score=target_value,  # CRITICAL: score now represents target dollar value
                )
                self.events.put(signal)

                action = "INCREASE" if target_value > current_value else "REDUCE"
                logging.info(
                    f"  {action}: {symbol} "
                    f"(${current_value:,.0f} -> ${target_value:,.0f})"
                )

    def _get_current_position_weights(self) -> dict:
        """Calculate current position weights"""
        portfolio_value = self.portfolio.current_holdings["total"]

        if portfolio_value <= 0:
            return {}

        weights = {}
        for symbol in self.symbol_list:
            position_value = self.portfolio.current_holdings.get(symbol, 0.0)
            if position_value > 0:
                weights[symbol] = position_value / portfolio_value

        return weights

    def get_results(self):
        """Get backtest results"""
        results = {
            "holdings": pd.DataFrame(self.portfolio.all_holdings),
            "trade_statistics": self.portfolio.get_trade_statistics(),
            "final_value": self.portfolio.current_holdings["total"],
            "processed_events": self.processed_events,
            "rebalance_count": self.rebalance_count,
        }

        return results

    def get_equity_curve(self):
        """Get equity curve DataFrame"""
        holdings_df = pd.DataFrame(self.portfolio.all_holdings)
        if holdings_df.empty:
            return pd.DataFrame()

        holdings_df["timestamp"] = pd.to_datetime(holdings_df["timestamp"])
        holdings_df.set_index("timestamp", inplace=True)
        holdings_df["returns"] = holdings_df["total"].pct_change()

        return holdings_df

    def _convert_fundamental_data(self, individual_data: pd.DataFrame) -> dict:
        """
        Convert individual stock DataFrame to the format expected by AlphaEngine

        Returns:
            Dict[symbol, dict] with fundamental metrics
        """
        fundamental_dict = {}
        for _, row in individual_data.iterrows():
            symbol = row["symbol"]

            fundamental_dict[symbol] = {
                # Value factors
                "market_cap": row.get("market_cap"),
                "book_value": row.get("total_equity"),
                "earnings": row.get("net_income"),
                # Quality factors
                "revenue": row.get("total_revenue"),
                "cogs": row.get("cost_of_revenue"),
                "total_assets": row.get("total_assets"),
                "book_equity": row.get("total_equity"),
                "roe": row.get("roe"),
                "net_income": row.get("net_income"),
            }

        return fundamental_dict
