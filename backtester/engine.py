import logging
import queue
import pandas as pd
import numpy as np
import config.trading_parameters as tp

from backtester.data import HistoricDBDataHandler
from backtester.execution import SimulatedExecutionHandler
from backtester.portfolio import Portfolio
from backtester.events import SignalEvent
from strategy.decision_engine import DecisionEngine
from analysis.performance import create_equity_curve_dataframe


class BacktestEngine:
    """ """

    def __init__(
        self,
        symbol_list,
        initial_capital,
        all_processed_data,
        start_date=None,
        risk_on_symbols=None,
        risk_off_symbols=None,
        data_manager=None,
        warmup_days=0,
        **strategy_kwargs,
    ):
        self.events = queue.Queue()
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital

        self.data_handler = HistoricDBDataHandler(
            self.events,
            self.symbol_list,
            all_processed_data,
            start_date=start_date,
            warmup_days=warmup_days,
        )

        self.portfolio = Portfolio(
            self.events,
            self.symbol_list,
            self.data_handler,
            self.initial_capital,
            all_processed_data,
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

        self.decision_engine = DecisionEngine(
            initial_capital=initial_capital,
            symbol_list=symbol_list,
            all_processed_data=all_processed_data,
            data_manager=None,
            **strategy_kwargs,
        )

        self.execution_handler = SimulatedExecutionHandler(
            self.events,
            self.data_handler,
            tp.TRADING_PARAMS.get("transaction_costs", {}),
        )

        self.rebalance_frequency = strategy_kwargs.get("rebalance_frequency", "weekly")
        self.last_rebalance_date = None
        self.current_targets = {}
        self.current_stop_targets = {}  # Cache for dynamic stop percentages
        self.min_signal_weight = 0.01
        self.processed_events = 0
        self.rebalance_count = 0
        self.diagnostics_log = []
        # Load tactical mode configuration
        tm_config = strategy_kwargs.get("tactical_mode", {})
        self.tactical_mode_enabled = tm_config.get("enabled", False)
        self.strategic_rebalance_day = tm_config.get("strategic_rebalance_day", 0)
        self.tactical_entry_enabled = tm_config.get("tactical_entry_enabled", False)
        self.tactical_min_ev = tm_config.get("tactical_min_ev_threshold", 0.02)
        self.tactical_min_conf = tm_config.get(
            "tactical_min_confidence_threshold", 0.85
        )
        self.tactical_cash_pct = tm_config.get("tactical_cash_deployment_pct", 0.25)

        # Load dynamic stop config
        self.dyn_stop_config = tp.RISK_PARAMS.get("stop_loss", {}).get(
            "dynamic_stop", {}
        )
        self.dyn_stop_enabled = self.dyn_stop_config.get("enabled", False)

        logging.info(
            f"BacktestEngine initialized with DecisionEngine "
            f"({self.rebalance_frequency} rebalancing). Dynamic Stops: {'ENABLED' if self.dyn_stop_enabled else 'DISABLED'}"
        )

    def _is_strategic_day(self, current_date) -> bool:
        """Checks if it's the designated day for a full rebalance."""
        if self.last_rebalance_date is None:
            return True  # First day is always a strategic day

        is_correct_weekday = current_date.weekday() == self.strategic_rebalance_day
        days_since_last = (current_date - self.last_rebalance_date).days

        return is_correct_weekday and days_since_last >= 5

    def run(self):
        """ """
        logging.info("Starting backtest with DecisionEngine...")
        event_counts = {"Market": 0, "Signal": 0, "Order": 0, "Fill": 0}
        warmup_event_count = 0

        while self.data_handler.continue_backtest:
            if self.data_handler.current_time_index >= len(self.data_handler.timeline):
                break

            current_timestamp = self.data_handler.timeline[
                self.data_handler.current_time_index
            ]
            current_date = current_timestamp.date()
            self.data_handler.update_bars()
            todays_market_data_dict = {}

            # === WARMUP PHASE ===
            is_warmup_phase = not self.data_handler.warmup_complete

            while True:
                try:
                    event = self.events.get(block=False)
                except queue.Empty:
                    break

                if event is not None:
                    if is_warmup_phase:
                        warmup_event_count += 1
                        if event.type == "Market":
                            self.portfolio.update_timeindex(event)
                            todays_market_data_dict[event.symbol] = event.data
                        elif event.type == "Order":
                            self.execution_handler.execute_order(event)
                        elif event.type == "Fill":
                            self.portfolio.on_fill(event)
                            if event.direction == "Sell" and hasattr(
                                self.decision_engine, "stop_manager"
                            ):
                                self.decision_engine.stop_manager.remove_position(
                                    event.symbol
                                )
                    else:
                        # Normal trading phase
                        self.processed_events += 1
                        event_counts[event.type] = event_counts.get(event.type, 0) + 1

                        if event.type == "Market":
                            self.portfolio.update_timeindex(event)
                            todays_market_data_dict[event.symbol] = event.data

                        elif event.type == "Signal":
                            self.portfolio.on_signal(event)
                        elif event.type == "Order":
                            self.execution_handler.execute_order(event)
                        elif event.type == "Fill":
                            self.portfolio.on_fill(event)

                            if event.direction == "Sell":
                                if hasattr(self.decision_engine, "stop_manager"):
                                    self.decision_engine.stop_manager.remove_position(
                                        event.symbol
                                    )
                                    logging.debug(
                                        f"Removed {event.symbol} from StopManager after sell confirmation."
                                    )

                            if event.direction == "Buy":
                                entry_price = (
                                    event.fill_cost / event.quantity
                                    if event.quantity > 0
                                    else 0
                                )
                                if entry_price > 0:
                                    # Inform Bayesian learner
                                    self.decision_engine.alpha_engine.record_signal_entry(
                                        symbol=event.symbol,
                                        entry_price=entry_price,
                                        timestamp=event.timestamp,
                                    )
                                    # Add position to StopManager with dynamic stop loss
                                    stop_pct = self.current_stop_targets.get(
                                        event.symbol
                                    )
                                    # This assumes DecisionEngine has a 'stop_manager' attribute
                                    if hasattr(self.decision_engine, "stop_manager"):
                                        self.decision_engine.stop_manager.add_position(
                                            symbol=event.symbol,
                                            entry_price=entry_price,
                                            stop_loss_pct=stop_pct,
                                        )

            # === WARMUP REBALANCING ===
            if (
                is_warmup_phase
                and todays_market_data_dict
                and (
                    self.last_rebalance_date is None
                    or (current_date - self.last_rebalance_date).days >= 7
                )
            ):
                market_df_for_day = pd.DataFrame.from_dict(
                    todays_market_data_dict, orient="index"
                )
                self._execute_warmup_update(current_timestamp, market_df_for_day)
                self.last_rebalance_date = current_date

            # === NORMAL TRADING REBALANCING ===
            if not is_warmup_phase and todays_market_data_dict:
                market_df_for_day = pd.DataFrame.from_dict(
                    todays_market_data_dict, orient="index"
                )

                if self._is_strategic_day(current_date):
                    self._execute_strategic_rebalance(
                        current_timestamp, market_df_for_day
                    )
                    self.last_rebalance_date = current_date
                    self.rebalance_count += 1
                elif self.tactical_mode_enabled:
                    self._execute_tactical_update(current_timestamp, market_df_for_day)

        logging.info("Backtest completed. Finalizing learning process...")
        self.decision_engine.alpha_engine.shutdown()
        self.decision_engine.alpha_engine.finalize_learning()
        logging.info("Learning finalized and priors saved.")

        logging.info(f"Warmup events processed: {warmup_event_count}")
        logging.info(f"Trading events processed: {self.processed_events}")
        logging.info(f"Rebalances: {self.rebalance_count}")
        for event_type, count in event_counts.items():
            logging.info(f"  {event_type}: {count}")

        final_value = self.portfolio.current_holdings["total"]
        total_return = (final_value / self.initial_capital - 1) * 100
        logging.info(f"Final value: ${final_value:,.2f} (Return: {total_return:.2f}%)")

    def _calculate_dynamic_stops(self, final_signals, market_data_for_day):
        """Calculates stop loss percentage based on E[Loss] and ATR.

        Args:
          final_signals:
          market_data_for_day:

        Returns:

        """
        if not self.dyn_stop_enabled:
            return {}

        eloss_w = self.dyn_stop_config.get("eloss_weight", 0.5)
        atr_w = self.dyn_stop_config.get("atr_weight", 0.5)
        atr_mult = self.dyn_stop_config.get("atr_multiplier", 2.0)
        min_stop = self.dyn_stop_config.get("min_stop_pct", 0.04)
        max_stop = self.dyn_stop_config.get("max_stop_pct", 0.18)

        stop_map = {}
        for symbol, signal in final_signals.items():
            # Bayesian Expected Loss component
            expected_loss = getattr(
                signal, "expected_loss", tp.RISK_PARAMS["stop_loss"]["fixed_stop_loss"]
            )

            # ATR Volatility component
            price = market_data_for_day.loc[symbol, "close"]
            atr = market_data_for_day.loc[symbol, "atr_14"]

            if price > 0 and atr > 0:
                atr_stop_pct = (atr * atr_mult) / price
            else:
                atr_stop_pct = tp.RISK_PARAMS["stop_loss"]["fixed_stop_loss"]

            # Combine the two components
            combined_stop = (expected_loss * eloss_w) + (atr_stop_pct * atr_w)

            # Clip to min/max boundaries
            final_stop_pct = np.clip(combined_stop, min_stop, max_stop)
            stop_map[symbol] = final_stop_pct

        return stop_map

    def _execute_warmup_update(self, timestamp, market_data_for_day):
        """

        Args:
          timestamp:
          market_data_for_day:

        Returns:

        """
        logging.debug(f"WARMUP: Updating model states at {timestamp.date()}")

        current_prices = market_data_for_day["close"].to_dict()
        self.decision_engine.alpha_engine.evaluate_signal_outcomes(
            current_prices=current_prices,
            timestamp=timestamp,
            market_data=self.decision_engine.all_data,
        )

        portfolio_value = self.portfolio.current_holdings["total"]
        current_position_weights = self._get_current_position_weights()
        available_cash = self.portfolio.current_holdings["cash"]
        current_position_values = self.portfolio.get_current_position_values()

        try:
            _, _ = self.decision_engine.run_pipeline(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                market_data_for_day=market_data_for_day,
                current_positions=current_position_weights,
                available_cash=available_cash,
                current_position_values=current_position_values,
            )
        except Exception as e:
            logging.warning(f"Warmup pipeline update failed at {timestamp.date()}: {e}")

    def _execute_strategic_rebalance(self, timestamp, market_data_for_day):
        """

        Args:
          timestamp:
          market_data_for_day:

        Returns:

        """
        logging.info(f"\n{'=' * 60}")
        logging.info(f"REBALANCING at {timestamp.date()}")
        logging.info(f"{'=' * 60}")

        for symbol in self.symbol_list:
            if symbol in market_data_for_day.index:
                quantity = self.portfolio.current_positions.get(symbol, 0)
                if quantity > 0:
                    current_price = market_data_for_day.loc[symbol, "close"]
                    self.portfolio.current_holdings[symbol] = current_price * quantity

        positions_value = sum(
            self.portfolio.current_holdings.get(s, 0) for s in self.symbol_list
        )
        self.portfolio.current_holdings["total"] = (
            self.portfolio.current_holdings["cash"] + positions_value
        )

        current_prices = market_data_for_day["close"].to_dict()
        self.decision_engine.alpha_engine.evaluate_signal_outcomes(
            current_prices=current_prices,
            timestamp=timestamp,
            market_data=self.decision_engine.all_data,
        )

        portfolio_value = self.portfolio.current_holdings["total"]
        available_cash = self.portfolio.current_holdings["cash"]
        current_position_weights = self._get_current_position_weights()
        current_position_values = self.portfolio.get_current_position_values()

        try:
            target_portfolio, diagnostics = self.decision_engine.run_pipeline(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                market_data_for_day=market_data_for_day,
                current_positions=current_position_weights,
                available_cash=available_cash,
                current_position_values=current_position_values,
            )
        except Exception as e:
            logging.error(f"DecisionEngine pipeline failed: {e}", exc_info=True)
            return

        # Calculate and cache dynamic stop targets for this rebalance cycle
        final_signals = diagnostics.get("final_signals", {})
        self.current_stop_targets = self._calculate_dynamic_stops(
            final_signals, market_data_for_day
        )

        diagnostic_entry = {
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "final_position_count": len(target_portfolio),
            "final_position_value": sum(target_portfolio.values()),
            **diagnostics,
        }
        self.diagnostics_log.append(diagnostic_entry)

        self._generate_signals_from_targets(timestamp, target_portfolio)
        self.current_targets = target_portfolio.copy()

        target_sum = sum(target_portfolio.values()) if target_portfolio else 0.0
        logging.info(
            f"Rebalancing complete: {len(target_portfolio)} target positions, "
            f"target exposure: ${target_sum:,.0f}"
        )

    def _generate_signals_from_targets(self, timestamp, target_portfolio_values: dict):
        """

        Args:
          timestamp:
          target_portfolio_values: dict:
          target_portfolio_values: dict:
          target_portfolio_values: dict:
          target_portfolio_values: dict:

        Returns:

        """
        current_values = self.portfolio.get_current_position_values()
        all_symbols = set(current_values.keys()) | set(target_portfolio_values.keys())

        for symbol in sorted(all_symbols):
            target_value = target_portfolio_values.get(symbol, 0.0)
            current_value = current_values.get(symbol, 0.0)
            if abs(target_value - current_value) < 100:
                continue
            if target_value <= 0 and current_value > 0:
                signal = SignalEvent(
                    timestamp=timestamp, symbol=symbol, signal_type="Exit"
                )
                self.events.put(signal)
            elif target_value > 0:
                signal = SignalEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type="Long",
                    score=target_value,
                )
                self.events.put(signal)

    def _get_current_position_weights(self) -> dict:
        """ """
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
        """ """
        data_handler = self.data_handler

        processed_holdings = create_equity_curve_dataframe(self.portfolio.all_holdings)

        if processed_holdings.empty:
            return {
                "holdings": pd.DataFrame(),
                "trade_statistics": self.portfolio.get_trade_statistics(),
                "final_value": self.initial_capital,
                "processed_events": self.processed_events,
                "rebalance_count": self.rebalance_count,
                "diagnostics_log": self.diagnostics_log,
            }

        if data_handler.timeline and data_handler.warmup_days > 0:
            warmup_end_timestamp = data_handler.timeline[
                data_handler.trading_start_index
            ]

            final_holdings = processed_holdings.loc[
                processed_holdings.index >= warmup_end_timestamp
            ].copy()

            num_removed = len(processed_holdings) - len(final_holdings)
            logging.info(
                f"Results: Excluded {num_removed} records due to Warmup (before {warmup_end_timestamp.date()})."
            )
        else:
            final_holdings = processed_holdings

        return {
            "holdings": final_holdings,
            "trade_statistics": self.portfolio.get_trade_statistics(),
            "final_value": self.portfolio.current_holdings["total"],
            "processed_events": self.processed_events,
            "rebalance_count": self.rebalance_count,
            "diagnostics_log": self.diagnostics_log,
        }

    def _execute_tactical_update(self, timestamp, market_data_for_day):
        """
        On tactical days, performs defensive sells (stops) and highly selective buys.
        """
        logging.debug(f"TACTICAL UPDATE at {timestamp.date()}")
        current_prices = market_data_for_day["close"].to_dict()
        current_atr = market_data_for_day["atr_14"].to_dict()

        # --- 1. DEFENSE: Check for stop-loss triggers ---
        if hasattr(self.decision_engine, "stop_manager"):
            stop_signals = self.decision_engine.stop_manager.update_stops(
                current_prices, current_atr, timestamp
            )
            for stop_signal in stop_signals:
                logging.info(
                    f"TACTICAL STOP: {stop_signal.action.upper()} for {stop_signal.symbol}. Reason: {stop_signal.reason}"
                )

                if stop_signal.action == "reduce_half":
                    current_value = self.portfolio.get_current_position_values().get(
                        stop_signal.symbol, 0
                    )
                    target_value = current_value / 2.0

                    reduce_signal = SignalEvent(
                        timestamp=timestamp,
                        symbol=stop_signal.symbol,
                        signal_type="Long",
                        score=target_value,
                    )
                    self.events.put(reduce_signal)

                else:
                    exit_signal = SignalEvent(
                        timestamp=timestamp,
                        symbol=stop_signal.symbol,
                        signal_type="Exit",
                    )
                    self.events.put(exit_signal)

    def _convert_fundamental_data(self, individual_data: pd.DataFrame) -> dict:
        """

        Args:
          individual_data: pd.DataFrame:
          individual_data: pd.DataFrame:
          individual_data: pd.DataFrame:
          individual_data: pd.DataFrame:

        Returns:

        """
        fundamental_dict = {}
        for _, row in individual_data.iterrows():
            symbol = row["symbol"]
            fundamental_dict[symbol] = {
                "market_cap": row.get("market_cap"),
                "book_value": row.get("total_equity"),
                "earnings": row.get("net_income"),
                "revenue": row.get("total_revenue"),
                "cogs": row.get("cost_of_revenue"),
                "total_assets": row.get("total_assets"),
                "book_equity": row.get("total_equity"),
                "roe": row.get("roe"),
                "net_income": row.get("net_income"),
            }
        return fundamental_dict
