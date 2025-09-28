import logging
import queue

import pandas as pd

import config
from backtester.data import HistoricDBDataHandler
from backtester.execution import SimulatedExecutionHandler
from backtester.portfolio import Portfolio


class BacktestEngine:
    def __init__(
        self,
        symbol_list,
        initial_capital,
        all_processed_data,
        start_date=None,
        risk_on_symbols=None,
        risk_off_symbols=None,
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

        # Initialize strategy
        strategy_name = strategy_kwargs.pop("name", config.CURRENT_STRATEGY)

        self.strategy = config.create_strategy(
            self.events, self.symbol_list, name=strategy_name, **strategy_kwargs
        )

        # Set portfolio reference if strategy supports it
        if hasattr(self.strategy, "set_portfolio_reference"):
            self.strategy.set_portfolio_reference(self.portfolio)
            logging.info("Portfolio reference set in strategy")

        # Initialize execution handler
        self.execution_handler = SimulatedExecutionHandler(
            self.events, self.data_handler
        )

        # Statistics
        self.processed_events = 0

        logging.info(
            f"BacktestEngine initialized for {len(symbol_list)} symbols with ${initial_capital:,.0f}"
        )

    def run(self):
        """Run the backtest"""
        logging.info("Starting backtest...")

        event_counts = {"Market": 0, "Signal": 0, "Order": 0, "Fill": 0}

        while self.data_handler.continue_backtest:
            # Update market data
            self.data_handler.update_bars()

            # Process event queue
            while True:
                try:
                    event = self.events.get(block=False)
                except queue.Empty:
                    break

                if event is not None:
                    self.processed_events += 1
                    event_type = event.type

                    # Count events
                    if event_type in event_counts:
                        event_counts[event_type] += 1

                    # Route events to handlers
                    if event_type == "Market":
                        self.strategy.calculate_signal(event)
                        self.portfolio.update_timeindex(event)

                        # Safety check
                        if event_counts["Market"] % 1000 == 0:
                            total_value = self.portfolio.current_holdings["total"]
                            if total_value <= 1000:
                                logging.warning(
                                    f"Low capital: ${total_value:.2f}, stopping"
                                )
                                self.data_handler.continue_backtest = False
                                break

                    elif event_type == "Signal":
                        self.portfolio.on_signal(event)

                    elif event_type == "Order":
                        self.execution_handler.execute_order(event)

                    elif event_type == "Fill":
                        self.portfolio.on_fill(event)

        # Log summary
        logging.info("Backtest completed")
        logging.info(f"Processed events: {self.processed_events}")
        for event_type, count in event_counts.items():
            logging.info(f"  {event_type}: {count}")

        # Final results
        final_value = self.portfolio.current_holdings["total"]
        total_return = (final_value / self.initial_capital - 1) * 100
        logging.info(f"Final value: ${final_value:,.2f} (Return: {total_return:.2f}%)")

    def get_results(self):
        """Get backtest results"""
        results = {
            "holdings": pd.DataFrame(self.portfolio.all_holdings),
            "trade_statistics": self.portfolio.get_trade_statistics(),
            "final_value": self.portfolio.current_holdings["total"],
            "processed_events": self.processed_events,
        }

        if hasattr(self.portfolio, "stop_manager"):
            results["stop_stats"] = (
                self.portfolio.stop_manager.get_effectiveness_metrics()
            )

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
