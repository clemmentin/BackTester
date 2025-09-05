import queue
from strategy import DummyStrategy,VolatilityTargetingStrategy
from portfolio import Portfolio
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from performance import create_equity_curve_dataframe, calculate_performance_metrics, plot_equity_curve


class BacktestEngine:
    def __init__(self, csv_dir, symbol_list,signal_csv_path, initial_capital,buy_threshold, sell_threshold):
        self.events = queue.Queue()
        self.csv_dir = csv_dir
        self.signal_csv_path = signal_csv_path
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.data_handler = HistoricCSVDataHandler(self.events, self.csv_dir, self.symbol_list,self.signal_csv_path)
        self.strategy = VolatilityTargetingStrategy(self.events,self.symbol_list,buy_threshold=buy_threshold,sell_threshold=sell_threshold)
        self.portfolio = Portfolio(self.events,self.symbol_list,self.data_handler,self.initial_capital)
        self.execution_handler = SimulatedExecutionHandler(self.events, self.data_handler)
    def run(self):
        print("--- Backtest Starting ---")
        while self.data_handler.continue_backtest:
            self.data_handler.update_bars()
            while True:
                try:
                    event = self.events.get(block = False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'Market':
                            print(f"ENGINE:   Dispatching MARKET event for {event.symbol}")
                            self.strategy.calculate_signal(event)
                            self.portfolio.update_timeindex(event)
                        elif event.type == 'Signal':
                            print(f"ENGINE:   Dispatching SIGNAL event for {event.symbol}")
                            self.portfolio.on_signal(event)
                        elif event.type == 'Order':
                            print(f"ENGINE:   Dispatching ORDER event for {event.symbol}")
                            self.execution_handler.execute_order(event)
                        elif event.type == 'Fill':
                            self.portfolio.on_fill(event)
        print("--- Backtest Finished ---")
if __name__ == "__main__":
    csv_directory = './data'
    symbols = ['SPY']
    signal_file = './data//VOL_PLOT.csv'
    capital = 100000.0
    buy = 17
    sell= 27
    engine = BacktestEngine(csv_dir = csv_directory, symbol_list=symbols, signal_csv_path=signal_file,initial_capital=capital,buy_threshold=buy,sell_threshold=sell)
    engine.run()
    print("\n--- Generating Performance Report ---")
    equity_curve = create_equity_curve_dataframe(engine.portfolio.all_holdings)
    metrics = calculate_performance_metrics(equity_curve)
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    print("\nPlotting equity curve...")
    plot_equity_curve(equity_curve, title=f"Volatility Targeting Strategy on {symbols[0]}")
    print("--- Report Generation Complete ---")