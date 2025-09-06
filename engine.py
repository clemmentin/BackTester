import queue
import logging
import numpy as np
import pandas as pd
from strategy import VolatilityTargetingStrategy
from portfolio import Portfolio
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from performance import create_equity_curve_dataframe, calculate_performance_metrics, plot_equity_curve


class BacktestEngine:
    def __init__(self, csv_dir, symbol_list,signal_csv_path, initial_capital,buy_threshold, sell_threshold,rsi_threshold):
        self.events = queue.Queue()
        self.csv_dir = csv_dir
        self.signal_csv_path = signal_csv_path
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.data_handler = HistoricCSVDataHandler(self.events, self.csv_dir, self.symbol_list,self.signal_csv_path)
        self.strategy = VolatilityTargetingStrategy(self.events,self.symbol_list,buy_threshold=buy_threshold,sell_threshold=sell_threshold,rsi_threshold = rsi_threshold)
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
def main():
    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename = 'optimization_log.log',
        filemode = 'w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    csv_directory = './data'
    symbols = ['SPY']
    signal_file = './data//VOL_PLOT.csv'
    capital = 100000.0
    buy = np.arange(16.0,30.0,4.0)
    sell = np.arange(26.0,40.0,4.0)
    rsi_thr = np.arange(40, 60, 2)
    all_results = []
    best_sharpe = -np.inf
    best_params = {}
    logging.info("--- Starting Parameter Optimization ---")
    num_combinations = len(buy) * len(sell)
    logging.info(f"Testing up to {num_combinations} combinations...")
    run_count = 0
    for rsi_thresh in rsi_thr:
        for buy_thresh in buy:
            for sell_thresh in sell:
                if sell_thresh <= buy_thresh:
                    continue
                run_count += 1
                log_header = f"[Run {run_count}/{num_combinations} | Buy < {buy_thresh}, Sell > {sell_thresh}]"
                # logging.info(f"{log_header} - Starting backtest...")
                engine = BacktestEngine(csv_dir=csv_directory,
                                        symbol_list=symbols,
                                        signal_csv_path=signal_file,
                                        initial_capital=capital,
                                        buy_threshold=buy_thresh,
                                        sell_threshold=sell_thresh,
                                        rsi_threshold = rsi_thresh)
                engine.run()
                equity_curve = create_equity_curve_dataframe(engine.portfolio.all_holdings)
                if not equity_curve.empty and equity_curve['returns'].std() != 0:
                    sharpe_ratio = (equity_curve['returns'].mean() / equity_curve['returns'].std()) * np.sqrt(252)
                    total_return = (equity_curve['total'].iloc[-1] / equity_curve['total'].iloc[0]) - 1
                else:
                    sharpe_ratio = 0.0
                    total_return = -1.0
                if run_count % 10 == 0:
                    logging.info(
                        f"{log_header} - Finished. Sharpe: {sharpe_ratio:.2f}, Total Return: {total_return * 100:.2f}%")
                current_run_results = {
                    'buy_threshold': buy_thresh,
                    'sell_threshold': sell_thresh,
                    'sharpe_ratio': sharpe_ratio
                }
                all_results.append(current_run_results)
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_params = {'buy': buy_thresh, 'sell': sell_thresh,'rsi':rsi_thresh}
    logging.info("--- Optimization Complete ---")
    results_df = pd.DataFrame(all_results)
    print("\n\n--- Optimization Complete: Final Report ---")
    print("\nAll Parameter Test Results (Top 10 by Sharpe Ratio):")
    print(results_df.sort_values(by='sharpe_ratio', ascending=False).head(10).round(2))
    print("\n--- Best Performing Parameters ---")
    print(f"Optimal Buy Threshold: {best_params.get('buy')}")
    print(f"Optimal Sell Threshold: {best_params.get('sell')}")
    print(f"Optimal RSI Threshold: {best_params.get('rsi')}")
    print(f"Resulting Best Sharpe Ratio: {best_sharpe:.2f}")
    if best_params:
        print("\n--- Running Final Backtest with Best Parameters ---")
        logging.info("--- Running Final Backtest with Best Parameters ---")
        final_engine = BacktestEngine(csv_dir=csv_directory,
                                      symbol_list=symbols,
                                      signal_csv_path=signal_file,
                                      initial_capital=capital,
                                      buy_threshold=best_params['buy'],
                                      sell_threshold=best_params['sell'],
                                      rsi_threshold = best_params['rsi'])
        final_engine.run()
        print("--- Backtest Finished ---")
        print("\n--- Final Performance Analysis ---")
        logging.info("--- Final Performance Analysis ---")
        final_equity_curve = create_equity_curve_dataframe(final_engine.portfolio.all_holdings)
        if not final_equity_curve.empty:
            final_metrics = calculate_performance_metrics(final_equity_curve)
            print("\nPerformance Metrics:")
            for metric, value in final_metrics.items():
                print(f"- {metric}: {value}")
                logging.info(f"- {metric}: {value}")
            print("\nDisplaying equity curve plot...")
            plot_equity_curve(final_equity_curve, title='Performance with Optimal Parameters')
        else:
            print("Could not generate performance metrics because the equity curve is empty.")
            logging.warning("Could not generate performance metrics because the equity curve is empty.")
    else:
        print("\nNo best parameters found, skipping final analysis.")
        logging.warning("No best parameters found, skipping final analysis.")

if __name__ == "__main__":
    main()