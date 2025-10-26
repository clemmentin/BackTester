import itertools
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

import config
from analysis.performance import create_equity_curve_dataframe
from backtester.engine import BacktestEngine
from config import strategy_config
from strategy.simulation.MonteCarlo import MonteCarloSimulator
from analysis.runner import generate_report


@dataclass
class OptimizationResult:
    """Stores the results of a parameter optimization run."""

    best_params: Dict
    best_score: float
    all_results: List[Tuple]
    optimization_time: float
    iterations: int


def calculate_strategy_score(
    equity_curve_df: pd.DataFrame, trade_stats: Dict = None
) -> float:
    """Calculate an enhanced, risk-adjusted score for a strategy's performance.
    A higher score indicates better risk-adjusted returns.

    Args:
      equity_curve_df: pd.DataFrame: 
      trade_stats: Dict:  (Default value = None)

    Returns:

    """
    if equity_curve_df.empty or len(equity_curve_df) < 20:
        return -1.0
    try:
        initial_value = equity_curve_df["total"].iloc[0]
        final_value = equity_curve_df["total"].iloc[-1]
        total_return = (final_value / initial_value) - 1
        start_date = equity_curve_df.index[0]
        end_date = equity_curve_df.index[-1]
        years = max((end_date - start_date).days / 365.25, 0.01)
        annual_return = (1 + total_return) ** (1 / years) - 1
        returns = equity_curve_df["total"].pct_change().dropna()
        if len(returns) < 10 or returns.std() == 0:
            return -1.0
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        cumulative = equity_curve_df["total"] / initial_value
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        calmar = annual_return / max(max_drawdown, 0.01)
        score = 0.5 * sharpe + 0.5 * calmar
        if max_drawdown > 0.25:
            score *= 0.5
        elif max_drawdown > 0.20:
            score *= 0.8
        return max(score, -1.0)
    except Exception as e:
        logging.error(f"Score calculation failed: {e}")
        return -1.0


def objective_function_wrapper(params_data_tuple: Tuple) -> float:
    """A top-level wrapper for multiprocessing. It runs a single backtest and returns its score.

    Args:
      params_data_tuple: Tuple: 

    Returns:

    """
    params, symbols, initial_capital, start_date, risk_on, risk_off, train_data = (
        params_data_tuple
    )
    try:
        engine = BacktestEngine(
            symbol_list=symbols,
            initial_capital=initial_capital,
            all_processed_data=train_data,
            start_date=start_date,
            risk_on_symbols=risk_on,
            risk_off_symbols=risk_off,
            warmup_days=config.WARMUP_PERIOD_DAYS,
            **params,
        )
        engine.run()

        # Check if backtest produced valid results
        if not engine.portfolio.all_holdings:
            logging.warning("Backtest produced no holdings")
            return -1.0

        equity_curve_df = create_equity_curve_dataframe(engine.portfolio.all_holdings)

        if equity_curve_df.empty:
            logging.warning("Empty equity curve")
            return -1.0

        trade_stats = engine.portfolio.get_trade_statistics()
        score = calculate_strategy_score(equity_curve_df, trade_stats)

        if np.isnan(score) or np.isinf(score):
            logging.warning(f"Invalid score calculated: {score}")
            return -1.0

        return score

    except Exception as e:
        logging.error(f"Backtest failed for params {params}: {e}")
        return -1.0


class ParameterOptimizer:
    """A class to handle strategy parameter optimization using different methods."""

    def __init__(self, objective_function, param_grid: Dict[str, List[Any]]):
        self.objective_function = objective_function
        self.param_grid = param_grid
        self.param_names = list(param_grid.keys())

    def optimize_grid_search_parallel(
        self,
        train_data: pd.DataFrame,
        start_date: str,
        max_workers: int = None,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """

        Args:
          train_data: pd.DataFrame: 
          start_date: str: 
          max_workers: int:  (Default value = None)
          show_progress: bool:  (Default value = True)

        Returns:

        """
        start_time = time.time()
        logging.info(
            f"Starting parallel grid search for period starting {start_date}..."
        )

        param_values = list(self.param_grid.values())
        all_combinations = list(itertools.product(*param_values))
        param_combinations_dicts = [
            dict(zip(self.param_names, combo)) for combo in all_combinations
        ]
        total_combinations = len(param_combinations_dicts)
        logging.info(f"Total parameter combinations to test: {total_combinations}")
        if max_workers is None:
            import multiprocessing as mp

            max_workers = min(mp.cpu_count() - 1, 10)

        task_data = [
            (
                params,
                config.SYMBOLS,
                config.INITIAL_CAPITAL,
                start_date,
                config.RISK_ON_SYMBOLS,
                config.RISK_OFF_SYMBOLS,
                train_data,
            )
            for params in param_combinations_dicts
        ]
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(self.objective_function, task): task[0]
                for task in task_data
            }
            completed = 0
            for future in as_completed(future_to_params):
                try:
                    score = future.result(timeout=180)
                    params = future_to_params[future]
                    results.append((params, score))
                except Exception as e:
                    params = future_to_params[future]
                    logging.error(f"Task failed for params {params}: {e}")
                    results.append((params, -np.inf))
                finally:
                    completed += 1
                    if show_progress:
                        print(
                            f"\r  Grid search progress: {completed}/{total_combinations} ({completed / total_combinations * 100:.1f}%)",
                            end="",
                        )
        print()
        end_time = time.time()
        if not results:
            return OptimizationResult(
                None, -np.inf, [], end_time - start_time, total_combinations
            )
        best_params, best_score = max(results, key=lambda item: item[1])
        return OptimizationResult(
            best_params, best_score, results, end_time - start_time, total_combinations
        )

    def optimize_bayesian(
        self,
        train_data: pd.DataFrame,
        start_date: str,
        n_calls: int = 120,
        n_initial_points: int = 25,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """

        Args:
          train_data: pd.DataFrame: 
          start_date: str: 
          n_calls: int:  (Default value = 120)
          n_initial_points: int:  (Default value = 25)
          show_progress: bool:  (Default value = True)

        Returns:

        """
        start_time = time.time()
        logging.info(
            f"Starting Bayesian optimization for period starting {start_date}..."
        )

        # Build search space from param_grid
        dimensions = []
        for param_name, values in self.param_grid.items():
            # Handle categorical parameters
            if param_name == "rebalance_frequency" or isinstance(values[0], str):
                dimensions.append(Categorical(values, name=param_name))
            # Handle float parameters
            elif all(isinstance(v, float) for v in values):
                dimensions.append(Real(min(values), max(values), name=param_name))
            # Handle int parameters
            elif all(isinstance(v, int) for v in values):
                dimensions.append(Integer(min(values), max(values), name=param_name))
            else:
                dimensions.append(Categorical(values, name=param_name))

        # Track results
        all_results = []
        iteration = [0]
        best_score_tracker = [-np.inf]
        valid_scores = []

        # Define objective for skopt
        @use_named_args(dimensions)
        def objective(**params):
            """

            Args:
              **params: 

            Returns:

            """
            iteration[0] += 1

            # Prepare task data
            task_data = (
                params,
                config.SYMBOLS,
                config.INITIAL_CAPITAL,
                start_date,
                config.RISK_ON_SYMBOLS,
                config.RISK_OFF_SYMBOLS,
                train_data,
            )

            try:
                # Evaluate
                score = self.objective_function(task_data)
                if np.isnan(score) or np.isinf(score):
                    logging.warning(f"Invalid score returned: {score}")
                    score = -1.0

                all_results.append((params.copy(), score))

                if score > -1.0:
                    valid_scores.append(score)
                    if score > best_score_tracker[0]:
                        best_score_tracker[0] = score

                if show_progress:
                    print(
                        f"\r  Bayesian opt: {iteration[0]}/{n_calls} | "
                        f"Current: {score:.3f} | Best: {best_score_tracker[0]:.3f}",
                        end="",
                    )

                return -score if score > -np.inf else 1000.0

            except Exception as e:
                logging.error(f"Failed evaluation: {e}")
                all_results.append((params.copy(), -1.0))
                return 1000.0

        try:
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func="EI",
                random_state=config.GLOBAL_RANDOM_SEED,
                verbose=False,
            )

            best_params = dict(zip(self.param_names, result.x))
            best_score = -result.fun if result.fun < 1000 else -1.0

        except ValueError as e:
            logging.error(f"Optimization failed: {e}")
            print()

            if valid_scores:
                best_idx = np.argmax([s for _, s in all_results if s > -1.0])
                best_params, best_score = all_results[best_idx]
            else:
                # Return default parameters if all failed
                best_params = {
                    name: values[0] for name, values in self.param_grid.items()
                }
                best_score = -1.0

        end_time = time.time()
        logging.info(f"Bayesian optimization completed in {end_time - start_time:.1f}s")
        logging.info(f"Best score: {best_score:.3f}")
        logging.info(f"Valid evaluations: {len(valid_scores)}/{n_calls}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=end_time - start_time,
            iterations=n_calls,
        )


def select_robust_parameters(optimization_results: List[Tuple]) -> Dict:
    """Selects robust parameters from optimization results.

    Args:
      optimization_results: List[Tuple]: 

    Returns:

    """
    if not optimization_results:
        return None
    valid_results = [r for r in optimization_results if r[1] > -np.inf]
    if not valid_results:
        logging.warning("No valid parameter combinations found during optimization.")
        return None
    valid_results.sort(key=lambda x: x[1], reverse=True)
    return valid_results[0][0]  # Return the params dictionary


def run_walk_forward_optimization(
    all_available_data: pd.DataFrame, start_date: str = None
) -> Dict[str, Any]:
    """Performs a walk-forward optimization of the strategy.

    Args:
      all_available_data: pd.DataFrame: 
      start_date: str:  (Default value = None)

    Returns:

    """
    logging.info("--- [WFO] Starting Walk-Forward Optimization ---")

    param_grid = strategy_config.OPTIMIZATION_PARAMS.get(
        strategy_config.CURRENT_STRATEGY.lower(), {}
    )
    if not param_grid:
        logging.error("No optimization parameters found. Aborting WFO.")
        return {}

    all_fold_results = []

    wfo_all_optimization_runs = []
    overall_start_date = pd.to_datetime(start_date)
    end_date = all_available_data.index.get_level_values("timestamp").max()
    train_years = config.WFO_TRAIN_PERIOD
    test_years = config.WFO_TEST_PERIOD

    train_start = overall_start_date
    all_oos_holdings = []
    all_oos_trades = []
    USE_BAYESIAN_OPTIMIZATION = True
    fold_capital = config.INITIAL_CAPITAL
    fold_num = 1

    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION CONFIGURATION")
    print("=" * 80)
    print(f"Training Period: {train_years} years")
    print(f"Testing Period: {test_years} year(s)")
    print(
        f"Optimization Method: {'Bayesian' if USE_BAYESIAN_OPTIMIZATION else 'Grid Search'}"
    )
    print(f"Parameters to optimize: {list(param_grid.keys())}")
    print(f"Start Date: {overall_start_date.date()}")
    print(f"End Date: {end_date.date()}")
    print("=" * 80 + "\n")

    while (train_start + relativedelta(years=train_years + test_years)) <= end_date:
        train_end = train_start + relativedelta(years=train_years)
        test_end = train_end + relativedelta(years=test_years)

        print("\n" + "=" * 80)
        print(f"WFO FOLD {fold_num}")
        print("=" * 80)
        clear_bayesian_priors()
        print(f"Training Period:  {train_start.date()} to {train_end.date()}")
        print(f"Testing Period:   {train_end.date()} to {test_end.date()}")
        print(f"Initial Capital:  ${fold_capital:,.2f}")
        print("-" * 80)

        timestamps = all_available_data.index.get_level_values("timestamp")
        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        train_data = all_available_data[train_mask]

        OOS_LOOKBACK_BUFFER = timedelta(days=2000)
        oos_data_start = train_end - OOS_LOOKBACK_BUFFER

        if (
            oos_data_start
            < all_available_data.index.get_level_values("timestamp").min()
        ):
            oos_data_start = all_available_data.index.get_level_values(
                "timestamp"
            ).min()

        test_mask = (timestamps >= oos_data_start) & (timestamps < test_end)
        test_data = all_available_data[test_mask]

        if train_data.empty:
            logging.warning("Insufficient training data for this fold, skipping.")
        else:
            optimizer = ParameterOptimizer(objective_function_wrapper, param_grid)

            if USE_BAYESIAN_OPTIMIZATION:
                opt_result = optimizer.optimize_bayesian(
                    train_data,
                    start_date=train_start.strftime("%Y-%m-%d"),
                    n_calls=50,
                    n_initial_points=10,
                    show_progress=True,
                )
            else:
                opt_result = optimizer.optimize_grid_search_parallel(
                    train_data, start_date=train_start.strftime("%Y-%m-%d")
                )

            if opt_result.best_params:
                print("\n" + "-" * 80)
                print("OPTIMIZATION RESULTS:")
                print("-" * 80)
                print(f"Best Score: {opt_result.best_score:.4f}")
                print(f"Optimization Time: {opt_result.optimization_time:.1f}s")
                print(f"Iterations: {opt_result.iterations}")
                print("\nBest Parameters Found:")
                print("-" * 80)

                for param_name, param_value in sorted(opt_result.best_params.items()):
                    if isinstance(param_value, float):
                        print(f"  {param_name:35s}: {param_value:.4f}")
                    else:
                        print(f"  {param_name:35s}: {param_value}")

                print("-" * 80)

                fold_result = {
                    "fold": fold_num,
                    "train_start": train_start.date(),
                    "train_end": train_end.date(),
                    "test_start": train_end.date(),
                    "test_end": test_end.date(),
                    "optimization_score": opt_result.best_score,
                    "optimization_time": opt_result.optimization_time,
                    **opt_result.best_params,
                }

                oos_engine = BacktestEngine(
                    symbol_list=config.SYMBOLS,
                    initial_capital=fold_capital,
                    all_processed_data=test_data,
                    start_date=train_end.strftime("%Y-%m-%d"),
                    risk_on_symbols=config.RISK_ON_SYMBOLS,
                    risk_off_symbols=config.RISK_OFF_SYMBOLS,
                    warmup_days=config.WARMUP_PERIOD_DAYS,
                    **opt_result.best_params,
                )
                oos_engine.run()

                oos_engine_results = oos_engine.get_results()

                holdings_df = oos_engine_results["holdings"]

                if not holdings_df.empty:
                    initial_fold_val = holdings_df["total"].iloc[0]
                    final_fold_val = holdings_df["total"].iloc[-1]
                    fold_return = (final_fold_val / initial_fold_val) - 1

                    fold_result["oos_initial_value"] = initial_fold_val
                    fold_result["oos_final_value"] = final_fold_val
                    fold_result["oos_return"] = fold_return

                    print("\nOUT-OF-SAMPLE PERFORMANCE:")
                    print("-" * 80)
                    print(f"  Initial Value: ${initial_fold_val:,.2f}")
                    print(f"  Final Value:   ${final_fold_val:,.2f}")
                    print(f"  Return:        {fold_return:+.2%}")
                    print(f"  New Capital:   ${final_fold_val:,.2f}")
                    print("=" * 80 + "\n")

                    scale_factor = fold_capital / initial_fold_val
                    monetary_cols = ["cash", "total"] + [
                        s for s in config.SYMBOLS if s in holdings_df.columns
                    ]
                    for col in monetary_cols:
                        holdings_df[col] *= scale_factor

                    all_oos_holdings.append(holdings_df)
                    all_oos_trades.extend(oos_engine.portfolio.all_trades)
                    fold_capital = holdings_df["total"].iloc[-1]

                all_fold_results.append(fold_result)

            else:
                logging.warning("Optimization failed for this fold.")

        train_start += relativedelta(years=test_years)
        fold_num += 1

    if all_fold_results:
        print("\n" + "=" * 80)
        print("WFO PARAMETER EVOLUTION SUMMARY")
        print("=" * 80)

        results_df = pd.DataFrame(all_fold_results)

        print("\nFold-by-Fold Results:")
        print("-" * 80)

        display_cols = ["fold", "train_start", "test_start", "optimization_score"]

        param_cols = [
            col
            for col in results_df.columns
            if col not in display_cols
            and col
            not in [
                "train_end",
                "test_end",
                "optimization_time",
                "oos_initial_value",
                "oos_final_value",
            ]
        ]

        display_cols.extend(param_cols)
        display_cols.append("oos_return")

        print(results_df[display_cols].to_string(index=False))

        csv_filename = (
            f"wfo_results_{overall_start_date.date()}_to_{end_date.date()}.csv"
        )
        results_df.to_csv(csv_filename, index=False)
        print(f"\n results saved to: {csv_filename}")

        print("\n" + "-" * 80)
        print("PARAMETER STABILITY ANALYSIS:")
        print("-" * 80)

        for param in param_cols:
            if param in results_df.columns:
                values = results_df[param].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()

                    cv = (std_val / mean_val) if mean_val != 0 else 0

                    stability = (
                        "STABLE"
                        if cv < 0.15
                        else ("MODERATE" if cv < 0.30 else "VOLATILE")
                    )

                    print(f"\n{param}:")
                    print(f"  Mean: {mean_val:.4f} | Std: {std_val:.4f} | CV: {cv:.2%}")
                    print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
                    print(f"  Stability: {stability}")

        print("=" * 80 + "\n")

    if not all_oos_holdings:
        logging.error("[WFO] No successful folds. Returning empty results.")
        return {}

    final_holdings = pd.concat(all_oos_holdings)
    final_holdings["returns"] = final_holdings["total"].pct_change()
    final_trades = pd.DataFrame(all_oos_trades)
    if not final_trades.empty:
        wfo_trades_filename = "trade_details_wfo_run.csv"
        final_trades.to_csv(wfo_trades_filename, index=False, encoding="utf-8-sig")

    logging.info(f"[WFO] Final combined out-of-sample results generated.")
    run_and_display_monte_carlo(final_holdings)

    return {
        "holdings": final_holdings,
        "trade_statistics": None,
        "closed_trades": final_trades,
        "wfo_fold_results": all_fold_results,
    }


def run_single_simple_backtest(
    all_available_data: pd.DataFrame, start_date: str = None
) -> Dict[str, Any]:
    """Runs a single backtest using default parameters and returns results with diagnostics.

    Args:
      all_available_data: pd.DataFrame: 
      start_date: str:  (Default value = None)

    Returns:

    """
    logging.info("--- [Backtester] Starting Single Backtest ---")
    params = strategy_config.get_current_strategy_params()

    engine = BacktestEngine(
        symbol_list=config.SYMBOLS,
        initial_capital=config.INITIAL_CAPITAL,
        all_processed_data=all_available_data,
        start_date=start_date,
        risk_on_symbols=config.RISK_ON_SYMBOLS,
        risk_off_symbols=config.RISK_OFF_SYMBOLS,
        warmup_days=config.WARMUP_PERIOD_DAYS,
        **params,
    )
    engine.run()

    results = engine.get_results()
    holdings_df = results["holdings"]
    diagnostics_log = results.get("diagnostics_log", [])

    if holdings_df.empty:
        logging.error("Single backtest produced no results.")
        return {}

    logging.info(
        f"Single backtest completed. Final capital: ${holdings_df['total'].iloc[-1]:,.2f}"
    )
    run_and_display_monte_carlo(holdings_df)

    closed_trades_df = pd.DataFrame(engine.portfolio.all_trades)

    if not closed_trades_df.empty:
        trades_filename = "trade_details_single_run.csv"
        closed_trades_df.to_csv(trades_filename, index=False, encoding="utf-8-sig")

    generate_report(
        holdings_df=holdings_df,
        closed_trades_df=closed_trades_df,
        initial_capital=config.INITIAL_CAPITAL,
        diagnostics_log=diagnostics_log,
    )

    return {
        "holdings": holdings_df,
        "trade_statistics": engine.portfolio.get_trade_statistics(),
        "closed_trades": closed_trades_df,
        "diagnostics_log": diagnostics_log,
    }


def run_and_display_monte_carlo(equity_curve: pd.DataFrame):
    """Runs a Monte Carlo simulation on an equity curve and logs the results.

    Args:
      equity_curve: pd.DataFrame: 

    Returns:

    """
    if (
        equity_curve.empty
        or "returns" not in equity_curve.columns
        or equity_curve["returns"].dropna().empty
    ):
        logging.warning(
            "[Monte Carlo] Equity curve is empty or lacks returns, skipping simulation."
        )
        return
    logging.info("\n" + "=" * 60)
    logging.info("           MONTE CARLO SIMULATION FORECAST")
    logging.info("=" * 60)
    try:
        returns_series = equity_curve["returns"].dropna()
        initial_value = equity_curve["total"].iloc[0]
        mc_sim = MonteCarloSimulator(returns_series)
        price_paths = mc_sim.simulate_price_paths(
            initial_price=initial_value, num_days=252, num_simulations=1000
        )
        risk_metrics = mc_sim.calculate_risk_metrics(price_paths)
        final_value_median = np.median(price_paths[:, -1])
        logging.info(
            f"Based on historical performance from {equity_curve.index[0].date()} to {equity_curve.index[-1].date()}:"
        )
        logging.info(f"  - Initial Portfolio Value: ${initial_value:,.2f}")
        logging.info(f"  - Forecast Horizon: 1 Year (252 trading days)")
        logging.info("-" * 60)
        logging.info(
            "  - Median Forecasted Final Value: ${:,.2f}".format(final_value_median)
        )
        logging.info(
            "  - Expected Annual Return: {:.2f}%".format(
                risk_metrics["expected_return"] * 100
            )
        )
        logging.info(
            "  - Annualized Volatility: {:.2f}%".format(
                risk_metrics["volatility"] * 100
            )
        )
        logging.info(
            "  - Probability of Loss: {:.2f}%".format(
                risk_metrics["probability_of_loss"] * 100
            )
        )
        logging.info("-" * 60)
        logging.info("  RISK ASSESSMENT:")
        logging.info(
            "  - 95% Value-at-Risk (VaR): {:.2f}% (There is a 5% chance of losing at least ${:,.2f})".format(
                risk_metrics["var_95"] * 100,
                abs(initial_value * risk_metrics["var_95"]),
            )
        )
        logging.info(
            "  - 95% Conditional VaR (CVaR): {:.2f}% (In the worst 5% of scenarios, the average loss is ${:,.2f})".format(
                risk_metrics["cvar_95"] * 100,
                abs(initial_value * risk_metrics["cvar_95"]),
            )
        )
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"[Monte Carlo] Simulation failed: {e}")


def clear_bayesian_priors(directory="./data/bayesian_priors"):
    """Deletes all .json files in the priors directory to ensure a clean slate.

    Args:
      directory:  (Default value = "./data/bayesian_priors")

    Returns:

    """
    import os
    import glob

    if os.path.exists(directory):
        files = glob.glob(os.path.join(directory, "*.json"))
        if not files:
            print("Bayesian priors directory is already clean.")
            return

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                logging.error(f"Error removing prior file {f}: {e}")
        print(f"Cleared {len(files)} file(s) from the Bayesian priors directory.")
    else:
        print("Bayesian priors directory does not exist, no cleanup needed.")
