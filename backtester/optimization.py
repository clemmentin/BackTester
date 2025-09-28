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
    """
    Calculate an enhanced, risk-adjusted score for a strategy's performance.
    A higher score indicates better risk-adjusted returns.
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
    """
    A top-level wrapper for multiprocessing. It runs a single backtest and returns its score.
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
        n_calls: int = 40,
        n_initial_points: int = 8,
        show_progress: bool = True,
    ) -> OptimizationResult:
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
                random_state=42,
                verbose=False,
            )

            print()  # New line

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
    """Selects robust parameters from optimization results."""
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
    """Performs a walk-forward optimization of the strategy."""
    logging.info("--- [WFO] Starting Walk-Forward Optimization ---")
    param_grid = strategy_config.OPTIMIZATION_PARAMS.get(
        strategy_config.CURRENT_STRATEGY.lower(), {}
    )
    if not param_grid:
        logging.error(
            "No optimization parameters found for the current strategy. Aborting WFO."
        )
        return {}

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
    while (train_start + relativedelta(years=train_years + test_years)) <= end_date:
        train_end = train_start + relativedelta(years=train_years)
        test_end = train_end + relativedelta(years=test_years)
        logging.info(
            f"\n===== WFO Fold {fold_num} | Train: {train_start.date()}-{train_end.date()} | Test: {train_end.date()}-{test_end.date()} ====="
        )

        timestamps = all_available_data.index.get_level_values("timestamp")
        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        train_data = all_available_data[train_mask]

        test_mask = (timestamps >= train_start) & (timestamps < test_end)
        test_data = all_available_data[test_mask]

        if train_data.empty:
            logging.warning("Insufficient training data for this fold, skipping.")
        else:
            optimizer = ParameterOptimizer(objective_function_wrapper, param_grid)
            if USE_BAYESIAN_OPTIMIZATION:
                # Use Bayesian optimization (MUCH faster)
                opt_result = optimizer.optimize_bayesian(
                    train_data,
                    start_date=train_start.strftime("%Y-%m-%d"),
                    n_calls=80,
                    n_initial_points=15,
                    show_progress=True,
                )
            else:
                opt_result = optimizer.optimize_grid_search_parallel(
                    train_data, start_date=train_start.strftime("%Y-%m-%d")
                )

            if opt_result.best_params:
                logging.info(
                    f"Best params found: {opt_result.best_params} (Score: {opt_result.best_score:.3f})"
                )
                oos_engine = BacktestEngine(
                    symbol_list=config.SYMBOLS,
                    initial_capital=fold_capital,
                    all_processed_data=test_data,
                    start_date=train_end.strftime("%Y-%m-%d"),
                    risk_on_symbols=config.RISK_ON_SYMBOLS,
                    risk_off_symbols=config.RISK_OFF_SYMBOLS,
                    **opt_result.best_params,
                )
                oos_engine.run()
                holdings_df = pd.DataFrame(oos_engine.portfolio.all_holdings)
                if not holdings_df.empty:
                    oos_results = (
                        holdings_df.copy()
                    )  # No need to filter again as engine starts at the right date
                    if not oos_results.empty:
                        initial_fold_val = oos_results["total"].iloc[0]
                        scale_factor = fold_capital / initial_fold_val
                        monetary_cols = ["cash", "total"] + [
                            s for s in config.SYMBOLS if s in oos_results.columns
                        ]
                        for col in monetary_cols:
                            oos_results[col] *= scale_factor

                        all_oos_holdings.append(oos_results)
                        all_oos_trades.extend(
                            oos_engine.portfolio.all_trades
                        )  # Collect trades
                        fold_capital = oos_results["total"].iloc[-1]
                        logging.info(
                            f"Fold {fold_num} complete. New capital: ${fold_capital:,.2f}"
                        )
            else:
                logging.warning(
                    "Optimization failed to find best parameters for this fold."
                )
        train_start += relativedelta(years=test_years)
        fold_num += 1
    if wfo_all_optimization_runs:
        logging.info(
            "\n--- [WFO] Saving detailed optimization results to CSV file... ---"
        )
        results_df = pd.DataFrame(wfo_all_optimization_runs)
        param_cols = list(param_grid.keys())
        other_cols = [col for col in results_df.columns if col not in param_cols]
        results_df = results_df[other_cols + param_cols]
        results_df.to_csv("wfo_optimization_results.csv", index=False)
        logging.info("Detailed results saved to 'wfo_optimization_results.csv'")

    if not all_oos_holdings:
        logging.error("[WFO] No successful folds. Returning empty results.")
        return {}

    final_holdings = (
        pd.concat(all_oos_holdings).reset_index(drop=True).set_index("timestamp")
    )
    final_holdings["returns"] = final_holdings["total"].pct_change()
    final_trades = pd.DataFrame(all_oos_trades)
    logging.info(f"[WFO] Final combined out-of-sample results generated.")
    run_and_display_monte_carlo(final_holdings)

    return {
        "holdings": final_holdings,
        "trade_statistics": None,  # WFO trade stats are complex, can be calculated in analysis
        "closed_trades": final_trades,
    }


def run_single_simple_backtest(
    all_available_data: pd.DataFrame, start_date: str = None
) -> Dict[str, Any]:
    """Runs a single backtest using default parameters and returns a results dictionary."""
    logging.info("--- [Backtester] Starting Single Backtest ---")
    params = strategy_config.get_current_strategy_params()

    engine = BacktestEngine(
        symbol_list=config.SYMBOLS,
        initial_capital=config.INITIAL_CAPITAL,
        all_processed_data=all_available_data,
        start_date=start_date,
        risk_on_symbols=config.RISK_ON_SYMBOLS,
        risk_off_symbols=config.RISK_OFF_SYMBOLS,
        **params,
    )
    engine.run()

    holdings_df = create_equity_curve_dataframe(engine.portfolio.all_holdings)
    if holdings_df.empty:
        logging.error("Single backtest produced no results.")
        return {}

    logging.info(
        f"Single backtest completed. Final capital: ${holdings_df['total'].iloc[-1]:,.2f}"
    )
    run_and_display_monte_carlo(holdings_df)

    # FIXED: Return a dictionary with all necessary results
    return {
        "holdings": holdings_df,
        "trade_statistics": engine.portfolio.get_trade_statistics(),
        "closed_trades": pd.DataFrame(engine.portfolio.all_trades),
    }


def run_and_display_monte_carlo(equity_curve: pd.DataFrame):
    """Runs a Monte Carlo simulation on an equity curve and logs the results."""
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
