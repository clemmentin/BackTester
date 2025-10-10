# main.py
# !/usr/bin/env python3
"""
Quantitative Strategy Backtesting System - Main Execution Engine
This is the primary entry point for running a performance backtest.
"""
import argparse
import cProfile
import logging
import os
import pstats
import sys
import time
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
import config.general_config as general_config
from backtester.optimization import (
    run_single_simple_backtest,
    run_walk_forward_optimization,
)
from data_pipeline.unified_data_manager import UnifiedDataManager


def setup_logging():
    """Configure logging for the entire application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("main_execution.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    # Silence overly verbose third-party libraries
    for logger_name in ["matplotlib", "yfinance", "urllib3", "fredapi"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def print_configuration_summary():
    """Prints a clear summary of the current backtest configuration."""
    print("\n" + "=" * 80)
    print(" " * 25 + "BACKTEST CONFIGURATION SUMMARY")
    print("=" * 80)
    run_mode = "Walk-Forward Optimization" if config.WFO_ENABLED else "Single Backtest"
    print(f"RUN MODE        : {run_mode}")
    print(f"STRATEGY        : {config.CURRENT_STRATEGY}")
    print(f"INITIAL CAPITAL : ${config.INITIAL_CAPITAL:,.0f}")
    print(
        f"BACKTEST PERIOD : {config.BACKTEST_START_DATE} to {config.BACKTEST_END_DATE}"
    )

    if config.CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        params = config.HYBRID_DUAL_PARAMS
        engine_params = params.get("alpha_engine", {})
        ic_enabled = engine_params.get("ic_enabled", True)

        print("\n--- Alpha Engine (3-Factor Model) ---")
        print(f"  Price Weight            : {engine_params.get('price_weight', 0):.0%}")
        print(
            f"  Reversal Weight         : {engine_params.get('reversal_weight', 0):.0%}"
        )
        print(
            f"  Momentum Weight         : {engine_params.get('momentum_weight', 0):.0%}"
        )
        print(f"  Dynamic IC Weighting    : {'Enabled' if ic_enabled else 'Disabled'}")

    print("\n--- Risk & Position Management ---")

    pos_mgmt = config.TRADING_PARAMS.get("position_management", {})
    min_pos = pos_mgmt.get("min_total_positions", "N/A")
    max_pos = pos_mgmt.get("max_total_positions", "N/A")

    exposure_mgmt = config.RISK_PARAMS.get("exposure_management", {})
    min_exp = exposure_mgmt.get("min_exposure_pct", "N/A")
    max_exp = exposure_mgmt.get("max_exposure_pct", "N/A")

    print(f"  Position Count Range    : {min_pos} to {max_pos} positions (dynamic)")
    if isinstance(min_exp, float) and isinstance(max_exp, float):
        print(f"  Portfolio Exposure Range: {min_exp:.0%} to {max_exp:.0%} (dynamic)")
    else:
        print(f"  Portfolio Exposure Range: N/A")

    print(
        f"  Max Drawdown Threshold  : {config.RISK_PARAMS['drawdown_control']['max_drawdown_threshold']:.0%}"
    )
    print(
        f"  Rebalance Frequency     : {config.TRADING_PARAMS['strategy_execution']['rebalance_frequency']}"
    )
    print("=" * 80)


def run_data_pipeline(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Initializes the data manager and builds the complete feature set."""
    print("\n[PHASE 1] Data Pipeline")
    print("-" * 80)
    try:
        manager = UnifiedDataManager()
        all_data = manager.build_feature_set(force_refresh=force_refresh)

        if all_data is None or all_data.empty:
            raise ValueError("Data pipeline returned an empty or None dataset.")

        date_range = all_data.index.get_level_values("timestamp")
        print("Data pipeline successful.")
        print(f"  - Total records loaded: {len(all_data):,}")
        print(f"  - Data range: {date_range.min().date()} to {date_range.max().date()}")
        return all_data
    except Exception as e:
        logging.error(f"Data pipeline failed catastrophically: {e}", exc_info=True)
        return None


def run_full_backtest(all_data: pd.DataFrame):
    """Orchestrates the backtest and subsequent performance analysis."""
    print("\n[PHASE 2] Backtest Execution")
    print("-" * 80)
    if config.WFO_ENABLED:
        logging.info("WFO mode enabled. Starting Walk-Forward Optimization...")
        backtest_results = run_walk_forward_optimization(
            all_data, start_date=general_config.BACKTEST_START_DATE
        )
    else:
        logging.info("Single backtest mode enabled.")
        backtest_results = run_single_simple_backtest(
            all_data, start_date=general_config.BACKTEST_START_DATE
        )

    if not backtest_results or backtest_results.get("holdings", pd.DataFrame()).empty:
        logging.critical(
            "CRITICAL: Backtest failed or produced no results. Aborting analysis."
        )
        return

    print("\n[PHASE 3] Performance Analysis & Reporting")
    print("-" * 80)
    from analysis.runner import (
        generate_report,
    )

    generate_report(
        backtest_results["holdings"],
        backtest_results["closed_trades"],
        config.INITIAL_CAPITAL,
    )


def main() -> int:
    """Main execution function to orchestrate the entire process."""
    start_time_global = time.time()

    parser = argparse.ArgumentParser(
        description="Quantitative Strategy Backtesting System"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force a full refresh of all market data from APIs.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling of the code.",
    )
    args = parser.parse_args()

    setup_logging()
    print_configuration_summary()

    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()

    try:
        all_data = run_data_pipeline(force_refresh=args.force_refresh)
        if all_data is None:
            return 1
        run_full_backtest(all_data)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        return 130
    except Exception:
        logging.critical("A fatal error occurred during main execution.", exc_info=True)
        return 1
    finally:
        if profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
            print("\n" + "=" * 80 + "\nPERFORMANCE PROFILE\n" + "=" * 80)
            stats.print_stats(30)

        total_duration = time.time() - start_time_global
        print("\n" + "=" * 80 + "\nEXECUTION SUMMARY\n" + "=" * 80)
        print(f"Total Run Time: {total_duration:.2f} seconds")
        print(f"End Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
