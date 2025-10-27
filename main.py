# main.py
#!/usr/bin/env python3
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
from config import general_config
import pandas as pd
import numpy as np
import random

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

# ============================================================================

from backtester.optimization import (
    run_single_simple_backtest,
    run_walk_forward_optimization,
)
from data_pipeline.unified_data_manager import UnifiedDataManager


def setup_logging():
    """Configure logging for the application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("main_execution.log", mode="w"),  # Log to file
            logging.StreamHandler(),  # Log to console
        ],
    )
    # Silence verbose third-party libraries
    for logger_name in ["matplotlib", "yfinance", "urllib3", "fredapi"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def print_configuration_summary():
    """Prints a summary of the current backtest configuration."""
    print("\n" + "=" * 80)
    print(" " * 25 + "BACKTEST CONFIGURATION SUMMARY")
    print("=" * 80)

    # --- CORRECTED: Access all variables directly from the 'config' package ---
    run_mode = "Walk-Forward Optimization" if config.WFO_ENABLED else "Single Backtest"
    print(f"RUN MODE        : {run_mode}")
    print(f"STRATEGY        : {config.CURRENT_STRATEGY}")
    print(f"INITIAL CAPITAL : ${config.INITIAL_CAPITAL:,.0f}")
    print(
        f"BACKTEST PERIOD : {config.BACKTEST_START_DATE} to {config.BACKTEST_END_DATE}"
    )

    if config.CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        # Get strategy params directly from the config object
        params = config.HYBRID_DUAL_PARAMS
        engine_params = params.get("alpha_engine", {})
        base_weights = engine_params.get("base_weights", {})
        ic_monitoring_params = params.get("ic_monitoring", {})
        ic_enabled = ic_monitoring_params.get("enabled", False)

        print("\n--- Alpha Engine  ---")
        # CORRECTED: Use the actual keys from the 'base_weights' dictionary
        print(f"  Price Weight            : {base_weights.get('price', 0):.0%}")
        print(f"  Reversal Weight         : {base_weights.get('reversal', 0):.0%}")
        print(f"  Liquidity Weight        : {base_weights.get('liquidity', 0):.0%}")
        # Note: 'momentum' is not a base weight in this config, so it will show 0%
        print(f"  Momentum Weight         : {base_weights.get('momentum', 0):.0%}")
        print(f"  Dynamic IC Weighting    : {'Enabled' if ic_enabled else 'Disabled'}")

    print("\n--- Risk & Position Management ---")

    # CORRECTED: Access RISK_PARAMS directly from config
    pos_mgmt = config.RISK_PARAMS.get("position_management", {})
    min_pos = pos_mgmt.get("min_total_positions", "N/A")
    max_pos = pos_mgmt.get("max_total_positions", "N/A")

    portfolio_limits = config.RISK_PARAMS.get("portfolio_limits", {})
    min_weight = portfolio_limits.get("min_single_position", "N/A")
    max_weight = portfolio_limits.get("max_single_position", "N/A")

    print(f"  Position Count Range    : {min_pos} to {max_pos} positions")
    # CORRECTED: Changed label and keys to reflect the actual config values
    if isinstance(min_weight, float) and isinstance(max_weight, float):
        print(f"  Single Position Weight  : {min_weight:.0%} to {max_weight:.0%}")
    else:
        print(f"  Single Position Weight  : N/A")

    print(
        f"  Max Drawdown Threshold  : {config.RISK_PARAMS['drawdown_control']['max_drawdown_threshold']:.0%}"
    )
    # CORRECTED: This value is in HYBRID_DUAL_PARAMS, not a non-existent 'params' object
    print(
        f"  Rebalance Frequency     : {config.HYBRID_DUAL_PARAMS.get('rebalance_frequency', 'N/A').capitalize()}"
    )
    print("=" * 80)


def run_data_pipeline(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Initialize data manager and build the feature set.
    An exception during this process will terminate the program.
    """
    print("\n[PHASE 1] Data Pipeline")
    print("-" * 80)

    manager = UnifiedDataManager()
    all_data = manager.build_feature_set(force_refresh=force_refresh)

    if all_data is None or all_data.empty:
        logging.error("Data pipeline returned an empty or None dataset.")
        return None

    date_range = all_data.index.get_level_values("timestamp")
    print("Data pipeline successful.")
    print(f"  - Total records loaded: {len(all_data):,}")
    print(f"  - Data range: {date_range.min().date()} to {date_range.max().date()}")
    return all_data


def run_full_backtest(all_data: pd.DataFrame):
    """
    Orchestrate the backtest and performance analysis.
    """
    print("\n[PHASE 2] Backtest Execution")
    print("-" * 80)
    manage_ml_data_file(mode="archive")

    # CORRECTED: Use config.BACKTEST_START_DATE directly
    if config.WFO_ENABLED:
        logging.info("WFO mode enabled. Starting Walk-Forward Optimization...")
        backtest_results = run_walk_forward_optimization(
            all_data, start_date=config.BACKTEST_START_DATE
        )
    else:
        logging.info("Single backtest mode enabled.")
        backtest_results = run_single_simple_backtest(
            all_data, start_date=config.BACKTEST_START_DATE
        )

    if not backtest_results or backtest_results.get("holdings", pd.DataFrame()).empty:
        logging.critical(
            "CRITICAL: Backtest failed or produced no results. Aborting analysis."
        )
        return


def manage_ml_data_file(mode="archive"):
    """
    Manages the ML training data file.
    """
    filepath = "./data/ml_training_data.csv"
    if os.path.exists(filepath):
        if mode == "archive":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = f"./data/archive/ml_training_data_{timestamp}.csv"
            os.makedirs("./data/archive", exist_ok=True)
            os.rename(filepath, archive_path)
            print(f"Archived existing ML data file to: {archive_path}")
        elif mode == "delete":
            os.remove(filepath)
            print(f"Deleted existing ML data file: {filepath}")


def set_global_seeds(seed_value: int):
    """
    Sets global random seeds for all relevant libraries to ensure reproducibility.

    Args:
        seed_value: The integer value to use as the seed.
    """
    try:
        # 1. Python's built-in random module
        random.seed(seed_value)

        # 2. NumPy for all its random operations
        np.random.seed(seed_value)

        # 3. Python's hash seed for reproducible dict/set iteration order (less critical in Py 3.7+)
        os.environ["PYTHONHASHSEED"] = str(seed_value)

        # 4. TensorFlow
        # import tensorflow as tf
        # tf.random.set_seed(seed_value)

        # 5. PyTorch
        # import torch
        # torch.manual_seed(seed_value)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed_value)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        print(f"Global random seeds set to: {seed_value}")

    except Exception as e:
        print(f"Error setting global seeds: {e}")


def main() -> int:
    """Main execution function without try/except blocks."""
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

    # Phase 1: Data Pipeline
    all_data = run_data_pipeline(force_refresh=args.force_refresh)
    if all_data is None:
        logging.critical("Data pipeline failed. Aborting execution.")
        return 1

    # Phase 2: Backtest
    run_full_backtest(all_data)

    if profiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
        print("\n" + "=" * 80 + "\nPERFORMANCE PROFILE\n" + "=" * 80)
        stats.print_stats(30)

    total_duration = time.time() - start_time_global
    print("\n" + "=" * 80 + "\nEXECUTION SUMMARY\n" + "=" * 80)
    print(f"Total Run Time: {total_duration:.2f} seconds")
    print(f"End Time: {datetime.now():%Y-%m-%d %H:%M%S}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    set_global_seeds(general_config.GLOBAL_RANDOM_SEED)
    sys.exit(main())
