#!/usr/bin/env python3
# main.py - Alpha Strategy Backtesting System
import argparse
import cProfile
import logging
import os
import pstats
import sys
import time
from datetime import datetime, timedelta
from pstats import SortKey
from typing import Any, Dict

import pandas as pd

# Add project path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Project imports
import config
import config.general_config as general_config
from analysis.runner import generate_report
from backtester.optimization import (run_single_simple_backtest,
                                     run_walk_forward_optimization)
from data_pipeline.unified_data_manager import UnifiedDataManager


def setup_logging():
    """Configure logging for the application"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("backtest.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_required_symbols() -> list:
    """Get all required symbols for backtesting"""
    base_etf_symbols = general_config.RISK_ON_SYMBOLS + general_config.RISK_OFF_SYMBOLS
    individual_stocks = getattr(general_config, "INDIVIDUAL_STOCKS", [])
    all_symbols = list(set(base_etf_symbols + individual_stocks))

    print("Symbol breakdown:")
    print(f"  Risk-on ETFs: {len(general_config.RISK_ON_SYMBOLS)}")
    print(f"  Risk-off ETFs: {len(general_config.RISK_OFF_SYMBOLS)}")
    if individual_stocks:
        print(f"  Individual stocks: {len(individual_stocks)}")
    print(f"  Total unique symbols: {len(all_symbols)}")
    return all_symbols


def run_data_pipeline() -> pd.DataFrame:
    """Run the unified data pipeline"""
    print("\nRunning data pipeline...")
    try:
        backtest_start_dt = datetime.strptime(
            general_config.BACKTEST_START_DATE, "%Y-%m-%d"
        )

        # Calculate data fetch start date using the warm-up period from general_config
        data_fetch_start_dt = backtest_start_dt - timedelta(
            days=general_config.WARMUP_PERIOD_DAYS * 1.5
        )
        data_fetch_start_str = data_fetch_start_dt.strftime("%Y-%m-%d")

        print(f"Backtest will start on: {general_config.BACKTEST_START_DATE}")
        print(
            f"Requesting data from: {data_fetch_start_str} to include a {general_config.WARMUP_PERIOD_DAYS}-day warm-up period."
        )

        # MODIFIED: Instantiate and use UnifiedDataManager directly
        manager = UnifiedDataManager()
        all_data = manager.build_feature_set(
            force_refresh=False,
            start_date=data_fetch_start_str,  # Pass the calculated start date
        )

        if all_data is None or all_data.empty:
            raise ValueError("Data pipeline returned an empty dataset")

        date_range = all_data.index.get_level_values("timestamp")
        print(
            f"Pipeline completed. {len(all_data)} total records loaded (including warm-up)."
        )
        print(
            f"Full data range: {date_range.min().date()} to {date_range.max().date()}"
        )
        return all_data
    except Exception as e:
        logging.error(f"Data pipeline failed: {e}", exc_info=True)
        return None


def run_backtest(all_data: pd.DataFrame) -> Dict[str, Any]:
    """Run backtest with WFO or simple mode, expecting a results dictionary."""
    print(f"\nStarting backtest...")

    if config.WFO_ENABLED:
        print("Walk-Forward Optimization is enabled.")
        # We assume run_walk_forward_optimization is correctly passing the start_date internally
        return run_walk_forward_optimization(
            all_data, start_date=general_config.BACKTEST_START_DATE
        )
    else:
        print("Running single backtest (WFO disabled)")
        return run_single_simple_backtest(
            all_data, start_date=general_config.BACKTEST_START_DATE
        )


def generate_analysis(backtest_results: dict):
    """Generate comprehensive analysis report from the backtest results."""
    print("\nGenerating performance analysis...")
    from analysis.runner import generate_report

    try:
        holdings_df = backtest_results.get("holdings", pd.DataFrame())
        # The trades data is also available
        closed_trades_df = backtest_results.get("closed_trades", pd.DataFrame())

        if holdings_df.empty:
            print("WARNING: No holdings data to analyze. Skipping report generation.")
            return

        generate_report(holdings_df, closed_trades_df, config.INITIAL_CAPITAL)

        print("Analysis report generated successfully.")

    except Exception as e:
        print(f"Analysis generation failed: {e}")
        logging.error(f"Analysis error: {e}", exc_info=True)


def print_configuration():
    """Prints a summary of the current backtest configuration."""
    print("\n" + "=" * 60)
    print("CURRENT BACKTEST CONFIGURATION")
    print("=" * 60)
    print(f"Strategy: {config.CURRENT_STRATEGY}")
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:,.0f}")
    print(
        f"Backtest Period: {config.BACKTEST_START_DATE} to {config.BACKTEST_END_DATE}"
    )

    alpha_params = config.ALPHA_UNIFIED_PARAMS
    print("\n--- Key Alpha Parameters (strategy_config.py) ---")
    print(
        f"  Alpha Weights: Momentum={alpha_params['momentum_weight']:.0%}, Technical={alpha_params['technical_weight']:.0%}"
    )
    print(
        f"  Momentum Windows: {alpha_params['momentum_short']}-{alpha_params['momentum_medium']}-{alpha_params['momentum_long']}"
    )

    print("\n--- Key Risk & Execution Parameters (trading_parameters.py) ---")
    print("  Portfolio Construction:")
    print(
        f"    - Max Positions: {config.TRADING_PARAMS['position_sizing']['max_positions']}"
    )
    print(
        f"    - Sizing Method: {config.TRADING_PARAMS['position_sizing']['position_size_method']}"
    )
    print(
        f"    - Rebalance Freq: {config.TRADING_PARAMS['strategy_execution']['rebalance_frequency']}"
    )
    print(f"  Risk Management:")
    print(
        f"    - Max Drawdown Threshold: {config.RISK_PARAMS['drawdown_control']['max_drawdown_threshold']:.0%}"
    )
    print(f"    - Stop Loss Enabled: {config.RISK_PARAMS['stop_loss']['enabled']}")
    print("=" * 60)


def main() -> int:
    """Main execution function"""
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description="Alpha Strategy Backtesting")
    parser.add_argument("--profile", action="store_true", help="启用性能分析")
    parser.add_argument("--time-phases", action="store_true", help="显示各阶段耗时")
    args = parser.parse_args()

    start_time = datetime.now()
    phase_timings = {}

    print("=" * 60)
    print("ALPHA UNIFIED STRATEGY BACKTESTING SYSTEM")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    setup_logging()
    print_configuration()

    # 如果启用性能分析
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        # Phase 1: Data Pipeline
        phase_start = time.perf_counter()
        print("\n[PHASE 1] Data Pipeline")
        print("-" * 40)
        all_data = run_data_pipeline()
        phase_timings["data_pipeline"] = time.perf_counter() - phase_start

        if all_data is None or all_data.empty:
            print("CRITICAL ERROR: No valid market data was loaded. Exiting.")
            return 1

        # Phase 2: Backtest Execution
        phase_start = time.perf_counter()
        print("\n[PHASE 2] Backtest Execution")
        print("-" * 40)
        backtest_results = run_backtest(all_data)
        phase_timings["backtest"] = time.perf_counter() - phase_start

        if (
            not backtest_results
            or backtest_results.get("holdings", pd.DataFrame()).empty
        ):
            print("CRITICAL ERROR: Backtest failed or produced no results. Exiting.")
            return 1

        # Phase 3: Performance Analysis
        phase_start = time.perf_counter()
        print("\n[PHASE 3] Performance Analysis")
        print("-" * 40)
        generate_analysis(backtest_results)
        phase_timings["analysis"] = time.perf_counter() - phase_start

        # 显示性能分析结果
        if args.profile:
            profiler.disable()
            print("\n" + "=" * 60)
            print("PERFORMANCE PROFILING RESULTS")
            print("=" * 60)
            stats = pstats.Stats(profiler)
            stats.sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(30)

            # 保存详细结果
            stats.dump_stats(f"profile_{datetime.now():%Y%m%d_%H%M%S}.prof")
            print(f"\n详细分析已保存，可使用 snakeviz 查看")

        # 显示各阶段耗时
        if args.time_phases or args.profile:
            print("\n" + "=" * 60)
            print("PHASE TIMING BREAKDOWN")
            print("=" * 60)
            total_time = sum(phase_timings.values())
            for phase, duration in phase_timings.items():
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                print(f"{phase:20}: {duration:8.2f}s ({percentage:5.1f}%)")
            print(f"{'TOTAL':20}: {total_time:8.2f}s")

        end_time = datetime.now()
        print("\n" + "=" * 60)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {end_time - start_time}")
        print("=" * 60)
        return 0

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        return 130
    except Exception as e:
        logging.error(
            "Main execution failed with an unhandled exception.", exc_info=True
        )
        print(f"\nFATAL ERROR: An unhandled exception occurred: {e}")
        return 1


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if os.getcwd() != project_root:
        os.chdir(project_root)
    sys.exit(main())
