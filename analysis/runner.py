import logging
import os
import sys
from datetime import timedelta

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .performance import (analyze_market_phases, calculate_advanced_metrics,
                          calculate_trade_stats_by_symbol,
                          create_interactive_dashboard,
                          generate_monthly_returns_table, plot_equity_curve)


def generate_report(
    holdings_df: pd.DataFrame, closed_trades_df: pd.DataFrame, initial_capital: float
):
    print("\n--- [Analysis] Starting Report Generation ---")
    if holdings_df is None or holdings_df.empty:
        print("--- [Analysis] FAILED: No holdings data to analyze. ---")
        return

    final_equity_curve_processed = holdings_df

    if len(final_equity_curve_processed) < 2:
        print("\nWARNING: Equity curve has less than 2 data points.")
        print("--- [Analysis] Finished (Not enough data for a full report) ---")
        return

    # --- Fetch and process benchmark data (SPY) ---
    print("Fetching benchmark data (SPY)...")
    benchmark_metrics = None
    spy_equity_curve_df = None
    try:
        start_date = final_equity_curve_processed.index.min()
        end_date = final_equity_curve_processed.index.max()

        # Fetch data starting a few days earlier to correctly calculate the first day's return
        spy_data = yf.download(
            "SPY",
            start=start_date - timedelta(days=3),
            end=end_date + timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )

        if not spy_data.empty:
            spy_data.index = pd.to_datetime(spy_data.index).normalize().tz_localize(None)
            final_equity_curve_processed.index = pd.to_datetime( final_equity_curve_processed.index).normalize().tz_localize(None)
            spy_data = spy_data.loc[spy_data.index >= start_date]
            if len(spy_data) >= 2:
                close_prices = spy_data["Close"]
                if hasattr(close_prices, 'values') and close_prices.values.ndim > 1:
                    close_prices = close_prices.iloc[:, 0] if close_prices.shape[1] > 0 else close_prices.iloc[:, 0]
                normalized_prices = close_prices / close_prices.iloc[0]
                equity_values = (initial_capital * normalized_prices).values
                if equity_values.ndim > 1:
                    equity_values = equity_values.flatten()
                spy_equity_curve = pd.Series(
                    equity_values,
                    index=spy_data.index,
                    name="total"
                )
                returns_values = close_prices.pct_change().fillna(0).values
                if returns_values.ndim > 1:
                    returns_values = returns_values.flatten()
                spy_returns = pd.Series(
                    returns_values,
                    index=spy_data.index,
                    name="returns"
                )
                spy_equity_curve_df = pd.DataFrame({
                    "total": spy_equity_curve,
                    "returns": spy_returns
                })
                spy_equity_curve_df = spy_equity_curve_df.loc[
                    (spy_equity_curve_df.index >= start_date) &
                    (spy_equity_curve_df.index <= end_date)
                    ]
                if len(spy_equity_curve_df) >= 2:
                    benchmark_metrics = calculate_advanced_metrics(spy_equity_curve_df, initial_capital)
                    print("Benchmark data processed successfully.")
                else:
                    print("WARNING: Benchmark data alignment resulted in insufficient data points.")
            else:
                print("WARNING: Insufficient benchmark data for the given period. Skipping benchmark comparison.")
        else:
            print("WARNING: Downloaded benchmark data is empty.")
    except Exception as e:
        print(
            f"WARNING: Could not process benchmark data: {e}. Skipping benchmark comparison."
        )

    # --- Calculate metrics for the strategy ---
    trade_stats = None
    final_metrics = calculate_advanced_metrics(holdings_df, initial_capital)

    # --- Print the final metrics table directly to the console ---
    print("\n" + "=" * 70)
    print("                 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 70)

    if benchmark_metrics:
        print(f"{'Metric':<30} | {'Strategy':>15} | {'Benchmark (SPY)':>18}")
        print("-" * 70)
        all_keys = sorted(
            list(set(final_metrics.keys()) | set(benchmark_metrics.keys()))
        )
        for key in all_keys:
            strat_val = final_metrics.get(key, "N/A")
            bench_val = benchmark_metrics.get(key, "N/A")
            print(f"{key:<30} | {str(strat_val):>15} | {str(bench_val):>18}")
    else:
        print(f"{'Metric':<30} | {'Value':>15}")
        print("-" * 50)
        for key, value in final_metrics.items():
            print(f"{key:<30} | {str(value):>15}")

    print("=" * 70)

    # --- Generate Visualizations and Additional Stats ---
    monthly_returns = generate_monthly_returns_table(final_equity_curve_processed)
    if not monthly_returns.empty:
        print("\n=== MONTHLY RETURNS TABLE (%) ===")
        print(monthly_returns.round(1))

    if closed_trades_df is not None and not closed_trades_df.empty:
        print("\n=== PER-SYMBOL TRADE ANALYSIS ===")
        symbol_stats = calculate_trade_stats_by_symbol(closed_trades_df)
        if not symbol_stats.empty:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                None,
            ):
                print(symbol_stats)
        else:
            print("No closed trades available to analyze by symbol.")

    print("\nDisplaying combined equity curve plot (a window may pop up)...")
    plot_equity_curve(
        final_equity_curve_processed,
        benchmark_equity=spy_equity_curve_df,
        title="Strategy Performance vs. Buy & Hold SPY",
    )

    print("Generating interactive dashboard file (detailed_backtest_analysis.html)...")
    create_interactive_dashboard(
        final_equity_curve_processed, benchmark_equity=spy_equity_curve_df
    )

    if spy_equity_curve_df is not None:
        analyze_market_phases(final_equity_curve_processed, spy_equity_curve_df)

    print("\n--- [Analysis] Finished Successfully ---")
