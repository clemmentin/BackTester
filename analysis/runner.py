# --- START OF FILE runner.py ---

import os
import sys
from datetime import timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import yfinance as yf

# Add project root to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .performance import (
    analyze_market_phases,
    calculate_advanced_metrics,
    calculate_trade_stats_by_symbol,
    create_interactive_dashboard,
    generate_monthly_returns_table,
    plot_equity_curve,
    analyze_filter_effectiveness,
    plot_signal_stability,
    plot_position_stability,
    plot_capital_utilization,
)


def _get_spy_benchmark(
    start_date: pd.Timestamp, end_date: pd.Timestamp, initial_capital: float
) -> Optional[pd.DataFrame]:
    """Fetches and processes SPY data to be used as a benchmark equity curve."""
    spy_data = yf.download(
        "SPY",
        start=start_date - timedelta(days=5),
        end=end_date + timedelta(days=5),
        auto_adjust=True,
        progress=False,
    )
    if spy_data.empty:
        print("WARNING: Failed to download SPY benchmark data.")
        return None

    spy_close = spy_data["Close"]

    if isinstance(spy_close, pd.DataFrame):
        if spy_close.shape[1] == 1:
            spy_close = spy_close.iloc[:, 0]
        else:
            print("WARNING: SPY data has unexpected structure. Benchmark unavailable.")
            return None

    if not isinstance(spy_close, pd.Series):
        print(
            f"WARNING: SPY Close data is {type(spy_close)}, expected Series. Benchmark unavailable."
        )
        return None

    spy_prices = spy_close.loc[start_date:end_date]

    if len(spy_prices) < 2:
        print("WARNING: Insufficient SPY data for the backtest period.")
        return None

    normalized_prices = spy_prices / spy_prices.iloc[0]
    equity_values = initial_capital * normalized_prices

    if not isinstance(equity_values, pd.Series):
        print(
            f"WARNING: SPY equity values are {type(equity_values)}, not a Series. Benchmark unavailable."
        )
        return None

    returns = equity_values.pct_change().fillna(0)

    spy_equity_curve = pd.DataFrame(
        {
            "total": equity_values.values,
            "returns": returns.values,
        },
        index=equity_values.index,
    )
    return spy_equity_curve


def generate_report(
    holdings_df: pd.DataFrame,
    closed_trades_df: pd.DataFrame,
    initial_capital: float,
    diagnostics_log: Optional[List[Dict]] = None,
):
    """Generate a comprehensive backtest report including performance and diagnostic analysis."""
    print("\n--- [Analysis] Starting Report Generation ---")

    if holdings_df is None or holdings_df.empty or len(holdings_df) < 2:
        print("--- [Analysis] FAILED: Not enough holdings data to analyze. ---")
        return

    # === Fetch Benchmark Data ===
    start_date, end_date = holdings_df.index.min(), holdings_df.index.max()
    spy_equity_curve_df = _get_spy_benchmark(start_date, end_date, initial_capital)

    # === Calculate Metrics ===
    strategy_metrics = calculate_advanced_metrics(holdings_df, initial_capital)
    benchmark_metrics = (
        calculate_advanced_metrics(spy_equity_curve_df, initial_capital)
        if spy_equity_curve_df is not None
        else {}
    )

    # === Print Performance Summary ===
    print("\n" + "=" * 70)
    print("                 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 70)
    metric_keys = list(strategy_metrics.keys())
    print(f"{'Metric':<30} | {'Strategy':>15} | {'Benchmark (SPY)':>18}")
    print("-" * 70)
    for key in metric_keys:
        strat_val = strategy_metrics.get(key, float("nan"))
        bench_val = benchmark_metrics.get(key, float("nan"))
        # Format values for printing
        strat_str = (
            f"{strat_val:,.2f}" if isinstance(strat_val, (int, float)) else "N/A"
        )
        bench_str = (
            f"{bench_val:,.2f}" if isinstance(bench_val, (int, float)) else "N/A"
        )
        print(f"{key:<30} | {strat_str:>15} | {bench_str:>18}")
    print("=" * 70)

    # === Monthly Returns ===
    monthly_returns = generate_monthly_returns_table(holdings_df)
    if not monthly_returns.empty:
        print("\n=== MONTHLY RETURNS TABLE (%) ===")
        print(monthly_returns.round(1))

    # === Per-Symbol Trade Analysis ===
    if closed_trades_df is not None and not closed_trades_df.empty:
        symbol_stats_raw = calculate_trade_stats_by_symbol(closed_trades_df)
        if not symbol_stats_raw.empty:
            # Format the numeric DataFrame for pretty printing
            symbol_stats_formatted = symbol_stats_raw.copy()
            symbol_stats_formatted["Win Rate"] = (
                symbol_stats_formatted["Win Rate"] * 100
            ).map("{:.1f}%".format)
            symbol_stats_formatted["Profit Factor"] = symbol_stats_formatted[
                "Profit Factor"
            ].map("{:.2f}".format)
            for col in ["Total PnL", "Avg Win ($)", "Avg Loss ($)"]:
                if col in symbol_stats_formatted.columns:
                    symbol_stats_formatted[col] = symbol_stats_formatted[col].map(
                        "${:,.2f}".format
                    )

            print("\n=== PER-SYMBOL TRADE ANALYSIS ===")
            with pd.option_context("display.max_rows", None, "display.width", 120):
                print(symbol_stats_formatted)

    # === Generate Visualizations ===
    print("\nGenerating performance plots and dashboards...")
    plot_equity_curve(
        holdings_df,
        benchmark_equity=spy_equity_curve_df,
        title="Strategy vs. Benchmark (SPY)",
    )
    create_interactive_dashboard(holdings_df, benchmark_equity=spy_equity_curve_df)
    print("  - equity_curve.png")
    print("  - detailed_backtest_analysis.html")

    if spy_equity_curve_df is not None:
        analyze_market_phases(holdings_df, spy_equity_curve_df)

    # === Diagnostic Analysis ===
    if diagnostics_log:
        print("\n" + "=" * 70)
        print("                 DIAGNOSTIC ANALYSIS")
        print("=" * 70)

        analyze_filter_effectiveness(diagnostics_log)

        print("\nGenerating diagnostic plots...")
        plot_signal_stability(diagnostics_log)

        min_pos_target = 5  # Default
        if "max_position_count" in diagnostics_log[0]:
            min_pos_target = max(3, int(diagnostics_log[0]["max_position_count"] * 0.3))
        plot_position_stability(diagnostics_log, min_positions_target=min_pos_target)

        plot_capital_utilization(diagnostics_log)
        print("  - signal_stability_analysis.png")
        print("  - position_stability_analysis.png")
        print("  - capital_utilization_analysis.png")
    else:
        print("\nWARNING: No diagnostic data available for analysis.")

    print("\n--- [Analysis] Finished Successfully ---")
