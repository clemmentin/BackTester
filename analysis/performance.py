# --- START OF FILE performance.py ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional


def create_equity_curve_dataframe(all_holdings: List[Dict]) -> pd.DataFrame:
    """Transforms a list of holdings snapshots into a clean equity curve DataFrame."""
    df = pd.DataFrame(all_holdings)
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df = (
        df.drop_duplicates(subset=["timestamp"], keep="last")
        .set_index("timestamp")
        .sort_index()
    )

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    df["positions_value"] = df.drop(columns=["cash", "total"], errors="ignore").sum(
        axis=1
    )
    df["returns"] = df["total"].pct_change().fillna(0)

    return df


def calculate_advanced_metrics(
    equity_curve: pd.DataFrame, initial_capital: float, risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculates a comprehensive set of risk-adjusted performance metrics."""
    if equity_curve is None or equity_curve.empty or len(equity_curve) < 20:
        return {}

    total_equity = equity_curve["total"]
    daily_returns = total_equity.pct_change().dropna()
    active_returns = daily_returns[daily_returns != 0].copy()
    if len(active_returns) < 20:
        active_returns = daily_returns

    final_value = total_equity.iloc[-1]
    total_return = final_value / initial_capital - 1

    days_held = (total_equity.index[-1] - total_equity.index[0]).days
    years_passed = max(days_held / 365.25, 1.0 / 252.0)

    annualized_return = (
        (1 + total_return) ** (1 / years_passed) - 1 if years_passed > 0 else 0
    )
    annualized_volatility = active_returns.std() * np.sqrt(252)

    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_volatility
        if annualized_volatility > 0
        else 0.0
    )

    negative_returns = active_returns[active_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (
        (annualized_return - risk_free_rate) / downside_deviation
        if downside_deviation > 0
        else 0.0
    )

    high_water_mark = total_equity.cummax()
    drawdown_series = (total_equity - high_water_mark) / high_water_mark
    max_drawdown = abs(drawdown_series.min())

    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
    max_dd_duration = _calculate_max_drawdown_duration(drawdown_series)

    monthly_returns = total_equity.resample("ME").last().pct_change()
    positive_months = (
        (monthly_returns > 0).sum() / len(monthly_returns)
        if len(monthly_returns) > 0
        else 0
    )

    return {
        "Total Return (%)": total_return * 100,
        "Annualized Return (%)": annualized_return * 100,
        "Volatility (%)": annualized_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Maximum Drawdown (%)": -max_drawdown * 100,
        "Max DD Duration (days)": float(max_dd_duration),
        "VaR (95%)": active_returns.quantile(0.05) * 100,
        "CVaR (95%)": active_returns[
            active_returns <= active_returns.quantile(0.05)
        ].mean()
        * 100,
        "Skewness": active_returns.skew(),
        "Kurtosis": active_returns.kurtosis(),
        "Positive Months (%)": positive_months * 100,
    }


def _calculate_max_drawdown_duration(drawdown_series: pd.Series) -> int:
    """Helper function to calculate the longest drawdown period in days."""
    in_dd = drawdown_series < 0
    if not in_dd.any():
        return 0

    grouper = (in_dd != in_dd.shift()).cumsum()
    dd_groups = drawdown_series[in_dd].groupby(grouper)
    if dd_groups.ngroups == 0:
        return 0

    max_duration = dd_groups.apply(lambda x: x.index[-1] - x.index[0]).max()
    return max_duration.days


def generate_monthly_returns_table(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Generates a pivot table of monthly returns."""
    if equity_curve.empty:
        return pd.DataFrame()

    monthly_returns = equity_curve["total"].resample("ME").last().pct_change()
    pivot_table = (
        monthly_returns.to_frame(name="returns").pivot_table(
            index=monthly_returns.index.year,
            columns=monthly_returns.index.strftime("%b"),
            values="returns",
        )
        * 100
    )

    # Ensure month columns are in calendar order
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    pivot_table = pivot_table.reindex(columns=months).fillna(0)

    yearly_returns = (
        equity_curve["total"]
        .resample("YE")
        .last()
        .pct_change()
        .reindex(equity_curve.index, method="bfill")
    )
    pivot_table["YTD"] = (
        equity_curve["total"]
        .groupby(equity_curve.index.year)
        .apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    )

    return pivot_table


def calculate_trade_stats_by_symbol(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates detailed trade statistics per symbol, returning raw numeric data."""
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return pd.DataFrame()

    grouped = trades_df.groupby("symbol")
    stats = grouped["pnl"].agg(["count", "sum"])
    stats["wins"] = grouped.apply(lambda x: (x["pnl"] > 0).sum())
    stats["losses"] = grouped.apply(lambda x: (x["pnl"] < 0).sum())
    stats["gross_profit"] = grouped.apply(lambda x: x[x["pnl"] > 0]["pnl"].sum())
    stats["gross_loss"] = abs(grouped.apply(lambda x: x[x["pnl"] < 0]["pnl"].sum()))

    stats["win_rate"] = stats["wins"] / stats["count"]
    stats["profit_factor"] = (stats["gross_profit"] / stats["gross_loss"]).replace(
        np.inf, 0
    )  # Avoid inf
    stats["avg_win"] = (stats["gross_profit"] / stats["wins"]).replace(np.inf, 0)
    stats["avg_loss"] = (stats["gross_loss"] / stats["losses"]).replace(np.inf, 0)

    result_df = stats[
        ["count", "win_rate", "profit_factor", "sum", "avg_win", "avg_loss"]
    ].rename(
        columns={
            "count": "Total Trades",
            "win_rate": "Win Rate",
            "profit_factor": "Profit Factor",
            "sum": "Total PnL",
            "avg_win": "Avg Win ($)",
            "avg_loss": "Avg Loss ($)",
        }
    )

    return result_df.sort_values("Total PnL", ascending=False)


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    benchmark_equity: Optional[pd.DataFrame] = None,
    title: str = "Strategy Performance",
):
    """Plots the equity curve and saves it to a file without showing it."""
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(
        equity_curve.index,
        equity_curve["total"],
        label="Strategy Equity",
        color="blue",
        linewidth=2,
    )
    ax.fill_between(
        equity_curve.index,
        equity_curve["total"],
        equity_curve["cash"],
        alpha=0.3,
        color="orange",
        label="Positions Value",
    )

    if benchmark_equity is not None:
        ax.plot(
            benchmark_equity.index,
            benchmark_equity["total"],
            label="Benchmark (SPY)",
            color="grey",
            linestyle="--",
            linewidth=1.5,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.legend(loc="upper left")
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    fig.tight_layout()
    plt.savefig("equity_curve.png", dpi=300)
    plt.close(fig)  # Close figure to free memory and prevent blocking


def create_interactive_dashboard(
    equity_curve: pd.DataFrame,
    benchmark_equity: Optional[pd.DataFrame] = None,
    trades_df: Optional[pd.DataFrame] = None,
):
    """Generates an interactive Plotly dashboard and saves it as an HTML file."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(
            "Portfolio Value",
            "Risk Asset Exposure",
            "Cumulative Alpha vs Benchmark",
            "Drawdown",
        ),
        vertical_spacing=0.04,
    )

    # Plot 1: Portfolio Value
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["total"],
            name="Strategy Value",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    if benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity["total"],
                name="Benchmark (SPY)",
                line=dict(color="grey", dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Plot 2: Position Exposure
    exposure = (equity_curve["positions_value"] / equity_curve["total"] * 100).fillna(0)
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=exposure,
            name="Risk Exposure (%)",
            line=dict(color="orange"),
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )

    # Plot 3: Relative Performance (Alpha)
    if benchmark_equity is not None:
        aligned_returns = pd.concat(
            [equity_curve["returns"], benchmark_equity["returns"]], axis=1, join="inner"
        ).fillna(0)
        aligned_returns.columns = ["strategy", "benchmark"]
        cumulative_alpha = (
            aligned_returns["strategy"] - aligned_returns["benchmark"]
        ).cumsum() * 100
        fig.add_trace(
            go.Scatter(
                x=cumulative_alpha.index,
                y=cumulative_alpha,
                name="Cumulative Alpha (%)",
                line=dict(color="green"),
            ),
            row=3,
            col=1,
        )

    # Plot 4: Drawdown
    hwm = equity_curve["total"].cummax()
    drawdown = (equity_curve["total"] - hwm) / hwm * 100
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill="tozeroy",
            name="Drawdown (%)",
            line=dict(color="red"),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title="Comprehensive Strategy Diagnosis Dashboard",
        height=1200,
        template="plotly_white",
    )
    fig.write_html("detailed_backtest_analysis.html")


def analyze_market_phases(equity_curve: pd.DataFrame, benchmark_equity: pd.DataFrame):
    """Prints a comparative performance analysis across different market phases."""
    print("\n=== MARKET PHASE ANALYSIS ===")
    phases = {
        "COVID Crash & Recovery": ("2020-01-01", "2021-12-31"),
        "Inflation & Rate Hikes": ("2022-01-01", "2023-12-31"),
        "Recent Period": ("2024-01-01", pd.to_datetime("today").strftime("%Y-%m-%d")),
    }
    for phase, (start, end) in phases.items():
        strat_slice = equity_curve.loc[start:end]
        bench_slice = benchmark_equity.loc[start:end]
        if len(strat_slice) < 2 or len(bench_slice) < 2:
            continue

        strat_ret = (strat_slice["total"].iloc[-1] / strat_slice["total"].iloc[0]) - 1
        bench_ret = (bench_slice["total"].iloc[-1] / bench_slice["total"].iloc[0]) - 1
        print(f"\n{phase} ({start} to {end}):")
        print(
            f"  Strategy Return: {strat_ret:+.1%}, Benchmark Return: {bench_ret:+.1%}, Alpha: {strat_ret - bench_ret:+.1%}"
        )


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================


def analyze_filter_effectiveness(diagnostics_log: List[Dict]):
    """Analyzes and prints a summary of signal filter effectiveness."""
    if not diagnostics_log:
        print("No diagnostic data for filter effectiveness analysis.")
        return

    df = pd.DataFrame(
        [d.get("filter_stats", {}) for d in diagnostics_log if d.get("filter_stats")]
    )
    if df.empty:
        return

    summary = df.sum()
    total = summary.get("total", 0)
    if total == 0:
        return

    print("\n=== FILTER EFFECTIVENESS ANALYSIS ===")
    passed = summary.get("passed_all", 0)
    print(f"Total Signals Processed: {int(total):,}")
    print(f"Signals Passed All Filters: {int(passed):,} ({passed / total:.1%})")
    print("-" * 40)

    reasons = summary.drop(["total", "passed_all"]).sort_values(ascending=False)
    for reason, count in reasons.items():
        if count > 0:
            print(
                f"{reason.replace('_', ' ').title():<30}: {int(count):>7,} ({count / total:5.1%})"
            )


def plot_signal_stability(diagnostics_log: List[Dict]):
    """Plots raw vs. filtered signal counts over time and saves the figure."""
    if not diagnostics_log:
        return
    df = pd.DataFrame(diagnostics_log)
    if (
        df.empty
        or "timestamp" not in df.columns
        or "raw_signal_counts" not in df.columns
    ):
        return

    df["raw_total"] = df["raw_signal_counts"].apply(
        lambda x: sum(x.values()) if isinstance(x, dict) else 0
    )
    df["filtered_total"] = df["filtered_signal_counts"].apply(
        lambda x: sum(x.values()) if isinstance(x, dict) else 0
    )

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df["timestamp"], df["raw_total"], label="Raw Signals", color="skyblue")
    ax.plot(
        df["timestamp"],
        df["filtered_total"],
        label="Filtered Signals (Used)",
        color="darkblue",
        marker="o",
        markersize=4,
    )
    ax.set_title("Signal Count Stability Over Time", fontsize=16)
    ax.set_ylabel("Signal Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("signal_stability_analysis.png", dpi=300)
    plt.close(fig)


def plot_position_stability(diagnostics_log: List[Dict], min_positions_target: int = 5):
    """Plots the number of positions held over time and saves the figure."""
    if not diagnostics_log:
        return
    df = pd.DataFrame(diagnostics_log)
    if (
        df.empty
        or "timestamp" not in df.columns
        or "final_position_count" not in df.columns
    ):
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(
        df["timestamp"],
        df["final_position_count"],
        label="Position Count",
        color="darkblue",
        marker="o",
        markersize=4,
    )
    ax.axhline(
        y=min_positions_target,
        color="red",
        linestyle="--",
        label=f"Min Target ({min_positions_target})",
    )
    ax.set_title("Position Count Stability", fontsize=16)
    ax.set_ylabel("Number of Positions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("position_stability_analysis.png", dpi=300)
    plt.close(fig)


def plot_capital_utilization(diagnostics_log: List[Dict]):
    """Plots target vs. actual capital utilization over time and saves the figure."""
    if not diagnostics_log:
        return
    df = pd.DataFrame(diagnostics_log)
    if df.empty or "timestamp" not in df.columns or "target_exposure" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(
        df["timestamp"],
        df["target_exposure"],
        label="Target Exposure ($)",
        color="green",
        linestyle="--",
    )
    ax.plot(
        df["timestamp"],
        df["final_position_value"],
        label="Actual Position Value ($)",
        color="blue",
    )
    ax.fill_between(
        df["timestamp"],
        df["final_position_value"],
        df["target_exposure"],
        where=df["final_position_value"] < df["target_exposure"],
        color="red",
        alpha=0.2,
        label="Under-utilized Capital",
    )

    ax.set_title("Capital Utilization: Target vs. Actual", fontsize=16)
    ax.set_ylabel("Capital ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("capital_utilization_analysis.png", dpi=300)
    plt.close(fig)
