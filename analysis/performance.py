import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _calculate_years_passed(equity_curve):
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    return (end_date - start_date).days / 365.2


def create_equity_curve_dataframe(all_holdings):
    df = pd.DataFrame(all_holdings)
    if "timestamp" not in df.columns:
        print("PERFORMANCE ERROR: 'timestamp' column not found in holdings records.")
        return pd.DataFrame(columns=["cash", "total", "positions_value", "returns"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)
    df["positions_value"] = df.drop(columns=["cash", "total"]).sum(axis=1)
    df["returns"] = df["total"].pct_change()
    df["returns"].fillna(0, inplace=True)
    return df


def calculate_performance_metrics(equity_curve, periods_per_year=252):
    total_return = (equity_curve["total"].iloc[-1] / equity_curve["total"].iloc[0]) - 1
    mean_daily_return = equity_curve["returns"].mean()
    std_daily_return = equity_curve["returns"].std()
    sharpe_ratio = mean_daily_return / std_daily_return
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(periods_per_year)
    high_water_mark = equity_curve["total"].cummax()
    drawdown = (equity_curve["total"] - high_water_mark) / high_water_mark
    max_drawdown = drawdown.min()
    annualized_volatility = std_daily_return * np.sqrt(periods_per_year)
    metrics = {
        "Total Return (%)": f"{total_return * 100:.2f}",
        "Annualized Volatility (%)": f"{annualized_volatility * 100:.2f}",
        "Sharpe Ratio": f"{annualized_sharpe_ratio:.2f}",
        "Maximum Drawdown (%)": f"{max_drawdown * 100:.2f}",
    }
    return metrics


def plot_equity_curve(
    equity_curve, benchmark_equity=None, title="Strategy Performance"
):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(
        equity_curve.index,
        equity_curve["total"],
        label="Total Equity",
        color="blue",
        linewidth=2,
    )
    ax.fill_between(
        equity_curve.index,
        equity_curve["cash"],
        equity_curve["total"],
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
    plt.show()


def create_interactive_dashboard(equity_curve, benchmark_equity=None, trades_df=None):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.2, 0.2, 0.25],
        subplot_titles=(
            "Portfolio Value",
            "Position Exposure",
            "Relative Performance",
            "Drawdown",
        ),
        vertical_spacing=0.03,
    )

    # Row 1: Portfolio Value
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["total"],
            name="Strategy Value",
            line=dict(color="blue", width=2),
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

    # Row 2: Position Exposure (Risk Asset Exposure)
    if "positions_value" in equity_curve.columns and "total" in equity_curve.columns:
        position_ratio = equity_curve["positions_value"] / equity_curve["total"] * 100
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=position_ratio,
                name="Risk Asset Exposure (%)",
                line=dict(color="orange"),
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Risk Exposure %", range=[0, 105], row=2, col=1)

    # Row 3: Relative Performance (Cumulative difference between strategy vs benchmark)
    if benchmark_equity is not None:
        strategy_returns = equity_curve["returns"]
        benchmark_returns = benchmark_equity["returns"]
        aligned_strategy, aligned_benchmark = strategy_returns.align(
            benchmark_returns, join="inner"
        )

        relative_returns = (aligned_strategy - aligned_benchmark).fillna(0)
        cumulative_relative = relative_returns.cumsum()

        fig.add_trace(
            go.Scatter(
                x=cumulative_relative.index,
                y=cumulative_relative * 100,
                name="Cumulative Excess Return (%)",
                line=dict(color="green" if cumulative_relative.iloc[-1] > 0 else "red"),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        fig.update_yaxes(title_text="Excess Return %", row=3, col=1)

    # Row 4: Drawdown
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
    fig.update_yaxes(title_text="Drawdown %", row=4, col=1)

    # Add trade markers (on the first subplot)
    if trades_df is not None:
        buys = trades_df[trades_df["direction"] == "Buy"]
        sells = trades_df[trades_df["direction"] == "Sell"]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["price"],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(color="green", symbol="triangle-up", size=8),
                ),
                row=1,
                col=1,
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["price"],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(color="red", symbol="triangle-down", size=8),
                ),
                row=1,
                col=1,
            )

    fig.update_layout(title="Comprehensive Strategy Diagnosis Dashboard", height=1000)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.show()
    fig.write_html("detailed_backtest_analysis.html")


def analyze_market_phases(equity_curve, benchmark_equity):
    print("\n=== MARKET PHASE ANALYSIS ===")

    phases = {
        "Post-GFC Recovery": ("2010-01-01", "2015-12-31"),
        "Pre-Trump Uncertainty": ("2016-01-01", "2016-12-31"),
        "Trump Rally": ("2017-01-01", "2018-12-31"),
        "Trade War Volatility": ("2019-01-01", "2019-12-31"),
        "COVID Crash & Recovery": ("2020-01-01", "2021-12-31"),
        "Inflation & Rate Hikes": ("2022-01-01", "2023-12-31"),
        "Recent Period": ("2024-01-01", "2025-01-01"),
    }

    for phase_name, (start_date, end_date) in phases.items():
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            phase_strategy = equity_curve[
                (equity_curve.index >= start_dt) & (equity_curve.index <= end_dt)
            ]
            phase_benchmark = benchmark_equity[
                (benchmark_equity.index >= start_dt)
                & (benchmark_equity.index <= end_dt)
            ]

            if len(phase_strategy) < 2 or len(phase_benchmark) < 2:
                continue

            strategy_return = (
                phase_strategy["total"].iloc[-1] / phase_strategy["total"].iloc[0]
            ) - 1
            benchmark_return = (
                phase_benchmark["total"].iloc[-1] / phase_benchmark["total"].iloc[0]
            ) - 1
            excess_return = strategy_return - benchmark_return

            strategy_hwm = phase_strategy["total"].cummax()
            strategy_dd = (
                (phase_strategy["total"] - strategy_hwm) / strategy_hwm
            ).min()

            benchmark_hwm = phase_benchmark["total"].cummax()
            benchmark_dd = (
                (phase_benchmark["total"] - benchmark_hwm) / benchmark_hwm
            ).min()

            print(f"\n{phase_name} ({start_date} to {end_date}):")
            print(f"  Strategy Return: {strategy_return * 100:.1f}%")
            print(f"  Benchmark Return: {benchmark_return * 100:.1f}%")
            print(f"  Excess Return: {excess_return * 100:.1f}%")
            print(f"  Strategy Max DD: {strategy_dd * 100:.1f}%")
            print(f"  Benchmark Max DD: {benchmark_dd * 100:.1f}%")

        except Exception as e:
            print(f"  Error analyzing {phase_name}: {e}")


def calculate_advanced_metrics(
    equity_curve: pd.DataFrame, initial_capital: float, risk_free_rate: float = 0.02
) -> dict:
    """
    REBUILT: A single, reliable function to calculate all key performance metrics.
    This is now the single source of truth for performance calculation.
    """
    if equity_curve.empty or len(equity_curve) < 20:
        return {"error": "Insufficient data for performance calculation."}

    # --- Basic Setup ---
    total_equity = equity_curve["total"]
    daily_returns = total_equity.pct_change().dropna()

    # Use only days with trading activity for rate-of-return metrics
    active_returns = daily_returns[daily_returns != 0].copy()
    if len(active_returns) < 20:
        active_returns = daily_returns  # Fallback if too few active days

    # --- Return Metrics ---
    final_value = total_equity.iloc[-1]
    total_return_pct = (final_value / initial_capital - 1) * 100

    days_held = (total_equity.index[-1] - total_equity.index[0]).days
    years_passed = max(days_held / 365.25, 1.0 / 252.0)  # Avoid division by zero

    annualized_return_pct = (
        (1 + total_return_pct / 100) ** (1 / years_passed) - 1
    ) * 100

    # --- Risk Metrics ---
    annualized_volatility_pct = active_returns.std() * np.sqrt(252) * 100

    # --- Risk-Adjusted Ratios (Corrected Formulas) ---
    if annualized_volatility_pct > 0:
        sharpe_ratio = (
            annualized_return_pct - risk_free_rate * 100
        ) / annualized_volatility_pct
    else:
        sharpe_ratio = 0.0

    negative_returns = active_returns[active_returns < 0]
    if not negative_returns.empty:
        downside_deviation_pct = negative_returns.std() * np.sqrt(252) * 100
        sortino_ratio = (
            (annualized_return_pct - risk_free_rate * 100) / downside_deviation_pct
            if downside_deviation_pct > 0
            else 0.0
        )
    else:
        sortino_ratio = float("inf")

    # --- Drawdown Analysis (More Robust) ---
    high_water_mark = total_equity.cummax()
    drawdown_series = (total_equity - high_water_mark) / high_water_mark
    max_drawdown_pct = abs(drawdown_series.min()) * 100

    if max_drawdown_pct > 0:
        calmar_ratio = annualized_return_pct / max_drawdown_pct
    else:
        calmar_ratio = float("inf")

    max_dd_duration = _calculate_max_drawdown_duration(drawdown_series)

    # --- Distributional Stats ---
    skewness = active_returns.skew()
    kurtosis = active_returns.kurtosis()
    var_95 = active_returns.quantile(0.05) * 100
    cvar_95 = (
        active_returns[active_returns <= active_returns.quantile(0.05)].mean() * 100
    )

    # --- Monthly/Daily Stats ---
    monthly_returns = total_equity.resample("ME").last().pct_change()
    positive_months_pct = (
        (monthly_returns > 0).sum() / len(monthly_returns) * 100
        if len(monthly_returns) > 0
        else 0
    )

    # Return formatted strings for direct display
    return {
        "Total Return (%)": f"{total_return_pct:.2f}",
        "Annualized Return (%)": f"{annualized_return_pct:.2f}",
        "Volatility (%)": f"{annualized_volatility_pct:.2f}",
        "Sharpe Ratio": f"{sharpe_ratio:.3f}",
        "Sortino Ratio": f"{sortino_ratio:.3f}",
        "Calmar Ratio": f"{calmar_ratio:.3f}",
        "Maximum Drawdown (%)": f"{-max_drawdown_pct:.2f}",
        "Max DD Duration (days)": f"{max_dd_duration}",
        "VaR (95%)": f"{var_95:.2f}%",
        "CVaR (95%)": f"{cvar_95:.2f}%",
        "Skewness": f"{skewness:.3f}",
        "Kurtosis": f"{kurtosis:.3f}",
        "Positive Months (%)": f"{positive_months_pct:.1f}",
    }


def _calculate_max_drawdown_duration(drawdown_series: pd.Series) -> int:
    """Calculates the longest period (in days) of being in a drawdown."""
    in_dd = drawdown_series < 0
    if not in_dd.any():
        return 0
    grouper = (in_dd != in_dd.shift()).cumsum()

    dd_groups = drawdown_series[in_dd].groupby(grouper)

    if dd_groups.ngroups == 0:
        return 0
    max_duration_delta = dd_groups.apply(
        lambda x: x.index[-1] - x.index[0]
    ).max()

    return max_duration_delta.days


def generate_monthly_returns_table(equity_curve):
    if equity_curve.empty:
        return pd.DataFrame()

    monthly_values = equity_curve["total"].resample("ME").last()
    monthly_returns = monthly_values.pct_change().dropna()
    monthly_table = pd.DataFrame(
        {
            "Year": monthly_returns.index.year,
            "Month": monthly_returns.index.month,
            "Return": monthly_returns.values,
        }
    )
    pivot_table = monthly_table.pivot(index="Year", columns="Month", values="Return")
    pivot_table = pivot_table.fillna(0) * 100
    yearly_groups = equity_curve["total"].groupby(equity_curve.index.year)
    yearly_returns_values = (
        yearly_groups.apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) if not x.empty else 0
        )
        * 100
    )
    pivot_table["YTD"] = yearly_returns_values.reindex(pivot_table.index).fillna(0)
    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    pivot_table = pivot_table.rename(columns=month_names)

    return pivot_table


def calculate_trade_stats_by_symbol(trades_df: pd.DataFrame):
    """
    Calculates detailed trading statistics for each symbol.

    Args:
        trades_df: DataFrame containing closed trades with columns like
                   ['symbol', 'pnl'].

    Returns:
        A DataFrame with aggregated stats per symbol.
    """
    if (
        trades_df is None
        or trades_df.empty
        or "symbol" not in trades_df.columns
        or "pnl" not in trades_df.columns
    ):
        print(
            "PERFORMANCE WARNING: Trade data is missing or malformed for per-symbol analysis."
        )
        return pd.DataFrame()

    stats_by_symbol = []

    for symbol, group in trades_df.groupby("symbol"):
        total_trades = len(group)
        wins = group[group["pnl"] > 0]
        losses = group[group["pnl"] < 0]

        num_wins = len(wins)
        num_losses = len(losses)

        win_rate = num_wins / total_trades if total_trades > 0 else 0

        total_pnl = group["pnl"].sum()

        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = wins["pnl"].mean() if num_wins > 0 else 0
        avg_loss = losses["pnl"].mean() if num_losses > 0 else 0

        stats_by_symbol.append(
            {
                "Symbol": symbol,
                "Total Trades": total_trades,
                "Win Rate (%)": f"{win_rate * 100:.1f}",
                "Profit Factor": (
                    f"{profit_factor:.2f}" if profit_factor != float("inf") else "Inf"
                ),
                "Total PnL ($)": f"{total_pnl:,.2f}",
                "Avg Win ($)": f"{avg_win:,.2f}",
                "Avg Loss ($)": f"{avg_loss:,.2f}",
            }
        )

    if not stats_by_symbol:
        return pd.DataFrame()

    result_df = pd.DataFrame(stats_by_symbol).set_index("Symbol")
    return result_df


def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.02):

    # Method 1: Using only trading days with actual changes
    returns = equity_curve["total"].pct_change().dropna()

    # Filter out days with zero returns (no trading activity)
    active_returns = returns[returns != 0]

    if len(active_returns) < 20:
        return 0

    # Calculate daily excess returns
    daily_rf = risk_free_rate / 252
    excess_returns = active_returns - daily_rf

    # Calculate Sharpe ratio
    if excess_returns.std() > 0:
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    return sharpe
