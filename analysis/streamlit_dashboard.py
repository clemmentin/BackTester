# --- START OF FILE streamlit_dashboard.py ---

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# --- Local Imports ---
from analysis.performance import (
    calculate_advanced_metrics,
    calculate_trade_stats_by_symbol,
    generate_monthly_returns_table,
)


def load_backtest_results():
    """
    Loads backtest artifacts. In a real scenario, this would read files.
    Returns a dictionary of DataFrames and other data structures.
    """
    # Placeholder: In a real implementation, you would load data from files,
    # for example, equity_curve_df = pd.read_csv('equity_curve.csv')
    return {
        "equity_curve": pd.DataFrame(),
        "closed_trades": pd.DataFrame(),
        "trade_stats": {},
        "risk_metrics": {},
        "monitoring_data": {},
        "diagnostics": pd.DataFrame(),  # For market state and alpha context
    }


def main():
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Quantitative Strategy Dashboard",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Quantitative Trading Strategy Dashboard")
    st.markdown("Monitoring and analysis of adaptive strategy performance.")

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    st.sidebar.info(
        f"""
        **Strategy**: `{config.CURRENT_STRATEGY}`  
        **WFO Enabled**: `{config.WFO_ENABLED}`  
        **Risk Assets**: `{len(config.RISK_ON_SYMBOLS)}`  
        **Max Drawdown**: `{config.RISK_PARAMS['drawdown_control']['max_drawdown_threshold'] * 100:.1f}%`
        """
    )
    if st.sidebar.button("Refresh Data"):
        st.rerun()

    # --- Data Loading ---
    data = load_backtest_results()
    equity_curve = data.get("equity_curve", pd.DataFrame())
    closed_trades = data.get("closed_trades", pd.DataFrame())
    trade_stats = data.get("trade_stats", {})
    monitoring_data = data.get("monitoring_data", {})
    diagnostics = data.get("diagnostics", pd.DataFrame())

    if equity_curve.empty:
        st.warning("No backtest results available. Please run a backtest first.")
        st.stop()

    # --- Main KPIs ---
    st.header("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    initial_capital = config.INITIAL_CAPITAL
    current_value = equity_curve["total"].iloc[-1]
    total_return = (current_value / initial_capital - 1) * 100

    col1.metric("Portfolio Value", f"${current_value:,.0f}", f"{total_return:+.1f}%")
    max_dd = equity_curve["drawdown_pct"].min()
    col2.metric("Max Drawdown", f"{max_dd:.1f}%", delta_color="inverse")
    win_rate = trade_stats.get("win_rate", 0) * 100
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    total_trades = trade_stats.get("total_trades", 0)
    col4.metric("Total Trades", f"{total_trades}")
    profit_factor = trade_stats.get("profit_factor", 0)
    col5.metric("Profit Factor", f"{profit_factor:.2f}")

    # --- Equity Curve & Risk Analysis Chart ---
    st.header("Performance & Risk Dynamics")
    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        shared_xaxes=True,
        subplot_titles=(
            "Portfolio Value",
            "Daily Returns",
            "Drawdown",
            "Risk Adjustment Factor",
        ),
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Scatter(
            x=equity_curve.index, y=equity_curve["total"], name="Portfolio Value"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=equity_curve.index,
            y=equity_curve["returns"] * 100,
            name="Daily Returns (%)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["drawdown_pct"],
            name="Drawdown (%)",
            fill="tozeroy",
            line=dict(color="red"),
        ),
        row=3,
        col=1,
    )

    # Plot the transition adjustment factor if available in diagnostics
    if not diagnostics.empty and "transition_adjustment" in diagnostics.columns:
        fig.add_trace(
            go.Scatter(
                x=diagnostics.index,
                y=diagnostics["transition_adjustment"],
                name="Transition Factor",
                line=dict(color="purple", dash="dot"),
            ),
            row=4,
            col=1,
        )
        fig.update_yaxes(range=[0, 1.1], row=4, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Alpha & Risk Context ---
    st.header("Alpha Engine & Risk Context")
    col_context1, col_context2 = st.columns(2)

    with col_context1:
        st.subheader("Current Market State")
        if not diagnostics.empty:
            latest_state = diagnostics.iloc[-1]
            st.info(
                f"""
            - **Market Regime**: `{latest_state.get('market_regime', 'N/A')}`
            - **Volatility State**: `{latest_state.get('volatility_regime', 'N/A')}`
            - **Transition Adjustment**: `{latest_state.get('transition_adjustment', 1.0):.2f}`
            - **System Status**: `{monitoring_data.get('system_status', 'UNKNOWN')}`
            """
            )
        else:
            st.info("No market state data available.")

    with col_context2:
        st.subheader("Recent Alerts")
        alerts = monitoring_data.get("recent_alerts", [])
        if alerts:
            for alert in alerts[-3:]:
                st.warning(
                    f"**{alert.get('severity', 'INFO')}**: {alert.get('message', 'No message.')}"
                )
        else:
            st.success("No recent alerts.")

    # --- Detailed Statistics & Trade Analysis ---
    st.header("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(
        ["Performance Metrics", "Monthly Returns", "Trade Analysis"]
    )

    with tab1:
        st.subheader("Performance Metrics")
        detailed_metrics = calculate_advanced_metrics(equity_curve, initial_capital)
        metrics_df = pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in detailed_metrics.items()]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Monthly Returns Heatmap")
        monthly_returns = generate_monthly_returns_table(equity_curve)
        if not monthly_returns.empty:
            fig_heatmap = px.imshow(
                monthly_returns.iloc[:, :-1],
                color_continuous_scale="RdYlGn",
                aspect="auto",
                text_auto=".1f",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab3:
        st.subheader("Per-Symbol Performance")
        if not closed_trades.empty:
            symbol_stats_df = calculate_trade_stats_by_symbol(closed_trades)
            st.dataframe(symbol_stats_df, use_container_width=True)

            pnl_by_symbol = (
                closed_trades.groupby("symbol")["pnl"]
                .sum()
                .sort_values(ascending=False)
            )
            fig_pnl = px.bar(
                pnl_by_symbol,
                title="Profit & Loss Contribution per Symbol",
                color=pnl_by_symbol.values,
                color_continuous_scale=px.colors.diverging.RdYlGn,
                color_continuous_midpoint=0,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No closed trades data to analyze.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        f"*Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )


if __name__ == "__main__":
    main()
