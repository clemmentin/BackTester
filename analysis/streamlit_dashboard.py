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
from analysis.performance import (calculate_advanced_metrics,
                                  calculate_trade_stats_by_symbol,
                                  generate_monthly_returns_table)


def load_backtest_results():
    """
    Loads backtest results data.

    IMPORTANT: This is a placeholder function. You must connect this to your actual
    data source (e.g., reading from a CSV, Parquet file, or database) where your
    backtest engine saves its results.
    """
    # This should load data from your backtest result files.
    # For now, it returns an example data structure.
    return {
        "equity_curve": pd.DataFrame(),  # Should have columns: 'total', 'returns'
        "closed_trades": pd.DataFrame(),  # Should have columns: 'symbol', 'pnl'
        "trade_stats": {},  # Aggregate trade stats
        "risk_metrics": {},  # Live risk metrics like current drawdown
        "monitoring_data": {},  # System status and alerts
    }


def main():
    st.set_page_config(
        page_title="Quantitative Strategy Dashboard",
        page_icon="bar_chart",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Quantitative Trading Strategy Dashboard")
    st.markdown("Real-time monitoring and analysis of trading strategy performance")

    # --- Sidebar Configuration Display ---
    st.sidebar.header("Configuration")
    st.sidebar.subheader("Current Settings")
    st.sidebar.info(
        f"""
    **Strategy**: {config.CURRENT_STRATEGY}  
    **WFO Enabled**: {config.WFO_ENABLED}  
    **Risk Assets**: {len(config.RISK_ON_SYMBOLS)}  
    **Max Drawdown**: {config.RISK_PARAMS['drawdown_control']['max_drawdown_threshold'] * 100:.1f}%
    """
    )

    if st.sidebar.button("Refresh Data"):
        st.rerun()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=365), datetime.now()),
        key="date_range",
    )

    # --- Load Data ---
    try:
        data = load_backtest_results()
        equity_curve = data.get("equity_curve", pd.DataFrame())
        closed_trades = data.get("closed_trades", pd.DataFrame())
        trade_stats = data.get("trade_stats", {})
        risk_metrics = data.get("risk_metrics", {})
        monitoring_data = data.get("monitoring_data", {})

        if equity_curve.empty:
            st.warning("No backtest results available. Please run a backtest first.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # --- Key Performance Indicators ---
    st.header("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    current_value = (
        equity_curve["total"].iloc[-1]
        if not equity_curve.empty
        else config.INITIAL_CAPITAL
    )
    total_return = (current_value / config.INITIAL_CAPITAL - 1) * 100

    with col1:
        st.metric("Portfolio Value", f"${current_value:,.0f}", f"{total_return:+.1f}%")
    with col2:
        max_dd = risk_metrics.get("current_drawdown", 0) * 100
        st.metric("Current Drawdown", f"{max_dd:.1f}%", delta_color="inverse")
    with col3:
        win_rate = trade_stats.get("win_rate", 0) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        total_trades = trade_stats.get("total_trades", 0)
        st.metric("Total Trades", f"{total_trades}")
    with col5:
        profit_factor = trade_stats.get("profit_factor", 0)
        st.metric("Profit Factor", f"{profit_factor:.2f}")

    # --- Equity Curve Chart ---
    st.header("Equity Curve Analysis")
    if not equity_curve.empty:
        fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.5, 0.25, 0.25],
            shared_xaxes=True,
            subplot_titles=("Portfolio Value", "Daily Returns", "Drawdown"),
            vertical_spacing=0.05,
        )
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve["total"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#00CC96", width=2),
            ),
            row=1,
            col=1,
        )
        if "returns" in equity_curve.columns:
            fig.add_trace(
                go.Bar(
                    x=equity_curve.index,
                    y=equity_curve["returns"] * 100,
                    name="Daily Returns (%)",
                    marker_color=equity_curve["returns"].apply(
                        lambda x: "#00CC96" if x >= 0 else "#FF6B6B"
                    ),
                ),
                row=2,
                col=1,
            )

        cumulative = equity_curve["total"]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=drawdown,
                fill="tozeroy",
                mode="lines",
                name="Drawdown (%)",
                line=dict(color="#FF6B6B"),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=800, showlegend=True, title_text="Portfolio Performance Analysis"
        )
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # --- Detailed Statistics Tables ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Detailed Performance Metrics")
        if not equity_curve.empty:
            detailed_metrics = calculate_advanced_metrics(
                equity_curve, trade_stats=trade_stats
            )
            metrics_df = pd.DataFrame(
                [{"Metric": k, "Value": v} for k, v in detailed_metrics.items()]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    with col_right:
        st.subheader("Monthly Returns Heatmap")
        if not equity_curve.empty:
            monthly_returns = generate_monthly_returns_table(equity_curve)
            if not monthly_returns.empty:
                fig_heatmap = px.imshow(
                    monthly_returns.iloc[:, :-1],
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    title="Monthly Returns (%)",
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Per-Symbol Performance Section ---
    st.header("Per-Symbol Performance Analysis")
    if not closed_trades.empty:
        st.subheader("Detailed Symbol Statistics")
        symbol_stats_df = calculate_trade_stats_by_symbol(closed_trades)
        if not symbol_stats_df.empty:
            st.dataframe(symbol_stats_df, use_container_width=True)

        st.subheader("Total PnL by Symbol")
        pnl_by_symbol = (
            closed_trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        )
        if not pnl_by_symbol.empty:
            fig_pnl = px.bar(
                pnl_by_symbol,
                x=pnl_by_symbol.index,
                y=pnl_by_symbol.values,
                labels={"x": "Symbol", "y": "Total PnL ($)"},
                title="Profit & Loss Contribution per Symbol",
                color=pnl_by_symbol.values,
                color_continuous_scale=px.colors.diverging.RdYlGn,
                color_continuous_midpoint=0,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.info("No closed trades data available to display per-symbol performance.")

    # --- Risk Monitoring Panel ---
    st.header("Risk Monitoring")
    alerts = monitoring_data.get("recent_alerts", [])
    if alerts:
        st.subheader("Recent Alerts")
        for alert in alerts[-5:]:
            severity = alert.get("severity", "INFO").upper()
            alert_type = alert.get("type", "UNKNOWN").upper()
            message = alert.get("message", "No message provided.")
            st.warning(f"**{severity} | {alert_type}**: {message}")

    system_status = monitoring_data.get("system_status", "UNKNOWN")
    st.info(f"**System Status**: {system_status}")

    # --- Configuration Adjustment Area ---
    st.header("Strategy Configuration")
    with st.expander("Adjust Risk and Sizing Settings"):
        # Correctly reference the nested drawdown parameter from RISK_PARAMS
        current_max_dd_pct = int(
            config.RISK_PARAMS["drawdown_control"]["max_drawdown_threshold"] * 100
        )
        new_max_dd = st.slider(
            "Maximum Drawdown Threshold (%)",
            min_value=5,
            max_value=40,
            value=current_max_dd_pct,
            step=1,
        )

        # Correctly reference the position sizing method from TRADING_PARAMS
        sizing_options = ["signal_weighted", "equal_weight", "risk_parity"]
        current_sizing_method = config.TRADING_PARAMS["position_sizing"][
            "position_size_method"
        ]

        # Ensure current method is in options list to prevent errors
        if current_sizing_method not in sizing_options:
            sizing_options.append(current_sizing_method)

        new_position_method = st.selectbox(
            "Position Sizing Method",
            sizing_options,
            index=sizing_options.index(current_sizing_method),
        )

        if st.button("Apply Changes"):
            st.info("Configuration changes will be applied on the next backtest run.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        f"*Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )


if __name__ == "__main__":
    main()
