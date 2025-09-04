import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def create_equity_curve_dataframe(all_holdings):
    df = pd.DataFrame(all_holdings)
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    df['positions_value'] = df.drop(columns=['cash', 'total']).sum(axis=1)
    df['returns'] = df['total'].pct_change()
    df.dropna(inplace=True)
    return df


def calculate_performance_metrics(equity_curve, periods_per_year=252):
    total_return = (equity_curve['total'].iloc[-1] / equity_curve['total'].iloc[0]) - 1
    mean_daily_return = equity_curve['returns'].mean()
    std_daily_return = equity_curve['returns'].std()
    sharpe_ratio = mean_daily_return / std_daily_return
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(periods_per_year)
    high_water_mark = equity_curve['total'].cummax()
    drawdown = (equity_curve['total'] - high_water_mark) / high_water_mark
    max_drawdown = drawdown.min()
    annualized_volatility = std_daily_return * np.sqrt(periods_per_year)
    metrics = {
        'Total Return (%)': f"{total_return * 100:.2f}",
        'Annualized Volatility (%)': f"{annualized_volatility * 100:.2f}",
        'Sharpe Ratio': f"{annualized_sharpe_ratio:.2f}",
        'Maximum Drawdown (%)': f"{max_drawdown * 100:.2f}"
    }
    return metrics


def plot_equity_curve(equity_curve, title='Strategy Performance'):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(equity_curve.index, equity_curve['total'], label='Total Equity', color='blue', linewidth=2)
    ax.fill_between(equity_curve.index, equity_curve['cash'], equity_curve['total'],alpha=0.3, color='orange', label='Positions Value')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='upper left')
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.show()