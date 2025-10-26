# Event-Driven Backtester for Quantitative Strategies
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced event-driven backtesting framework in Python, designed to develop and test adaptive quantitative trading strategies. The system's core is an alpha engine and risk manager that work together to respond to changing market conditions.

---

### Key Features

-   **Multi-Factor Alpha Engine:** Generates signals from a 4-factor model (`Reversal`, `Liquidity`, `Momentum`, `Price`), adjusting its logic based on the current market state (e.g., Bull, Bear, Crisis).

-   **Bayesian Learning Core:** Features a **Bayesian EV (Expected Value) Calculator** that continuously learns from past trade outcomes. This allows it to more accurately predict the profitability and risk of future signals.

-   **Proactive Risk Management:**
    -   **Market State Detection:** Uses statistical models (like GARCH for volatility) to classify the market into clear, understandable states.
    -   **Smooth Risk Scaling:** Instead of making abrupt changes, the system smoothly adjusts portfolio exposure based on how much the current market deviates from its historical patterns.

-   **Walk-Forward Optimization (WFO):** Includes a robust WFO framework to optimize the strategy's key sensitivity parameters, ensuring it performs well on new, unseen data.

-   **High-Performance Engine:**
    -   **Parallelized Calculation:** Uses multi-core processing to calculate alpha factors concurrently, significantly speeding up backtests.
    -   **Persistent Caching:** Implements smart caching for computationally heavy tasks, making iterative testing much faster.

-   **Advanced Diagnostics & Logging:**
    -   Generates comprehensive performance reports and visualizations.
    -   Produces a detailed trade log (`ml_training_data.csv`) with rich data on the market conditions for each trade, creating a perfect dataset for future machine learning analysis.

---

### Core Technology

**Python | Pandas | NumPy | Statsmodels | Plotly | scikit-optimize | yfinance**

---

### How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure API Key (Optional):**
    To fetch macroeconomic data, create a `.env` file with your FRED API key:
    ```bash
    echo "FRED_API_KEY=your_api_key_here" > .env
    ```

4.  **Run the backtest:**
    ```bash
    python main.py
    ```

### Outputs

The files generated depend on whether you run a single backtest or a Walk-Forward Optimization (WFO), as set in `strategy_config.py`.

| Run Mode          | Output Files                      | Note                                                        |
| :---------------- | :-------------------------------- | :---------------------------------------------------------- |
| **Single Backtest** | `detailed_backtest_analysis.html` | Interactive Plotly dashboard for the entire period.         |
|                   | `equity_curve.png`                | Static equity curve plot.                                   |
| **WFO Run**       | `wfo_results_...csv`              | Summary of optimized parameters for each fold.              |
|                   | `trade_details_wfo_run.csv`       | Full trade list from all out-of-sample periods.             |
| **All Runs**      | `ml_training_data.csv`            | Log of all signal outcomes for learning and future analysis.|