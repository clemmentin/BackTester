# Event-Driven Backtester for Quantitative Strategies
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An event-driven backtesting framework in Python for developing and testing quantitative trading strategies, built with a focus on modularity and clear logic.

---

### Key Features

-   **Event-Driven Engine:** A from-scratch implementation that handles market data, signals, orders, and fills sequentially to realistically simulate trading.
-   **Modular 4-Layer Architecture:** Separates signal generation (4-factor alpha model), strategy filtering, risk management, and portfolio construction into distinct, logical layers.
-   **Dynamic Risk Management:** Automatically adjusts portfolio exposure and position count based on market conditions and portfolio drawdown.
-   **Walk-Forward Optimization:** Includes a framework to test strategy robustness and find optimal parameters over rolling time windows.
-   **Efficient Data Pipeline:** Fetches market and economic data with a robust caching system (database-first with a file-based fallback) to speed up runs.
-   **Detailed Performance Analysis:** Generates comprehensive reports and interactive `Plotly` dashboards to visualize backtest results.

---

### Core Tech

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
    *(An interactive HTML report will be generated in the project directory.)*