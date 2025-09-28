# Event-Driven Backtesting Engine for Quantitative Strategies
An institutional-grade, event-driven backtesting framework built from scratch in Python. It is designed for the rigorous research, development, and validation of sophisticated quantitative trading strategies.

---

### Key Features

-   **Intelligent Alpha Engine:** A central "brain" that fuses signals from multiple alpha sources (e.g., momentum, technicals) using dynamic, market-regime-aware weighting. It leverages parallel processing for high-performance computation.

-   **Proactive Risk Management:** A multi-factor `RiskManager` that calculates a composite risk score to determine a global `RiskMode` (e.g., `Normal`, `Defensive`), which directly influences position sizing and signal generation.

-   **Robust Optimization Suite:** Features **Walk-Forward Optimization (WFO)** and **Bayesian Optimization** (`scikit-optimize`) to rigorously test strategy robustness and prevent overfitting.

-   **Interactive Analysis Dashboard:** A comprehensive dashboard built with `Streamlit` and `Plotly` for in-depth visualization of performance, risk metrics, and simulation results.

-   **Resilient Data Pipeline:** A full-featured data pipeline with **PostgreSQL** integration for efficient data fetching, feature engineering, and caching.

---

### Tech Stack

-   **Core Engine:** Python, Multiprocessing
-   **Quantitative & Data:** Pandas, NumPy, TA-Lib, scikit-optimize, ARCH
-   **Database & Pipeline:** PostgreSQL, SQLAlchemy, yfinance, fredapi
-   **Visualization & Web:** Streamlit, Plotly, Matplotlib
-   **Development:** Git

---

### How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/clemmentin/BackTester.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment:**
    Create a `.env` file in the root directory and set up your database credentials. 

4.  **Set up the database:**
    This command will create the necessary database and tables.
    ```bash
    python data_pipeline/db_setup.py
    ```
5.  **Run a full data pipeline and backtest:**
    This script will handle data ingestion, feature engineering, and execute a backtest using the default configuration.
    ```bash
    python main.py
    ```