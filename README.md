# Event-Driven Backtester for Quantitative Strategies
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An event-driven backtesting framework in Python for developing and testing quantitative trading strategies. The system is designed to be adaptive, with a core engine that adjusts its behavior based on changing market conditions.

---

### Core Components

-   **Multi-Factor Model:** Generates trading signals using a combination of four factors: `Reversal`, `Liquidity`, `Momentum`, and `Price`. The logic for each factor is adjusted based on the current market state (e.g., Bull, Bear, Crisis).

-   **Bayesian Learning Module:** Includes a module that attempts to predict the Expected Value (EV) of signals by learning from the outcomes of past trades.

-   **Risk Management:**
    -   **Market State Detection:** Uses statistical models like GARCH to classify the market into different states.
    -   **Dynamic Exposure:** Smoothly adjusts the total portfolio investment level based on how much the current market deviates from historical patterns.

-   **Walk-Forward Optimization (WFO):** Provides a framework to test and optimize the strategy's key parameters on out-of-sample data, aiming to ensure its robustness.

-   **Performance & Diagnostics:**
    -   **Parallel Processing:** Uses multiple CPU cores to speed up the calculation of alpha factors.
    -   **Data Caching:** Saves the results of time-consuming calculations to make subsequent runs faster.
    -   **Analysis Tools:** Generates performance reports, visualizations, and a detailed log of all trades (`ml_training_data.csv`) for further analysis.

---

### Technologies Used

Python | Pandas | NumPy | Statsmodels | Plotly | scikit-optimize | yfinance

---

### How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up the environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure API Key (Optional):**
    To fetch macroeconomic data from FRED, create a `.env` file with your API key:
    ```bash
    echo "FRED_API_KEY=your_api_key_here" > .env
    ```

4.  **Run the backtest:**
    ```bash
    python main.py
    ```

### Outputs

The generated files depend on the run mode set in `strategy_config.py`.

| Run Mode          | Output Files                      | Description                                                 |
| :---------------- | :-------------------------------- | :---------------------------------------------------------- |
| **Single Backtest** | `detailed_backtest_analysis.html` | An interactive dashboard of the backtest performance.       |
|                   | `equity_curve.png`                | A static plot of the strategy's equity curve.               |
| **WFO Run**       | `wfo_results_...csv`              | A summary of optimized parameters from each WFO fold.       |
|                   | `trade_details_wfo_run.csv`       | A complete list of trades from all out-of-sample periods.   |
| **All Runs**      | `ml_training_data.csv`            | A detailed log of all trade outcomes for analysis.          |

---

### Project Findings & Future Work

This project started as a backtesting platform and has become a tool for deeper research. The process of building and testing the system has led to some important findings that guide the next steps.

---

### Key Findings & Future Research Directions

This project is not just a backtesting framework but an active research platform. The process of building and rigorously testing the system has led to a key insight that guides its future development.

#### Key Finding: A Signal Enhancement Engine

Our most critical finding comes from a direct A/B test, comparing the predictive power of the raw alpha signals against the signals processed by the full, adaptive `AlphaEngine` (which includes a Bayesian EV model, IC-dynamic weighting, and market-regime context).

The results are striking.

1.  **Without the Engine's processing**, the raw alpha factors show only a weak, noisy correlation with future returns. The trendlines are nearly flat, indicating low predictive power.

    *   **[View Diagnostic Plots for Raw Signals (EV Off)](./docs/images/factor_diagnose_without_ev.png)**

2.  **With the full Engine enabled**, the system acts as a powerful **signal enhancement engine**. It successfully filters noise and amplifies the underlying alpha. The correlation between the final processed signal scores and actual returns becomes dramatically stronger and statistically significant.

    *   **[View Diagnostic Plots for Processed Signals (EV On)](./docs/images/factor_diagnose_with_ev.png)**

This A/B test empirically demonstrates that the adaptive components of the `AlphaEngine` are highly effective at transforming low-grade, noisy inputs into high-quality, actionable trading signals.

#### The Next Frontier: From Signal Generation to Optimal Execution

Despite generating high-quality signals, the strategy's overall performance is still constrained. This indicates that the current, simplistic execution model is the next critical bottleneck. The strategic edge gained in signal generation is not being fully capitalized upon during the trade execution phase.

---

### Primary Research Direction: Adaptive Execution via Hierarchical Reinforcement Learning

This leads directly to my primary research proposal, which focuses on designing an intelligent execution agent to unlock the full potential of these high-quality signals.

*   **Hypothesis:** *How* a trade is executed can be as important as *what* is traded. The optimal execution policy must adapt to the prevailing macro-market regime and intraday price action.

*   **Proposed Method:** I plan to develop a two-layer Reinforcement Learning (RL) framework:
    1.  **A Strategic Layer:** Identifies the current macro-market state (the "weather").
    2.  **A Tactical Layer:** Deploys a specialized RL agent trained to find the optimal second-by-second trading strategy for that specific "weather."

*   **Goal:** To prove that a sophisticated execution agent can generate significant "execution alpha" by translating strong, long-term signals into a series of superiorly timed, short-term trades. 