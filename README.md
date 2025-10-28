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

### Key Findings & Future Research Directions

This project serves as an active research platform, and its development has uncovered several key insights that open up multiple avenues for future research.

#### Key Finding: The Challenge of Low Signal-to-Noise Ratio

Our most critical finding comes from a direct diagnostic analysis of the raw alpha signals ([see plots](./docs/images/)). The analysis reveals two core challenges of quantitative trading:

1.  **Alpha Signals are Inherently Weak and Noisy:** Raw signals show a statistically significant but weak correlation with future returns (Spearman's ρ ≈ 0.05-0.06).
2.  **Complex Models Struggle with Noisy Inputs:** A sophisticated Bayesian learning model, designed to predict Expected Value (EV), currently fails to generate reliable forecasts from this noisy data, showing near-zero correlation with actual returns.

This empirically demonstrates that advanced modeling alone cannot compensate for a lack of fundamental predictive power in the input signals. This leads to several exciting, open-ended research questions.

---

### Potential Research Directions

Based on these findings, I am actively exploring the following research directions to enhance the system's intelligence and robustness:

#### 1. Intelligent Execution with Reinforcement Learning

*   **Question:** Instead of trying to perfect a long-term forecast, can we generate "execution alpha" by learning the optimal *short-term* trading policy?
*   **Hypothesis:** The small edge in the raw signals can be dramatically amplified by an intelligent agent that adapts to intraday market dynamics and macro-economic regimes.
*   **Potential Method:** Develop a **Hierarchical Reinforcement Learning** framework. A strategic layer would identify the market regime (inspired by works on MMC clustering), and a tactical RL agent would learn the optimal execution policy for each regime (inspired by works on Policy Gradients for execution).

#### 2. Advanced Bayesian Workflow and Diagnostics

*   **Question:** Why is the current Bayesian EV model failing, and how can we build a more reliable one?
*   **Hypothesis:** The model's failure is not a failure of Bayesian methods themselves, but a result of a flawed modeling process (e.g., poor choice of priors, model misspecification).
*   **Potential Method:** Implement a systematic **Bayesian Workflow** for model development. This would involve:
    *   Applying advanced diagnostic tools like **Visual Predictive Checks (VPC)** and **Simulation-Based Calibration (SBC)** to deeply understand the current model's failure modes.
    *   Exploring more **structured, informative priors** that can better incorporate domain knowledge about the alpha signals.
    *   Investigating methods to handle the **non-stationarity (concept drift)** of financial data within the Bayesian framework.

#### 3. Alpha Factor Engineering & Refinement

*   **Question:** Can we improve the foundational predictive power of the raw alpha signals themselves?
*   **Hypothesis:** The current alpha factors can be refined or combined in non-linear ways to create stronger, more robust signals.
*   **Potential Method:**
    *   Conduct a deep-dive analysis into the failure of the `Reversal` factor.
    *   Explore feature engineering techniques to create interaction terms between factors.
    *   Use machine learning models (e.g., Gradient Boosting Trees) to explore non-linear relationships between raw indicators and future returns.