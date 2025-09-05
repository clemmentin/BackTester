# Quantitative Backtesting Engine & Strategy

A quantitative research project featuring a modular, event-driven backtesting engine built from scratch in Python, used to validate a trading strategy based on a GARCH volatility model developed in SAS.

---

### Key Features

-   **Modular Engine (Python):** Event-driven architecture built from the ground up, allowing for easy extension and testing of different strategies, data sources, or execution models.
-   **Econometric Model (SAS):** Replication of a published academic study to forecast S&P 500 volatility using a GARCH(1,1) model.
-   **Dual-Filter Strategy:** A quantitative strategy that combines the GARCH volatility forecast (risk filter) with a 200-day moving average (trend filter) to dynamically manage portfolio exposure.

---

### Tech Stack

-   **Languages:** Python, SAS
-   **Core Libraries:** Pandas, NumPy, Matplotlib

---

### How to Run

1.  **Prerequisites:** Python 3.x and required libraries (`pip install pandas numpy matplotlib`).
2.  **Clone the repository:** `git clone [your-repo-url]`
3.  **Run the backtest:**
    ```bash
    python engine.py
    ```

The script will execute the backtest using the included data files and generate a performance summary and an equity curve chart.