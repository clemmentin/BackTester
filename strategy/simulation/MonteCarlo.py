from typing import Dict
import numpy as np
import pandas as pd


class MonteCarloSimulator:
    def __init__(self, returns_data: pd.Series):
        self.returns_data = returns_data.dropna()
        if self.returns_data.empty:
            raise ValueError(
                "Input returns data for Monte Carlo simulation cannot be empty."
            )

        self.mean_return = self.returns_data.mean()
        self.std_return = self.returns_data.std()

    def simulate_price_paths(
        self, initial_price: float, num_days: int, num_simulations: int = 1000
    ) -> np.ndarray:
        """
        Simulates future price paths using the historical return distribution.
        """
        # Generate random returns based on historical mean and std
        random_returns = np.random.normal(
            self.mean_return, self.std_return, size=(num_simulations, num_days)
        )

        # Create price paths
        price_paths = np.zeros((num_simulations, num_days + 1))
        price_paths[:, 0] = initial_price

        for day in range(num_days):
            price_paths[:, day + 1] = price_paths[:, day] * (1 + random_returns[:, day])

        return price_paths

    def calculate_risk_metrics(
        self, price_paths: np.ndarray, confidence_level: float = 0.05
    ) -> Dict:
        """
        Calculates key risk metrics from the simulated price paths.
        """
        final_prices = price_paths[:, -1]
        initial_price = price_paths[0, 0]

        # Calculate simulated returns
        simulated_returns = (final_prices / initial_price) - 1

        # Value at Risk (VaR)
        var = np.percentile(simulated_returns, confidence_level * 100)

        # Conditional Value at Risk (CVaR) or Expected Shortfall
        cvar = simulated_returns[simulated_returns <= var].mean()

        # Probability of loss
        prob_loss = (simulated_returns < 0).mean()

        return {
            "expected_return": simulated_returns.mean(),
            "volatility": simulated_returns.std(),
            "var_95": var,
            "cvar_95": cvar,
            "probability_of_loss": prob_loss,
        }
