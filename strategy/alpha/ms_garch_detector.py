import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from numpy.linalg import LinAlgError

import config


class MsGarchDetector:
    """
    Detects market volatility regimes using a Markov-Switching GARCH model.
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        # Get lookback period from config, with a default fallback.
        self.lookback_days = kwargs.get("garch_lookback_days", 1000)
        # Cache stores results with a date object as the key.
        self._cache = {}

    def get_volatility_state(
        self, market_returns: pd.Series, timestamp: pd.Timestamp
    ) -> str:
        """
        Calculates the market volatility state (high/low) for a given timestamp.
        Results are cached daily to avoid re-computation.
        """
        cache_key = timestamp.date()

        # Return cached result if available for the day.
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Prepare data and check if there's enough history to proceed.
        required_len = self.lookback_days // 2
        data = market_returns.loc[:timestamp].tail(self.lookback_days).dropna()
        if len(data) < required_len:
            self.logger.warning(f"Insufficient data for GARCH model at {cache_key}.")
            return "unknown"

        try:
            # Define and fit the GARCH model. This is the computationally expensive step.
            model = MarkovAutoregression(
                data.values * 100, k_regimes=2, order=1, switching_variance=True
            )
            np.random.seed(config.GLOBAL_RANDOM_SEED)
            results = model.fit(search_reps=10, disp=False)

            # Check if the model returned the expected variance parameters.
            vol_param_1 = "sigma2[0]"
            vol_param_2 = "sigma2[1]"
            if not (vol_param_1 in results.params and vol_param_2 in results.params):
                self.logger.error(
                    f"Could not find variance parameters in GARCH results for {cache_key}."
                )
                return "unknown"

            # Check if variance values are valid.
            vols = [results.params[vol_param_1], results.params[vol_param_2]]
            if any(v <= 0 for v in vols):
                self.logger.warning(
                    f"GARCH model returned non-positive variance for {cache_key}."
                )
                return "unknown"

            # Determine which regime corresponds to high volatility.
            high_vol_state_index = np.argmax(vols)

            # Get the model's predicted probabilities for each regime.
            probabilities = results.smoothed_marginal_probabilities
            if (
                probabilities is None
                or probabilities.ndim != 2
                or probabilities.shape[0] == 0
            ):
                probabilities = results.filtered_marginal_probabilities
                if (
                    probabilities is None
                    or probabilities.ndim != 2
                    or probabilities.shape[0] == 0
                ):
                    self.logger.error(
                        f"GARCH model returned invalid probabilities for {cache_key}."
                    )
                    return "unknown"

            prob_high_vol = probabilities[-1, high_vol_state_index]
            current_state = "high_vol" if prob_high_vol > 0.7 else "low_vol"

            # Store the successful result in the cache.
            self._cache[cache_key] = current_state
            return current_state

        except (LinAlgError, ValueError) as e:
            # Catch critical errors during model fitting.
            self.logger.error(
                f"A critical error occurred in GARCH fitting for {cache_key}: {e}"
            )
            return "unknown"
