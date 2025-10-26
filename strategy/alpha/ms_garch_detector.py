import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from numpy.linalg import LinAlgError
import os
from typing import Dict
import config


class MsGarchDetector:
    """ """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        garch_params = kwargs.get("market_detector_params", {})
        self.lookback_days = garch_params.get("garch_lookback_days", 1000)
        self.search_reps = garch_params.get("garch_search_reps", 10)
        self.high_vol_threshold = garch_params.get("garch_high_vol_threshold", 0.7)

        garch_const = config.get_trading_param("GARCH_CONSTANTS", default={}).get(
            "model", {}
        )
        self.k_regimes = garch_const.get("k_regimes", 2)
        self.order = garch_const.get("order", 1)
        self.return_scaling = garch_const.get("return_scaling", 100)

        # --- NEW: Persistent Caching ---
        self.cache_dir = "./cache"
        self.cache_path = os.path.join(self.cache_dir, "garch_states_cache.csv")
        # Update this version if GARCH logic changes to invalidate the old cache
        self.cache_version = "garch_v1.0"
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """NEW: Loads GARCH states from a persistent file cache."""
        if not os.path.exists(self.cache_path):
            return {}

        try:
            with open(self.cache_path, "r") as f:
                first_line = f.readline().strip()
                if first_line != f"version,{self.cache_version}":
                    self.logger.warning(
                        f"GARCH cache version mismatch. Expected {self.cache_version}, found different. Discarding cache."
                    )
                    return {}

            df = pd.read_csv(self.cache_path, header=1, index_col=0, parse_dates=True)
            # Convert loaded DataFrame into the dictionary format {date: state}
            cache_dict = {
                idx.date(): state for idx, state in df["state"].to_dict().items()
            }
            self.logger.info(f"Loaded {len(cache_dict)} GARCH states from cache.")
            return cache_dict

        except Exception as e:
            self.logger.error(f"Failed to load GARCH cache: {e}")
            return {}

    def _save_cache(self):
        """NEW: Saves the current GARCH states to the persistent cache."""
        if not self._cache:
            return

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Convert dict {date: state} to a DataFrame for saving
            df = pd.DataFrame.from_dict(self._cache, orient="index", columns=["state"])
            df.index.name = "date"
            df.sort_index(inplace=True)

            with open(self.cache_path, "w") as f:
                f.write(f"version,{self.cache_version}\n")
                df.to_csv(f, header=True)
            self.logger.debug(f"Saved GARCH cache to {self.cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to save GARCH cache: {e}")

    def get_volatility_state(
        self, market_returns: pd.Series, timestamp: pd.Timestamp
    ) -> str:
        """

        Args:
          market_returns: pd.Series: 
          timestamp: pd.Timestamp: 

        Returns:

        """
        cache_key = timestamp.date()
        if cache_key in self._cache:
            return self._cache[cache_key]

        required_len = self.lookback_days // 2
        data = market_returns.loc[:timestamp].tail(self.lookback_days).dropna()
        if len(data) < required_len:
            self.logger.warning(f"Insufficient data for GARCH model at {cache_key}.")
            return "unknown"

        try:
            # Use parameters for model definition and fitting
            model = MarkovAutoregression(
                data.values * self.return_scaling,
                k_regimes=self.k_regimes,
                order=self.order,
                switching_variance=True,
            )
            np.random.seed(config.GLOBAL_RANDOM_SEED)
            results = model.fit(search_reps=self.search_reps, disp=False)

            param_names = getattr(results.model, "param_names", None)
            params = results.params
            sigma_indices = []
            if param_names:
                for i, name in enumerate(param_names):
                    if isinstance(name, str) and name.startswith("sigma2"):
                        sigma_indices.append(i)

            if len(sigma_indices) < 2:
                self.logger.error(
                    f"Could not find variance parameters in GARCH results for {cache_key}."
                )
                return "unknown"

            sigma_indices = sorted(sigma_indices)[:2]
            vols = [float(params[i]) for i in sigma_indices]

            if any(v <= 0 or not np.isfinite(v) for v in vols):
                self.logger.warning(
                    f"GARCH model returned invalid variance for {cache_key}."
                )
                return "unknown"

            high_vol_state_index = np.argmax(vols)
            probabilities = getattr(results, "smoothed_marginal_probabilities", None)
            if probabilities is None:
                probabilities = getattr(
                    results, "filtered_marginal_probabilities", None
                )

            if probabilities is None or not hasattr(probabilities, "shape"):
                self.logger.error(
                    f"GARCH model returned invalid probabilities for {cache_key}."
                )
                return "unknown"

            probabilities = np.asarray(probabilities)
            if probabilities.ndim == 2 and probabilities.shape[1] == self.k_regimes:
                # Use parameter for probability threshold
                prob_high_vol = probabilities[-1, high_vol_state_index]
                current_state = (
                    "high_vol" if prob_high_vol > self.high_vol_threshold else "low_vol"
                )
                # MODIFIED: Update cache and save to disk
                self._cache[cache_key] = current_state
                self._save_cache()
                return current_state
            else:
                self.logger.error(
                    f"GARCH probabilities have unexpected shape {probabilities.shape} for {cache_key}."
                )
                return "unknown"

        except (LinAlgError, ValueError) as e:
            self.logger.error(
                f"A critical error occurred in GARCH fitting for {cache_key}: {e}"
            )
            return "unknown"
