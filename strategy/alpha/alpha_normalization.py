import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging
from strategy.contracts import RawAlphaSignalDict


class AlphaNormalizer:
    """
    Performs cross-sectional normalization and neutralization for alpha signals.
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.normalization_method = kwargs.get("normalization_method", "rank")
        self.winsorize_percentile = kwargs.get("winsorize_percentile", 0.02)
        self.neutralize_beta = kwargs.get("neutralize_beta", True)
        self.min_samples = kwargs.get("min_samples_for_normalization", 10)

        # Beta cache parameters loaded from config.
        self._beta_cache = {}
        self._cache_max_age_days = kwargs.get("beta_cache_max_age_days", 20)
        self._cache_max_size = kwargs.get("beta_cache_max_size", 100)

    def normalize_factors(
        self,
        alpha_signals: RawAlphaSignalDict,
        market_data: Optional[pd.DataFrame] = None,
    ) -> RawAlphaSignalDict:
        """
        The main pipeline to normalize raw alpha signals.
        """
        # ... (This function's internal comments and logic are already clean)
        if not alpha_signals:
            return alpha_signals

        symbols = list(alpha_signals.keys())
        scores = np.array([alpha_signals[s].score for s in symbols])
        scores = self._validate_and_clean_scores(scores)

        if len(scores) < self.min_samples:
            return alpha_signals

        scores = self._winsorize_percentile(scores)
        normalized_scores = self._apply_normalization(scores)

        if self.neutralize_beta and market_data is not None:
            normalized_scores = self._beta_neutralize(
                normalized_scores, symbols, market_data
            )

        norm_quality = self._assess_normalization_quality(scores, normalized_scores)

        for i, symbol in enumerate(symbols):
            alpha_signals[symbol].score = np.clip(normalized_scores[i], -1.0, 1.0)
            alpha_signals[symbol].confidence *= norm_quality

        return alpha_signals

    def _get_betas_with_cache(
        self, symbols: List[str], market_data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        Calculates or retrieves from cache the market betas for a list of symbols.
        """
        cache_key = tuple(sorted(symbols))
        latest_time = market_data.index.get_level_values("timestamp").max()

        # Check cache for existing, non-expired beta values.
        if cache_key in self._beta_cache:
            cached_betas, cache_time = self._beta_cache[cache_key]
            if (latest_time - cache_time).days < self._cache_max_age_days:
                return cached_betas

        # If not in cache or expired, calculate new betas.
        betas = self._calculate_betas(symbols, market_data)

        # Update cache with the new values.
        if betas is not None:
            self._beta_cache[cache_key] = (betas, latest_time)
            # Ensure the cache does not grow indefinitely.
            if len(self._beta_cache) > self._cache_max_size:
                oldest_key = min(self._beta_cache, key=lambda k: self._beta_cache[k][1])
                del self._beta_cache[oldest_key]

        return betas

    def _calculate_betas(
        self, symbols: List[str], market_data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        Calculates market beta for each symbol using SPY as a market proxy.
        This function is vectorized for performance.
        """
        # 1. Check for presence of market proxy (SPY).
        if "SPY" not in market_data.index.get_level_values("symbol"):
            self.logger.debug("SPY not in market_data, cannot calculate betas.")
            return None

        # 2. Reshape price data for vectorized operations.
        prices = market_data["close"].unstack(level="symbol")
        if "SPY" not in prices.columns:
            self.logger.debug("SPY price data not available for the period.")
            return None

        valid_symbols = [s for s in symbols if s in prices.columns]
        if not valid_symbols:
            self.logger.debug("None of the requested symbols found in price data.")
            return None

        # 3. Calculate returns and check for sufficient data length.
        returns = prices.pct_change()
        if len(returns.dropna()) < 30:
            self.logger.debug("Insufficient data for beta calculation.")
            return None

        market_returns = returns["SPY"]
        stock_returns = returns[valid_symbols]

        # 4. Calculate market variance, handling the case of zero variance.
        market_variance = market_returns.var()
        if market_variance < 1e-9:
            return np.ones(len(symbols))

        # 5. Calculate covariance and beta in a single vectorized step.
        # Note: DataFrame.cov does not accept a Series as "other" argument; build covariance matrix instead.
        cols = list(dict.fromkeys(valid_symbols + ["SPY"]))
        cov_matrix = returns[cols].cov()
        covariance_vector = cov_matrix.loc[valid_symbols, "SPY"]
        beta_vector = covariance_vector / market_variance
        beta_map = beta_vector.to_dict()

        # 6. Build the final array, using a default beta of 1.0 for any missing symbols.
        final_betas = np.array([beta_map.get(s, 1.0) for s in symbols])

        # 7. Clip results to a reasonable range.
        return np.clip(final_betas, -3.0, 3.0)

    # ... (other helper methods like _validate_and_clean_scores, etc., remain the same)
    def _validate_and_clean_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        invalid_mask = ~np.isfinite(scores)
        if np.any(invalid_mask):
            scores = np.where(invalid_mask, 0.0, scores)
        return scores

    def _winsorize_percentile(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) < 3:
            return scores
        lower_bound = np.percentile(scores, self.winsorize_percentile * 100)
        upper_bound = np.percentile(scores, (1 - self.winsorize_percentile) * 100)
        return np.clip(scores, lower_bound, upper_bound)

    def _apply_normalization(self, scores: np.ndarray) -> np.ndarray:
        if self.normalization_method == "zscore":
            return self._zscore_normalize(scores)
        elif self.normalization_method == "minmax":
            return self._minmax_normalize(scores)
        else:
            return self._rank_normalize(scores)

    def _zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) < 2 or np.std(scores) < 1e-8:
            return scores - np.mean(scores)
        return np.clip((scores - np.mean(scores)) / np.std(scores, ddof=1), -3.0, 3.0)

    def _rank_normalize(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) < 2:
            return scores
        ranks = stats.rankdata(scores, method="average")
        return (
            2.0 * (ranks - 1) / (len(ranks) - 1) - 1.0
            if len(ranks) > 1
            else np.array([0.0])
        )

    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) < 2:
            return scores
        min_val, max_val = np.min(scores), np.max(scores)
        if max_val > min_val + 1e-8:
            return 2.0 * ((scores - min_val) / (max_val - min_val)) - 1.0
        return np.zeros_like(scores)

    def _beta_neutralize(
        self, scores: np.ndarray, symbols: List[str], market_data: pd.DataFrame
    ) -> np.ndarray:
        betas = self._get_betas_with_cache(symbols, market_data)
        if betas is None or len(betas) != len(scores) or np.std(betas) < 1e-4:
            return scores

        correlation = np.corrcoef(scores, betas)[0, 1]
        if np.isnan(correlation) or np.abs(correlation) < 0.01:
            return scores

        slope = correlation * (np.std(scores) / np.std(betas))
        neutralized = scores - slope * (betas - np.mean(betas))
        return neutralized - np.mean(neutralized)

    def _assess_normalization_quality(
        self, original: np.ndarray, normalized: np.ndarray
    ) -> float:
        if len(original) < 3 or np.std(original) < 1e-8 or np.std(normalized) < 1e-8:
            return 0.5
        rank_corr, _ = stats.spearmanr(original, normalized)
        return np.clip(abs(rank_corr), 0.3, 1.0) if not np.isnan(rank_corr) else 0.5
