import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal
from strategy.alpha.market_detector import MarketState, MarketRegime


class ReversalAlphaModule:
    """ """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        params = kwargs.get("reversal_alpha_params", {})

        # MODIFIED: Removed fixed lookback periods. They will now be determined dynamically.
        self.extreme_loser_threshold = kwargs.get(
            "extreme_loser_threshold", params.get("extreme_loser_threshold", -0.12)
        )

        self.regime_lookback_weights = kwargs.get(
            "regime_lookback_weights",
            params.get(
                "regime_lookback_weights",
                {
                    "CRISIS": {"short": 0.80, "long": 0.20},
                    "VOLATILE": {"short": 0.75, "long": 0.25},
                    "BEAR": {"short": 0.65, "long": 0.35},
                    "RECOVERY": {"short": 0.55, "long": 0.45},
                    "NORMAL": {"short": 0.50, "long": 0.50},
                    "BULL": {"short": 0.40, "long": 0.60},
                    "STRONG_BULL": {"short": 0.30, "long": 0.70},
                },
            ),
        )

        self.score_weights = {
            "strength": 0.70,
            "quality": 0.30,
        }

        self.volume_surge_threshold = kwargs.get(
            "volume_surge_threshold", params.get("volume_surge_threshold", 1.15)
        )
        self.min_confidence = kwargs.get(
            "min_confidence", params.get("min_confidence", 0.30)
        )
        self.min_avg_volume = kwargs.get(
            "min_avg_volume", params.get("min_avg_volume", 300_000)
        )
        self.min_dollar_volume = kwargs.get(
            "min_dollar_volume", params.get("min_dollar_volume", 15_000_000)
        )
        self.regime_config = kwargs.get(
            "regime_config",
            params.get(
                "regime_config",
                {
                    "CRISIS": {"strength": 2.0, "threshold_mult": 1.3},
                    "VOLATILE": {"strength": 1.8, "threshold_mult": 1.2},
                    "BEAR": {"strength": 1.5, "threshold_mult": 1.1},
                    "RECOVERY": {"strength": 1.2, "threshold_mult": 1.0},
                    "NORMAL": {"strength": 0.9, "threshold_mult": 1.15},
                    "BULL": {"strength": 0.5, "threshold_mult": 1.4},
                    "STRONG_BULL": {"strength": 0.3, "threshold_mult": 1.8},
                },
            ),
        )
        self.logger.info(
            "ReversalAlpha [V8-AdaptiveLookback]: Using dynamic lookbacks based on MarketState."
        )

    def _get_dynamic_lookbacks(self, market_state: MarketState) -> Tuple[int, int]:
        """NEW: Determines reversal lookback periods based on the market state.

        Args:
          market_state: MarketState: 

        Returns:

        """
        base_short, base_long = 21, 42

        # Adjust based on volatility regime
        if market_state.volatility_regime == "high_vol":
            multiplier = 0.75  # Shorter lookbacks in high vol
        elif market_state.volatility_regime == "low_vol":
            multiplier = 1.15  # Longer lookbacks in low vol
        else:
            multiplier = 1.0

        # Further adjust based on market regime
        if market_state.regime in [MarketRegime.CRISIS, MarketRegime.BEAR]:
            multiplier *= 0.8
        elif market_state.regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
            multiplier *= 1.2

        short_lookback = int(np.clip(base_short * multiplier, 10, 40))
        long_lookback = int(np.clip(base_long * multiplier, 20, 80))

        return short_lookback, long_lookback

    def _calculate_reversal_scores(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        lookback_period: int,
        liquid_symbols: pd.Index,
        threshold_multiplier: float,
    ) -> pd.DataFrame:
        """

        Args:
          prices: pd.DataFrame: 
          volumes: pd.DataFrame: 
          timestamp: pd.Timestamp: 
          lookback_period: int: 
          liquid_symbols: pd.Index: 
          threshold_multiplier: float: 

        Returns:

        """
        if lookback_period > len(prices):
            return pd.DataFrame()

        lookback_date = prices.index[prices.index.get_loc(timestamp) - lookback_period]
        start_prices = prices.loc[lookback_date, liquid_symbols]
        end_prices = prices.loc[timestamp, liquid_symbols]

        valid_prices = start_prices.gt(0) & end_prices.notna()
        if not valid_prices.any():
            return pd.DataFrame()

        short_return = (end_prices[valid_prices] / start_prices[valid_prices]) - 1.0
        df = pd.DataFrame({"return": short_return})

        dynamic_loser_threshold = self.extreme_loser_threshold * threshold_multiplier
        df = df[df["return"] < dynamic_loser_threshold]
        if df.empty:
            return pd.DataFrame()

        df["strength_score"] = np.clip(np.power(-df["return"] * 2.5, 0.8), 0, 1.0)
        volume_slice = volumes.loc[lookback_date:timestamp, df.index]
        avg_reversal_volume = volume_slice.mean()
        long_term_avg_volume = (
            volumes[df.index]
            .rolling(window=60)
            .mean()
            .loc[timestamp]
            .replace(0, np.nan)
        )
        volume_surge_ratio = avg_reversal_volume / long_term_avg_volume
        df["quality_score"] = (
            (volume_surge_ratio / self.volume_surge_threshold)
            .clip(0.6, 1.4)
            .fillna(0.8)
        )
        df["score"] = (
            df["strength_score"] * self.score_weights["strength"]
            + df["quality_score"] * self.score_weights["quality"]
        )
        return df

    def calculate_batch_reversal_trend_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_state: MarketState,  # MODIFIED: Now accepts the full MarketState object
    ) -> Dict[str, RawAlphaSignal]:
        """

        Args:
          prices: pd.DataFrame: 
          volumes: pd.DataFrame: 
          timestamp: pd.Timestamp: 
          market_state: MarketState: 
          # MODIFIED: Now accepts the full MarketState object: 

        Returns:

        """

        # NEW: Dynamically determine lookback periods for this specific timestamp.
        short_lookback, long_lookback = self._get_dynamic_lookbacks(market_state)
        max_lookback = max(short_lookback, long_lookback)

        if (
            prices.empty
            or timestamp not in prices.index
            or prices.index.get_loc(timestamp) < max_lookback + 5
        ):
            return {}

        market_regime = market_state.regime.value
        regime_params = self.regime_config.get(market_regime.upper(), {})
        regime_strength = regime_params.get("strength", 1.0)
        threshold_multiplier = regime_params.get("threshold_mult", 1.0)

        avg_volume = volumes.rolling(window=20).mean().loc[timestamp]
        avg_price = prices.rolling(window=20).mean().loc[timestamp]
        dollar_volume = avg_volume * avg_price
        liquid_symbols = dollar_volume[
            (avg_volume >= self.min_avg_volume)
            & (dollar_volume >= self.min_dollar_volume)
        ].index

        if liquid_symbols.empty:
            return {}

        df_short = self._calculate_reversal_scores(
            prices,
            volumes,
            timestamp,
            short_lookback,
            liquid_symbols,
            threshold_multiplier,
        )
        df_long = self._calculate_reversal_scores(
            prices,
            volumes,
            timestamp,
            long_lookback,
            liquid_symbols,
            threshold_multiplier,
        )

        all_signal_symbols = df_short.index.union(df_long.index)
        if all_signal_symbols.empty:
            return {}

        df = pd.DataFrame(index=all_signal_symbols)
        df["short_score"] = (
            df_short["score"].reindex(all_signal_symbols, fill_value=0.0)
            if not df_short.empty
            else 0.0
        )
        df["long_score"] = (
            df_long["score"].reindex(all_signal_symbols, fill_value=0.0)
            if not df_long.empty
            else 0.0
        )

        lookback_weights = self.regime_lookback_weights.get(
            market_regime.upper(), {"short": 0.5, "long": 0.5}
        )
        w_short = lookback_weights["short"]
        w_long = lookback_weights["long"]

        df["reversal_score"] = (df["short_score"] * w_short) + (
            df["long_score"] * w_long
        )

        short_quality = (
            df_short["quality_score"].reindex(all_signal_symbols, fill_value=0.0)
            if not df_short.empty
            else 0.0
        )
        long_quality = (
            df_long["quality_score"].reindex(all_signal_symbols, fill_value=0.0)
            if not df_long.empty
            else 0.0
        )
        df["base_quality"] = np.where(
            df["short_score"] > 0, short_quality, long_quality
        )

        df["final_score"] = np.clip(df["reversal_score"] * regime_strength, 0, 1.0)

        base_confidence = df["reversal_score"].clip(0.35, 0.85)
        quality_bonus = (df["base_quality"] - 1.0).clip(-0.1, 0.1)
        df["confidence"] = (base_confidence + quality_bonus).clip(0.30, 0.90)

        final_df = df[
            (df["confidence"] >= self.min_confidence) & (df["final_score"] >= 0.15)
        ]
        if final_df.empty:
            return {}

        return {
            symbol: RawAlphaSignal(
                symbol=symbol,
                score=float(row.final_score),
                confidence=float(row.confidence),
                expected_value=0.0,
                components={
                    "reversal_score": float(row.reversal_score),
                    "reversal_short_score": float(row.short_score),
                    "reversal_long_score": float(row.long_score),
                    "reversal_dyn_short_lookback": short_lookback,
                    "reversal_dyn_long_lookback": long_lookback,
                    "regime_w_short": w_short,
                    "regime_w_long": w_long,
                },
            )
            for symbol, row in final_df.iterrows()
        }
