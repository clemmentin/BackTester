import logging
from typing import Dict
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal


class ReversalAlphaModule:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        params = kwargs.get("reversal_alpha_params", {})

        self.reversal_lookback = params.get("reversal_lookback", 21)
        self.extreme_loser_threshold = params.get("extreme_loser_threshold", -0.12)

        self.score_weights = {
            "strength": 0.70,
            "quality": 0.30,
        }

        self.volume_surge_threshold = params.get("volume_surge_threshold", 1.15)
        self.min_confidence = params.get("min_confidence", 0.30)

        self.min_avg_volume = params.get("min_avg_volume", 300_000)
        self.min_dollar_volume = params.get("min_dollar_volume", 15_000_000)

        self.regime_config = params.get(
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
        )
        self.logger.info(
            "ReversalAlpha [V6-Fixed]: Two-factor model, relaxed liquidity"
        )

    def calculate_batch_reversal_trend_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, RawAlphaSignal]:
        if (
            prices.empty
            or timestamp not in prices.index
            or prices.index.get_loc(timestamp) < self.reversal_lookback + 5
        ):
            return {}

        regime_params = self.regime_config.get(market_regime.upper(), {})
        regime_strength = regime_params.get("strength", 1.0)
        threshold_multiplier = regime_params.get("threshold_mult", 1.0)

        # Liquidity filter - relaxed
        avg_volume = volumes.rolling(window=20).mean().loc[timestamp]
        avg_price = prices.rolling(window=20).mean().loc[timestamp]
        dollar_volume = avg_volume * avg_price
        liquid_symbols = dollar_volume[
            (avg_volume >= self.min_avg_volume)
            & (dollar_volume >= self.min_dollar_volume)
        ].index

        if liquid_symbols.empty:
            return {}

        # Calculate returns
        lookback_date = prices.index[
            prices.index.get_loc(timestamp) - self.reversal_lookback
        ]
        start_prices = prices.loc[lookback_date, liquid_symbols]
        end_prices = prices.loc[timestamp, liquid_symbols]

        valid_prices = start_prices.gt(0) & end_prices.notna()
        if not valid_prices.any():
            return {}

        short_return = (end_prices[valid_prices] / start_prices[valid_prices]) - 1.0
        df = pd.DataFrame({"short_return": short_return})

        # Filter losers
        dynamic_loser_threshold = self.extreme_loser_threshold * threshold_multiplier
        df = df[df["short_return"] < dynamic_loser_threshold]

        if df.empty:
            return {}

        df["strength_score"] = np.clip(
            np.power(-df["short_return"] * 2.5, 0.8),  # Sublinear for extreme cases
            0,
            1.0,
        )

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
            .clip(0.6, 1.4)  # More moderate range
            .fillna(0.8)
        )

        # Combined score
        df["reversal_score"] = (
            df["strength_score"] * self.score_weights["strength"]
            + df["quality_score"] * self.score_weights["quality"]
        )

        df["final_score"] = np.clip(df["reversal_score"] * regime_strength, 0, 1.0)

        # Confidence based on score magnitude and quality
        base_confidence = df["reversal_score"].clip(0.35, 0.85)
        quality_bonus = (df["quality_score"] - 1.0).clip(-0.1, 0.1)
        df["confidence"] = (base_confidence + quality_bonus).clip(0.30, 0.90)

        # Filter
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
                components={
                    "reversal_score": float(row.reversal_score),
                    "strength_score": float(row.strength_score),
                    "quality_score": float(row.quality_score),
                    "regime": market_regime,
                },
            )
            for symbol, row in final_df.iterrows()
        }
