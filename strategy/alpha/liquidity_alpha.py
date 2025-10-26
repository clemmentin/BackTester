import logging
from typing import Dict
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal


class LiquidityAlphaModule:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        liquidity_params = kwargs.get("liquidity_alpha_params", {})

        # Try flat parameters first (from optimizer), then nested dict
        self.volume_lookback = kwargs.get(
            "volume_lookback", liquidity_params.get("volume_lookback", 20)
        )
        self.amihud_lookback = kwargs.get(
            "amihud_lookback", liquidity_params.get("amihud_lookback", 20)
        )
        self.min_confidence = kwargs.get(
            "min_confidence", liquidity_params.get("min_confidence", 0.30)
        )
        self.min_avg_dollar_volume = kwargs.get(
            "min_avg_dollar_volume",
            liquidity_params.get("min_avg_dollar_volume", 1_000_000),
        )

        self.regime_multipliers = kwargs.get(
            "regime_multipliers",
            liquidity_params.get(
                "regime_multipliers",
                {
                    "crisis": 1.5,
                    "bear": 1.1,
                    "volatile": 1.1,
                    "normal": 1.0,
                    "bull": 0.95,
                    "strong_bull": 0.75,
                },
            ),
        )

        # (MODIFIED) Load scoring and filtering parameters
        self.score_threshold = kwargs.get(
            "score_threshold", liquidity_params.get("score_threshold", 0.12)
        )
        self.volume_ratio_filter_threshold = kwargs.get(
            "volume_ratio_filter_threshold",
            liquidity_params.get("volume_ratio_filter_threshold", 2.5),
        )

        self.high_liquidity_percentile = kwargs.get(
            "high_liquidity_percentile",
            liquidity_params.get("high_liquidity_percentile", 0.90),
        )
        self.good_liquidity_percentile = kwargs.get(
            "good_liquidity_percentile",
            liquidity_params.get("good_liquidity_percentile", 0.75),
        )
        self.poor_liquidity_percentile = kwargs.get(
            "poor_liquidity_percentile",
            liquidity_params.get("poor_liquidity_percentile", 0.25),
        )
        self.low_liquidity_percentile = kwargs.get(
            "low_liquidity_percentile",
            liquidity_params.get("low_liquidity_percentile", 0.10),
        )

        # (MODIFIED) Load confidence calculation parameters
        self.confidence_base = kwargs.get(
            "confidence_base", liquidity_params.get("confidence_base", 0.35)
        )
        self.confidence_extremeness_mult = kwargs.get(
            "confidence_extremeness_mult",
            liquidity_params.get("confidence_extremeness_mult", 0.80),
        )
        self.confidence_extreme_threshold = kwargs.get(
            "confidence_extreme_threshold",
            liquidity_params.get("confidence_extreme_threshold", 0.85),
        )
        self.confidence_low_threshold = kwargs.get(
            "confidence_low_threshold",
            liquidity_params.get("confidence_low_threshold", 0.15),
        )
        self.confidence_extreme_bonus = kwargs.get(
            "confidence_extreme_bonus",
            liquidity_params.get("confidence_extreme_bonus", 0.15),
        )

        self.logger.info(
            f"LiquidityAlpha [Regime-Aware] initialized: "
            f"Using regime multipliers and fixed components."
        )

    def calculate_batch_liquidity_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, RawAlphaSignal]:
        if prices.empty or volumes.empty or timestamp not in prices.index:
            return {}

        required_len = max(self.volume_lookback, self.amihud_lookback) + 10
        if len(prices) < required_len:
            return {}

        end_loc = prices.index.get_loc(timestamp)
        start_loc = max(0, end_loc - required_len)
        price_slice = prices.iloc[start_loc : end_loc + 1]
        volume_slice = volumes.iloc[start_loc : end_loc + 1]

        avg_dollar_volume = (price_slice * volume_slice).mean()
        valid_symbols = avg_dollar_volume[
            avg_dollar_volume >= self.min_avg_dollar_volume
        ].index
        if len(valid_symbols) == 0:
            return {}

        price_slice = price_slice[valid_symbols]
        volume_slice = volume_slice[valid_symbols]

        df = pd.DataFrame(index=price_slice.columns)
        df = self._calculate_volume_price_synergy(price_slice, volume_slice, df)
        df = self._calculate_amihud_liquidity(price_slice, volume_slice, df)
        df = self._generate_composite_score(df, market_regime)
        df = self._calculate_confidence(df)

        # (MODIFIED) Use parameters for filtering
        volume_ratio_filter = (
            df.get("up_down_volume_ratio", 1.0) < self.volume_ratio_filter_threshold
        )
        final_df = df[
            (df["confidence"] >= self.min_confidence)
            & (abs(df["score"]) >= self.score_threshold)
            & volume_ratio_filter
        ].copy()

        if final_df.empty:
            self.logger.info(
                f"No liquidity signals passed the final filter for {timestamp.date()}. "
                f"Initial candidates: {len(df)}"
            )
            return {}

        self.logger.info(
            f"Liquidity signals: {len(final_df)}/{len(df)} passed filters."
        )

        return {
            symbol: RawAlphaSignal(
                symbol=symbol,
                score=float(np.clip(row["score"], -1.0, 1.0)),
                confidence=float(np.clip(row["confidence"], 0.30, 0.90)),
                expected_value=0.0,
                components={
                    "liquidity_percentile": float(row.get("liquidity_percentile", 0.5)),
                    "liquidity_composite_score": float(row.get("composite_score", 0)),
                    "liquidity_volume_ratio_filter": float(
                        row.get("up_down_volume_ratio", 0)
                    ),
                },
            )
            for symbol, row in final_df.iterrows()
        }

    def _calculate_volume_price_synergy(
        self, price_slice: pd.DataFrame, volume_slice: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        price_changes = price_slice.pct_change().iloc[-self.volume_lookback :]
        volumes_in_period = volume_slice.iloc[-self.volume_lookback :]
        up_mask = price_changes > 0
        down_mask = price_changes < 0
        avg_up_volume = (volumes_in_period * up_mask).sum() / (up_mask.sum() + 1)
        avg_down_volume = (volumes_in_period * down_mask).sum() / (down_mask.sum() + 1)
        df["up_down_volume_ratio"] = avg_up_volume / (avg_down_volume + 1)
        return df

    def _calculate_amihud_liquidity(
        self, price_slice: pd.DataFrame, volume_slice: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        returns = price_slice.pct_change().abs()
        dollar_volumes = price_slice * volume_slice
        amihud_illiquidity = (
            (returns / (dollar_volumes + 1e-9))
            .rolling(window=self.amihud_lookback)
            .mean()
            .iloc[-1]
        )
        df["liquidity_percentile"] = amihud_illiquidity.rank(pct=True)
        return df

    def _generate_composite_score(
        self, df: pd.DataFrame, market_regime: str
    ) -> pd.DataFrame:
        if "liquidity_percentile" not in df.columns:
            df["score"] = 0.0
            df["composite_score"] = 0.0
            return df

        liquidity_pct = df["liquidity_percentile"]

        # (MODIFIED) Scoring logic using parameters from config
        liquidity_quality_score = np.select(
            [
                liquidity_pct > self.high_liquidity_percentile,
                liquidity_pct > self.good_liquidity_percentile,
                liquidity_pct < self.poor_liquidity_percentile,
                liquidity_pct < self.low_liquidity_percentile,
            ],
            [
                0.60 + (liquidity_pct - self.high_liquidity_percentile) * 4.0,
                0.35 + (liquidity_pct - self.good_liquidity_percentile) * 1.67,
                -0.35 + (liquidity_pct - self.poor_liquidity_percentile) * 1.67,
                -0.60 + (liquidity_pct - self.low_liquidity_percentile) * 4.0,
            ],
            default=0.0,
        )

        df["composite_score"] = np.clip(liquidity_quality_score, -1.0, 1.0)
        regime_mult = self.regime_multipliers.get(market_regime.lower(), 1.0)
        df["score"] = df["composite_score"] * regime_mult

        return df

    def _calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        if "liquidity_percentile" not in df.columns:
            df["confidence"] = 0.0
            return df

        liquidity_pct = df["liquidity_percentile"]
        extremeness = abs(liquidity_pct - 0.5)

        # (MODIFIED) Confidence calculation using parameters
        base_confidence = (
            self.confidence_base + extremeness * self.confidence_extremeness_mult
        )
        extreme_bonus = (
            (liquidity_pct > self.confidence_extreme_threshold)
            | (liquidity_pct < self.confidence_low_threshold)
        ) * self.confidence_extreme_bonus
        df["confidence"] = np.clip(base_confidence + extreme_bonus, 0.30, 0.90)
        return df
