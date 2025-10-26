import logging
from typing import Dict
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal


class PriceAlphaModule:
    """ """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        price_params = kwargs.get("price_alpha_params", {})

        # Basic mean reversion params - try flat first, then nested
        self.rsi_period = kwargs.get("rsi_period", price_params.get("rsi_period", 14))
        self.rsi_oversold = kwargs.get(
            "rsi_oversold", price_params.get("rsi_oversold", 30)
        )
        self.rsi_overbought = kwargs.get(
            "rsi_overbought", price_params.get("rsi_overbought", 70)
        )
        self.bb_period = kwargs.get("bb_period", price_params.get("bb_period", 20))
        self.bb_std = kwargs.get("bb_std", price_params.get("bb_std", 2.0))

        # High certainty mode params
        self.enable_high_certainty_mode = kwargs.get(
            "enable_high_certainty_mode",
            price_params.get("enable_high_certainty_mode", True),
        )
        # Load more parameters for high certainty mode
        self.rsi_short_period = kwargs.get(
            "rsi_short_period", price_params.get("rsi_short_period", 5)
        )
        self.rsi_long_period = kwargs.get(
            "rsi_long_period", price_params.get("rsi_long_period", 28)
        )
        self.rsi_short_oversold = kwargs.get(
            "rsi_short_oversold", price_params.get("rsi_short_oversold", 35)
        )
        self.rsi_long_oversold = kwargs.get(
            "rsi_long_oversold", price_params.get("rsi_long_oversold", 45)
        )
        self.rsi_short_overbought = kwargs.get(
            "rsi_short_overbought", price_params.get("rsi_short_overbought", 65)
        )
        self.rsi_long_overbought = kwargs.get(
            "rsi_long_overbought", price_params.get("rsi_long_overbought", 55)
        )
        self.bb_deviation_threshold = kwargs.get(
            "bb_deviation_threshold", price_params.get("bb_deviation_threshold", 1.0)
        )

        self.base_confidence_on_signal = kwargs.get(
            "base_confidence_on_signal",
            price_params.get("base_confidence_on_signal", 0.65),
        )
        self.volume_confirmation_multiplier = kwargs.get(
            "volume_confirmation_multiplier",
            price_params.get("volume_confirmation_multiplier", 1.5),
        )
        self.confidence_volume_bonus = kwargs.get(
            "confidence_volume_bonus", price_params.get("confidence_volume_bonus", 0.15)
        )

        # General params
        self.min_confidence = kwargs.get(
            "min_confidence", price_params.get("min_confidence", 0.35)
        )
        self.min_score_threshold = kwargs.get(
            "min_score_threshold", price_params.get("min_score_threshold", 0.40)
        )

        # (MODIFIED) Load calculation constants
        self.alignment_short_weight = kwargs.get(
            "alignment_short_weight", price_params.get("alignment_short_weight", 0.5)
        )
        self.alignment_long_weight = kwargs.get(
            "alignment_long_weight", price_params.get("alignment_long_weight", 0.3)
        )
        self.quality_denominator = kwargs.get(
            "quality_denominator", price_params.get("quality_denominator", 50.0)
        )
        self.bb_penetration_mult = kwargs.get(
            "bb_penetration_mult", price_params.get("bb_penetration_mult", 0.15)
        )
        self.base_oversold_score = kwargs.get(
            "base_oversold_score", price_params.get("base_oversold_score", 0.5)
        )
        self.base_overbought_score = kwargs.get(
            "base_overbought_score", price_params.get("base_overbought_score", -0.5)
        )
        self.quality_mult = kwargs.get(
            "quality_mult", price_params.get("quality_mult", 0.3)
        )

        self.logger.info(
            "PriceAlpha [REFACTORED] initialized with high certainty mean reversion."
        )

    def calculate_batch_price_signals(
        self,
        prices: pd.DataFrame,
        opens: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, RawAlphaSignal]:
        """

        Args:
          prices: pd.DataFrame: 
          opens: pd.DataFrame: 
          volumes: pd.DataFrame: 
          timestamp: pd.Timestamp: 
          market_regime: str:  (Default value = "NORMAL")

        Returns:

        """
        if market_regime.upper() in ["CRISIS", "VOLATILE"]:
            self.logger.info(
                f"PriceAlphaModule disabled in {market_regime.upper()} regime."
            )
            return {}

        if prices.empty or timestamp not in prices.index:
            return {}

        required_len = max(self.rsi_period, self.bb_period, self.rsi_long_period) + 5
        if len(prices) < required_len:
            return {}

        # (MODIFIED) Increase sensitivity in favorable regimes by adjusting thresholds.
        if market_regime.upper() in ["BEAR", "NORMAL"]:
            self.logger.debug(
                f"Increasing PriceAlphaModule sensitivity for {market_regime.upper()} regime."
            )
            current_min_score_threshold = self.min_score_threshold * 0.85
            current_min_confidence = self.min_confidence * 0.90
        else:
            current_min_score_threshold = self.min_score_threshold
            current_min_confidence = self.min_confidence

        df = pd.DataFrame(index=prices.columns)
        df = self._calculate_mean_reversion_score(prices, volumes, df)
        df = self._generate_composite_score(df)
        df = self._calculate_confidence(df, volumes)

        final_df = df[
            (df["confidence"] >= current_min_confidence)
            & (abs(df["score"]) >= current_min_score_threshold)
        ].copy()

        if final_df.empty:
            return {}

        return {
            symbol: RawAlphaSignal(
                symbol=symbol,
                score=float(np.clip(row["score"], -1.0, 1.0)),
                confidence=float(np.clip(row["confidence"], 0.30, 0.95)),
                expected_value=0.0,
                components={
                    "price_mean_reversion_score": float(
                        row.get("mean_reversion_score", 0)
                    ),
                },
            )
            for symbol, row in final_df.iterrows()
        }

    def _calculate_mean_reversion_score(
        self, prices: pd.DataFrame, volumes: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """

        Args:
          prices: pd.DataFrame: 
          volumes: pd.DataFrame: 
          df: pd.DataFrame: 

        Returns:

        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        gain_14 = gain.rolling(window=self.rsi_period).mean()
        loss_14 = loss.rolling(window=self.rsi_period).mean()
        rs_14 = gain_14 / (loss_14 + 1e-9)
        rsi_14 = (100 - (100 / (1 + rs_14))).iloc[-1]

        gain_5 = gain.rolling(window=self.rsi_short_period).mean()
        loss_5 = loss.rolling(window=self.rsi_short_period).mean()
        rs_5 = gain_5 / (loss_5 + 1e-9)
        rsi_5 = (100 - (100 / (1 + rs_5))).iloc[-1]

        gain_28 = gain.rolling(window=self.rsi_long_period).mean()
        loss_28 = loss.rolling(window=self.rsi_long_period).mean()
        rs_28 = gain_28 / (loss_28 + 1e-9)
        rsi_28 = (100 - (100 / (1 + rs_28))).iloc[-1]

        sma = prices.rolling(window=self.bb_period).mean().iloc[-1]
        std = prices.rolling(window=self.bb_period).std().iloc[-1]
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        current_price = prices.iloc[-1]

        price_deviation = (current_price - sma) / (std + 1e-9)
        volatility_percentile = std / sma
        volatility_adj = (1.0 / (1.0 + volatility_percentile * 5)).clip(0.5, 1.0)
        score = pd.Series(0.0, index=prices.columns)

        if self.enable_high_certainty_mode:
            # (MODIFIED) Use parameters for thresholds
            oversold_mask = rsi_14 < self.rsi_oversold
            oversold_confirmed = (
                oversold_mask
                & (rsi_5 < self.rsi_short_oversold)
                & (rsi_28 < self.rsi_long_oversold)
            )
            oversold_confirmed &= price_deviation < -self.bb_deviation_threshold

            overbought_mask = rsi_14 > self.rsi_overbought
            overbought_confirmed = (
                overbought_mask
                & (rsi_5 > self.rsi_short_overbought)
                & (rsi_28 > self.rsi_long_overbought)
            )
            overbought_confirmed &= price_deviation > self.bb_deviation_threshold

            # (MODIFIED) Use parameters for quality calculation
            oversold_quality = np.where(
                oversold_confirmed,
                (
                    (self.rsi_oversold - rsi_14)
                    + (self.rsi_short_oversold - rsi_5) * self.alignment_short_weight
                    + (self.rsi_long_oversold - rsi_28) * self.alignment_long_weight
                )
                / self.quality_denominator,
                0.0,
            )
            overbought_quality = np.where(
                overbought_confirmed,
                (
                    (rsi_14 - self.rsi_overbought)
                    + (rsi_5 - self.rsi_short_overbought) * self.alignment_short_weight
                    + (rsi_28 - self.rsi_long_overbought) * self.alignment_long_weight
                )
                / self.quality_denominator,
                0.0,
            )

            bb_penetration_oversold = ((lower - current_price) / (std + 1e-9)).clip(
                0, 2
            )
            bb_penetration_overbought = ((current_price - upper) / (std + 1e-9)).clip(
                0, 2
            )

            # (MODIFIED) Use parameters for scoring
            oversold_base_score = (
                self.base_oversold_score
                + oversold_quality * self.quality_mult
                + bb_penetration_oversold * self.bb_penetration_mult
            )
            overbought_base_score = (
                self.base_overbought_score
                - overbought_quality * self.quality_mult
                - bb_penetration_overbought * self.bb_penetration_mult
            )

            score = np.where(
                oversold_confirmed, oversold_base_score * volatility_adj, score
            )
            score = np.where(
                overbought_confirmed, overbought_base_score * volatility_adj, score
            )

        else:
            rsi_score = np.select(
                [rsi_14 < self.rsi_oversold, rsi_14 > self.rsi_overbought],
                [
                    (self.rsi_oversold - rsi_14) / self.rsi_oversold,
                    (self.rsi_overbought - rsi_14) / (100 - self.rsi_overbought),
                ],
                default=0.0,
            )
            bb_position = (current_price - lower) / (upper - lower + 1e-9)
            bb_score = 1.0 - 2.0 * bb_position
            score = rsi_score * 0.6 + bb_score * 0.4

        df["mean_reversion_score"] = pd.Series(score, index=prices.columns).clip(
            -1.0, 1.0
        )
        df["raw_rsi"] = rsi_14
        if not volumes.empty and len(volumes) >= 20:
            avg_volume = volumes.iloc[-20:].mean()
            current_volume = volumes.iloc[-1]
            df["volume_ratio"] = (current_volume / (avg_volume + 1e-9)).clip(0, 10)
        else:
            df["volume_ratio"] = 1.0

        return df

    def _generate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
          df: pd.DataFrame: 

        Returns:

        """
        df["score"] = df["mean_reversion_score"].clip(-1.0, 1.0)
        return df

    def _calculate_confidence(
        self, df: pd.DataFrame, volumes: pd.DataFrame
    ) -> pd.DataFrame:
        """

        Args:
          df: pd.DataFrame: 
          volumes: pd.DataFrame: 

        Returns:

        """
        if self.enable_high_certainty_mode:
            base_confidence = np.where(
                df["score"] != 0,
                self.base_confidence_on_signal,
                0.30,
            )
            volume_bonus = np.where(
                (df["score"] != 0)
                & (df["volume_ratio"] >= self.volume_confirmation_multiplier),
                self.confidence_volume_bonus,
                0.0,
            )
            df["confidence"] = (base_confidence + volume_bonus).clip(0.30, 0.95)
        else:
            base_confidence = 0.40 + abs(df["score"]) * 0.50
            rsi_extreme_bonus = (
                ((df["raw_rsi"] < 20) | (df["raw_rsi"] > 80)) * 0.20
            ).fillna(0)
            df["confidence"] = (base_confidence + rsi_extreme_bonus).clip(0.30, 0.95)

        return df
