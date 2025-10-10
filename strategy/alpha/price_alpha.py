import logging
from typing import Dict
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal


class PriceAlphaModule:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        price_params = kwargs.get("price_alpha_params", {})

        # Pillar 1: Mean Reversion Params
        self.rsi_period = price_params.get("rsi_period", 14)
        self.rsi_oversold = price_params.get("rsi_oversold", 30)
        self.rsi_overbought = price_params.get("rsi_overbought", 70)
        self.bb_period = price_params.get("bb_period", 20)
        self.bb_std = price_params.get("bb_std", 2.0)

        # Pillar 2: Short-Term Reversal Params
        self.reversal_sensitivity = price_params.get("reversal_sensitivity", 15)

        # Pillar 3: Volatility Quality Params
        self.volatility_period = price_params.get("volatility_period", 30)
        self.low_vol_percentile = price_params.get(
            "low_vol_percentile", 0.20
        )  # Top 20% least volatile stocks

        # General Params
        self.min_confidence = price_params.get("min_confidence", 0.30)
        self.min_score_threshold = price_params.get("min_score_threshold", 0.15)

        self.logger.info(
            "PriceAlpha [RECONSTRUCTED] initialized with three-pillar model."
        )

    def calculate_batch_price_signals(
        self,
        prices: pd.DataFrame,
        opens: pd.DataFrame,
        # volumes is no longer needed, keeping signature for compatibility
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, RawAlphaSignal]:
        """
        Main function to generate signals by combining the three pillars.
        """
        if prices.empty or opens.empty or timestamp not in prices.index:
            return {}

        required_len = max(self.rsi_period, self.bb_period, self.volatility_period) + 5
        if len(prices) < required_len:
            return {}

        df = pd.DataFrame(index=prices.columns)

        # --- Calculate scores for each of the three pillars ---
        df = self._calculate_mean_reversion_score(prices, df)
        df = self._calculate_short_term_reversal_score(prices, opens, df)
        df = self._calculate_volatility_quality_score(prices, df)

        # --- Combine pillar scores into a final composite score ---
        df = self._generate_composite_score(df)

        # --- Calculate final confidence ---
        df = self._calculate_confidence(df)

        # --- Final Filtering ---
        final_df = df[
            (df["confidence"] >= self.min_confidence)
            & (abs(df["score"]) >= self.min_score_threshold)
        ].copy()

        if final_df.empty:
            return {}

        # --- Generate Signal Objects ---
        return {
            symbol: RawAlphaSignal(
                symbol=symbol,
                score=float(np.clip(row["score"], -1.0, 1.0)),
                confidence=float(np.clip(row["confidence"], 0.30, 0.95)),
                components={
                    "mean_reversion_score": float(row.get("mean_reversion_score", 0)),
                    "short_term_reversal_score": float(
                        row.get("short_term_reversal_score", 0)
                    ),
                    "volatility_quality_score": float(
                        row.get("volatility_quality_score", 0)
                    ),
                },
            )
            for symbol, row in final_df.iterrows()
        }

    def _calculate_mean_reversion_score(
        self, prices: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Pillar 1: Calculates mean reversion signals from RSI and Bollinger Bands."""
        # RSI Score
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0.0).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # A score of +1 for deeply oversold, -1 for deeply overbought
        rsi_score = np.select(
            [rsi < self.rsi_oversold, rsi > self.rsi_overbought],
            [
                (self.rsi_oversold - rsi) / self.rsi_oversold,  # e.g. (30-10)/30 = 0.66
                (self.rsi_overbought - rsi)
                / (100 - self.rsi_overbought),  # e.g. (70-90)/30 = -0.66
            ],
            default=0.0,
        )

        # Bollinger Band Score
        sma = prices.rolling(window=self.bb_period).mean().iloc[-1]
        std = prices.rolling(window=self.bb_period).std().iloc[-1]
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std

        # A score of +1 for touching the lower band, -1 for the upper band
        bb_position = (prices.iloc[-1] - lower) / (upper - lower + 1e-9)
        bb_score = 1.0 - 2.0 * bb_position  # Scale to [-1, 1]

        # Combine with higher weight on the more sensitive indicator (RSI)
        df["mean_reversion_score"] = (rsi_score * 0.6 + bb_score * 0.4).clip(-1.0, 1.0)
        df["raw_rsi"] = rsi  # Store for confidence calculation
        return df

    def _calculate_short_term_reversal_score(
        self, prices: pd.DataFrame, opens: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pillar 2: RECONFIGURED. This pillar is a source of noise and conflict.
        It will be DISABLED by giving it a zero weight in the composite score.
        We return a zero score to avoid breaking the data flow.
        """
        df["short_term_reversal_score"] = 0.0
        return df

    def _calculate_volatility_quality_score(
        self, prices: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pillar 3: RECONFIGURED. Low volatility is a standalone factor, not a price pattern.
        It will be DISABLED by giving it a zero weight to maintain the purity of the
        mean-reversion signal.
        """
        df["volatility_quality_score"] = 0.0
        df["vol_percentile"] = 0.5  # Neutral value
        return df

    def _generate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines the three pillar scores into a final score.
        RECONSTRUCTED: Focus 100% on the pure Mean Reversion signal.
        """
        # The weights now explicitly disable the other two pillars.
        weights = {
            "mean_reversion": 1.0,
            "short_term_reversal": 0.0,
            "volatility_quality": 0.0,
        }

        # The final score IS the mean reversion score.
        df["score"] = (
            df["mean_reversion_score"] * weights["mean_reversion"]
            + df["short_term_reversal_score"] * weights["short_term_reversal"]
            + df["volatility_quality_score"] * weights["volatility_quality"]
        )
        df["score"] = df["score"].clip(-1.0, 1.0)
        return df

    def _calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates signal confidence based on the strength of the mean reversion signal.
        """
        # Base confidence from the magnitude of the final (mean reversion) score
        base_confidence = 0.40 + abs(df["score"]) * 0.50

        # Add a significant bonus if the stock is in a strong mean-reversion state
        # (deeply overbought/oversold RSI).
        rsi_extreme_bonus = (
            ((df["raw_rsi"] < 20) | (df["raw_rsi"] > 80)) * 0.20
        ).fillna(0)

        df["confidence"] = (base_confidence + rsi_extreme_bonus).clip(0.30, 0.95)
        return df
