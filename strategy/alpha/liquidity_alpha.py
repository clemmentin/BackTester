import logging
from typing import Dict
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal


class LiquidityAlphaModule:
    """
    Generates alpha signals based on liquidity and volume anomalies.

    This refactored version corrects the interpretation of several sub-factors
    based on historical data analysis, focusing on:
    1. Volume Momentum: Rising volume as a sign of growing interest.
    2. OBV (On-Balance Volume): Correctly interpreting its trend and divergence signals.
    3. Illiquidity Premium: Capturing the finding that less liquid stocks (higher Amihud)
       sometimes show better future performance.
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        liquidity_params = kwargs.get("liquidity_alpha_params", {})

        # --- Sub-Factor Calculation Parameters ---
        self.volume_lookback = liquidity_params.get("volume_lookback", 20)
        self.amihud_lookback = liquidity_params.get("amihud_lookback", 20)
        self.obv_signal_period = liquidity_params.get("obv_signal_period", 10)

        # --- Toggles for enabling/disabling sub-factors ---
        self.use_amihud = liquidity_params.get("use_amihud", True)
        self.use_obv = liquidity_params.get("use_obv", True)

        # --- Quality & Filtering Parameters ---
        self.min_confidence = liquidity_params.get("min_confidence", 0.30)
        self.min_avg_dollar_volume = liquidity_params.get(
            "min_avg_dollar_volume", 1_000_000
        )
        self.volume_spike_threshold = liquidity_params.get(
            "volume_spike_threshold", 2.0  # Increased threshold for a clearer signal
        )

        self.logger.info(
            f"LiquidityAlpha [Refactored] initialized: "
            f"Amihud={'ON' if self.use_amihud else 'OFF'}, "
            f"OBV={'ON' if self.use_obv else 'OFF'}, "
            f"min_dollar_vol=${self.min_avg_dollar_volume:,.0f}"
        )

    def calculate_batch_liquidity_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, RawAlphaSignal]:
        """
        Main function to generate liquidity-based alpha signals for all symbols.
        """
        # --- 1. Data Validation and Preparation ---
        if prices.empty or volumes.empty or timestamp not in prices.index:
            self.logger.debug(f"Insufficient data at {timestamp}")
            return {}

        required_len = (
            max(self.volume_lookback, self.amihud_lookback, self.obv_signal_period) + 10
        )
        if len(prices) < required_len:
            self.logger.debug(
                f"Need {required_len} periods, have {len(prices)} at {timestamp}"
            )
            return {}

        end_loc = prices.index.get_loc(timestamp)
        start_loc = max(0, end_loc - required_len)
        price_slice = prices.iloc[start_loc : end_loc + 1]
        volume_slice = volumes.iloc[start_loc : end_loc + 1]

        # --- 2. Pre-filter illiquid stocks to save computation ---
        avg_dollar_volume = (price_slice * volume_slice).mean()
        valid_symbols = avg_dollar_volume[
            avg_dollar_volume >= self.min_avg_dollar_volume
        ].index
        if len(valid_symbols) == 0:
            return {}

        price_slice = price_slice[valid_symbols]
        volume_slice = volume_slice[valid_symbols]

        # --- 3. Calculate All Sub-Factor Components ---
        df = pd.DataFrame(index=price_slice.columns)
        df = self._calculate_volume_metrics(volume_slice, df)
        df = self._calculate_volume_price_synergy(price_slice, volume_slice, df)
        if self.use_obv:
            df = self._calculate_obv_signals(price_slice, volume_slice, df)
        if self.use_amihud:
            df = self._calculate_amihud_liquidity(price_slice, volume_slice, df)

        # --- 4. Generate Composite Score and Confidence ---
        df = self._generate_composite_score(df)
        df = self._calculate_confidence(df)

        # --- 5. Final Filtering and Signal Generation ---
        final_df = df[
            (df["confidence"] >= self.min_confidence) & (abs(df["score"]) >= 0.15)
        ].copy()

        if final_df.empty:
            return {}

        # Generate signal objects for stocks that passed the filter
        return {
            symbol: RawAlphaSignal(
                symbol=symbol,
                score=float(np.clip(row["score"], -1.0, 1.0)),
                confidence=float(np.clip(row["confidence"], 0.30, 0.90)),
                components={
                    "volume_momentum": float(row.get("volume_momentum", 0)),
                    "up_down_volume_ratio": float(row.get("up_down_volume_ratio", 0)),
                    "obv_trend": float(row.get("obv_trend", 0)),
                    "obv_price_divergence": float(row.get("obv_price_divergence", 0)),
                    "liquidity_percentile": float(row.get("liquidity_percentile", 0.5)),
                    "composite_score": float(row.get("composite_score", 0)),
                },
            )
            for symbol, row in final_df.iterrows()
        }

    def _calculate_volume_metrics(
        self, volume_slice: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates metrics related to volume changes and momentum."""
        # Volume ratio vs recent average (for spike detection)
        avg_volume = (
            volume_slice.iloc[:-1].rolling(window=self.volume_lookback).mean().iloc[-1]
        )
        df["volume_ratio"] = volume_slice.iloc[-1] / (avg_volume + 1)

        # Volume momentum: trend of recent volume vs longer-term volume
        recent_avg_vol = volume_slice.iloc[-5:].mean()
        past_avg_vol = volume_slice.iloc[-self.volume_lookback : -5].mean()
        df["volume_momentum"] = (recent_avg_vol / (past_avg_vol + 1)) - 1.0
        return df

    def _calculate_volume_price_synergy(
        self, price_slice: pd.DataFrame, volume_slice: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates if more volume occurs on up days vs down days."""
        price_changes = price_slice.pct_change().iloc[-self.volume_lookback :]
        volumes_in_period = volume_slice.iloc[-self.volume_lookback :]

        up_mask = price_changes > 0
        down_mask = price_changes < 0

        avg_up_volume = (volumes_in_period * up_mask).sum() / (up_mask.sum() + 1)
        avg_down_volume = (volumes_in_period * down_mask).sum() / (down_mask.sum() + 1)

        df["up_down_volume_ratio"] = avg_up_volume / (avg_down_volume + 1)
        return df

    def _calculate_obv_signals(
        self, price_slice: pd.DataFrame, volume_slice: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates On-Balance-Volume (OBV) trend and divergence."""
        price_direction = np.sign(price_slice.diff().fillna(0))
        obv = (price_direction * volume_slice).cumsum()

        # OBV trend: a measure of recent OBV change
        obv_change = obv.iloc[-1] / (obv.iloc[-self.obv_signal_period] + 1e-9) - 1.0
        df["obv_trend"] = obv_change

        # OBV-Price divergence: does OBV confirm the recent price trend?
        price_change = (
            price_slice.iloc[-1] / price_slice.iloc[-self.obv_signal_period] - 1.0
        )
        df["obv_price_divergence"] = obv_change - price_change
        return df

    def _calculate_amihud_liquidity(
        self, price_slice: pd.DataFrame, volume_slice: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates Amihud illiquidity measure."""
        returns = price_slice.pct_change().abs()
        dollar_volumes = price_slice * volume_slice

        # Amihud = average(|return| / $volume) over a period. Higher means MORE illiquid.
        amihud_illiquidity = (
            (returns / (dollar_volumes + 1e-9))
            .rolling(window=self.amihud_lookback)
            .mean()
            .iloc[-1]
        )

        # We rank illiquidity. A higher percentile means the stock is MORE illiquid than its peers.
        df["liquidity_percentile"] = amihud_illiquidity.rank(pct=True)
        return df

    def _generate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines all sub-factors into a final score, respecting the findings from data analysis.
        """
        # --- 1. Isolate the only valuable signal ---

        liquidity_quality_score = (df.get("liquidity_percentile", 0.5) - 0.5) * 2.0

        # --- 2. Define weights to reflect this finding ---
        weights = {
            "volume_momentum": 0.025,
            "pv_synergy": 0.0,
            "obv_trend": 0.025,
            "obv_divergence": 0.05,
            "liquidity_quality": 0.90,
        }

        # --- 3. Calculate the weighted average ---

        volume_momentum_score = np.tanh(df.get("volume_momentum", 0.0))
        pv_synergy_score = np.tanh((df.get("up_down_volume_ratio", 1.0) - 1.0) * 2.0)
        obv_trend_score = np.tanh(df.get("obv_trend", 0.0))
        obv_divergence_score = np.tanh(df.get("obv_price_divergence", 0.0))

        df["composite_score"] = (
            liquidity_quality_score * weights["liquidity_quality"]
            + volume_momentum_score * weights["volume_momentum"]
            + pv_synergy_score * weights["pv_synergy"]
            + obv_trend_score * weights["obv_trend"]
            + obv_divergence_score * weights["obv_divergence"]
        )

        # The composite_score is now our final score, clipped to the standard [-1, 1] range.
        df["score"] = df["composite_score"].clip(-1.0, 1.0)
        return df

    def _calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates signal confidence based on agreement and strength of components.
        """
        # Base confidence on the magnitude of the final score
        base_confidence = 0.35 + abs(df["score"]) * 0.30

        # Add a bonus for a significant volume spike, as it indicates high interest
        volume_spike_bonus = (df["volume_ratio"] > self.volume_spike_threshold) * 0.15

        # Add a bonus for very high illiquidity, as this is a core part of the refactored signal
        liquidity_bonus = (df.get("liquidity_percentile", 0.5) > 0.85) * 0.10

        # Combine and clip to the final confidence range
        df["confidence"] = (
            base_confidence + volume_spike_bonus + liquidity_bonus
        ).clip(0.30, 0.90)

        return df
