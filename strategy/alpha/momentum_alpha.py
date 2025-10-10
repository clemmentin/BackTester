import logging
from typing import Dict
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal


class MomentumAlphaModule:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        momentum_params = kwargs.get("momentum_alpha_params", {})

        # --- Core Momentum Parameters ---
        self.formation_period = momentum_params.get("formation_period", 84)
        self.skip_period = momentum_params.get("skip_period", 14)
        self.min_absolute_momentum = momentum_params.get("min_absolute_momentum", 0.02)
        self.min_confidence = momentum_params.get("min_confidence", 0.25)

        # --- Volatility Control ---
        self.use_volatility_control = momentum_params.get(
            "use_volatility_control", True
        )
        self.max_volatility_percentile = momentum_params.get(
            "max_volatility_percentile", 0.90
        )

        # --- Quality Filter ---
        self.use_quality_filter = momentum_params.get("use_quality_filter", True)
        self.high_dd_penalty = momentum_params.get(
            "high_dd_penalty", 0.20
        )  # PENALTY not bonus
        self.max_dd_threshold = momentum_params.get(
            "max_dd_threshold", 0.15
        )  # Penalize DD > 15%

        self.logger.info(
            f"MomentumAlpha [V7-Corrected]: Raw momentum with quality penalty for high drawdown"
        )

    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown during the formation period."""
        running_max = price_series.expanding(min_periods=1).max()
        drawdown = (running_max - price_series) / running_max
        return drawdown.max()

    def calculate_batch_momentum_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_regime: str = "NORMAL",
    ) -> Dict[str, RawAlphaSignal]:

        # --- 1. Basic Validation ---
        if prices.empty or timestamp not in prices.index:
            return {}

        required_len = self.formation_period + self.skip_period
        if len(prices) < required_len:
            return {}

        end_date_loc = prices.index.get_loc(timestamp)
        formation_end_date = prices.index[end_date_loc - self.skip_period]
        formation_start_date = prices.index[end_date_loc - required_len]

        # --- 2. Calculate RAW MOMENTUM (IC=0.0622) ---
        price_slice = prices.loc[formation_start_date:formation_end_date]
        start_prices = price_slice.iloc[0]
        end_prices = price_slice.iloc[-1]

        raw_momentum = (end_prices / start_prices) - 1.0
        raw_momentum = raw_momentum.dropna()

        if raw_momentum.empty:
            return {}

        # Filter minimum momentum
        raw_momentum = raw_momentum[raw_momentum > self.min_absolute_momentum]
        if raw_momentum.empty:
            return {}

        df = pd.DataFrame(raw_momentum, columns=["raw_momentum"])

        # --- 3. Calculate Returns for Volatility ---
        log_returns = np.log(price_slice / price_slice.shift(1)).dropna()

        # --- 4. Volatility Filter ---
        if self.use_volatility_control:
            volatility = log_returns[df.index].std() * np.sqrt(252)
            vol_threshold = volatility.quantile(self.max_volatility_percentile)
            df = df[volatility[df.index] <= vol_threshold]

            if df.empty:
                return {}

        # --- 5. Calculate Quality Indicators ---
        if self.use_quality_filter:
            # Max Drawdown: IC=-0.1069, so HIGH DD = WEAK quality = PENALTY
            df["max_drawdown"] = price_slice[df.index].apply(
                self._calculate_max_drawdown
            )

            # Positive days ratio
            df["pos_days_ratio"] = (log_returns[df.index] > 0).sum() / log_returns[
                df.index
            ].count()

            # --- 6. Calculate Scores (REWRITTEN BASED ON DIAGNOSTIC DATA) ---
            df["base_score"] = (df["raw_momentum"].rank(pct=True) - 0.5) * 2.0

            if self.use_quality_filter:
                drawdown_percentile = df["max_drawdown"].rank(pct=True)
                drawdown_score = (drawdown_percentile - 0.5) * 2.0

                pos_days_percentile = df["pos_days_ratio"].rank(pct=True)
                smoothness_score = (
                    0.5 - pos_days_percentile
                ) * 2.0  # Scaled to [-1, 1]

                # --- 7. Combine Scores with Data-Driven Weights ---
                base_weight = 0.40  # Raw momentum IC: ~0.05
                drawdown_weight = 0.50  # Max Drawdown IC: ~0.10 (Strongest Signal!)
                smoothness_weight = (
                    0.10  # Positive Days Ratio IC: ~ -0.02 (Weakest Signal)
                )

                df["score"] = (
                    df["base_score"] * base_weight
                    + drawdown_score * drawdown_weight
                    + smoothness_score * smoothness_weight
                )
            else:
                df["score"] = df["base_score"]

            # --- 8. Final Clipping ---
            df["score"] = df["score"].clip(-1.0, 1.0)

        df["final_percentile"] = df["score"].rank(method="average", pct=True)

        # --- 9. Confidence Based on Signal Strength ---
        dist_from_median = (df["final_percentile"] - 0.5).abs()

        # Higher confidence for more extreme positions
        conds = [dist_from_median > 0.35, dist_from_median > 0.25]
        choices = [0.75, 0.60]
        df["confidence"] = np.select(conds, choices, default=0.40)

        # --- 10. Final Filtering ---
        final_df = df[df["confidence"] >= self.min_confidence].copy()
        if final_df.empty:
            return {}

        # --- 11. Generate Signals ---
        return {
            symbol: RawAlphaSignal(
                symbol=symbol,
                score=float(np.clip(row.score, -1.0, 1.0)),
                confidence=float(np.clip(row.confidence, 0.3, 0.95)),
                components={
                    "raw_momentum": float(row.raw_momentum),
                    "final_percentile": float(row.final_percentile),
                    "max_drawdown": float(row.get("max_drawdown", 0)),
                    "pos_days_ratio": float(row.get("pos_days_ratio", 0)),
                    "quality_penalty_applied": bool(
                        self.use_quality_filter
                        and row.get("max_drawdown", 0) > self.max_dd_threshold
                    ),
                },
            )
            for symbol, row in final_df.iterrows()
        }
