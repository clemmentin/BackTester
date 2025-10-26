import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from strategy.contracts import RawAlphaSignal
from strategy.alpha.market_detector import MarketState, MarketRegime


class MomentumAlphaModule:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        momentum_params = kwargs.get("momentum_alpha_params", {})

        # MODIFIED: Removed fixed lookback periods. They are now determined dynamically.
        self.min_absolute_momentum = kwargs.get(
            "min_absolute_momentum", momentum_params.get("min_absolute_momentum", 0.02)
        )
        self.min_confidence = kwargs.get(
            "min_confidence", momentum_params.get("min_confidence", 0.25)
        )

        self.use_volatility_control = kwargs.get(
            "use_volatility_control",
            momentum_params.get("use_volatility_control", True),
        )
        self.max_volatility_percentile = kwargs.get(
            "max_volatility_percentile",
            momentum_params.get("max_volatility_percentile", 0.75),
        )

        self.use_quality_filter = kwargs.get(
            "use_quality_filter", momentum_params.get("use_quality_filter", True)
        )
        self.drawdown_penalty_factor = kwargs.get(
            "drawdown_penalty_factor",
            momentum_params.get("drawdown_penalty_factor", 10.0),
        )

        self.high_confidence_threshold = kwargs.get(
            "high_confidence_threshold",
            momentum_params.get("high_confidence_threshold", 0.35),
        )
        self.medium_confidence_threshold = kwargs.get(
            "medium_confidence_threshold",
            momentum_params.get("medium_confidence_threshold", 0.25),
        )
        self.high_confidence_value = kwargs.get(
            "high_confidence_value", momentum_params.get("high_confidence_value", 0.75)
        )
        self.medium_confidence_value = kwargs.get(
            "medium_confidence_value",
            momentum_params.get("medium_confidence_value", 0.60),
        )
        self.default_confidence_value = kwargs.get(
            "default_confidence_value",
            momentum_params.get("default_confidence_value", 0.40),
        )

        self.logger.info(
            f"MomentumAlpha [V9-AdaptiveLookback]: Using dynamic lookbacks based on MarketState."
        )

    def _get_dynamic_periods(self, market_state: MarketState) -> Tuple[int, int]:
        """NEW: Determines momentum formation and skip periods based on the market state."""
        base_formation, base_skip = 84, 14

        # Adjust based on volatility regime
        if market_state.volatility_regime == "high_vol":
            multiplier = 0.80  # Shorter, more reactive periods in high vol
        elif market_state.volatility_regime == "low_vol":
            multiplier = 1.10  # Longer periods to confirm trends in low vol
        else:
            multiplier = 1.0

        # Further adjust based on market regime
        if market_state.regime in [MarketRegime.CRISIS, MarketRegime.VOLATILE]:
            multiplier *= 0.85
        elif market_state.regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
            multiplier *= 1.15

        formation_period = int(np.clip(base_formation * multiplier, 40, 120))
        # Keep skip period proportional but with a floor
        skip_period = int(np.clip(base_skip * multiplier, 7, 21))

        return formation_period, skip_period

    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        running_max = price_series.expanding(min_periods=1).max()
        drawdown = (running_max - price_series) / running_max
        return drawdown.max()

    def calculate_batch_momentum_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_state: MarketState,  # MODIFIED: Accepts full market state
    ) -> Dict[str, RawAlphaSignal]:

        # NEW: Dynamically determine lookback periods.
        formation_period, skip_period = self._get_dynamic_periods(market_state)

        if prices.empty or timestamp not in prices.index:
            return {}

        required_len = formation_period + skip_period
        if len(prices) < required_len:
            return {}

        end_date_loc = prices.index.get_loc(timestamp)
        if end_date_loc < required_len:
            return {}
        formation_end_date_loc = end_date_loc - skip_period
        formation_start_date_loc = end_date_loc - required_len
        formation_end_date = prices.index[formation_end_date_loc]
        formation_start_date = prices.index[formation_start_date_loc]

        price_slice = prices.loc[formation_start_date:formation_end_date]
        if price_slice.empty:
            return {}

        start_prices = price_slice.iloc[0]
        end_prices = price_slice.iloc[-1]
        raw_momentum = (end_prices / start_prices) - 1.0
        raw_momentum = raw_momentum.dropna()
        raw_momentum = raw_momentum[raw_momentum > self.min_absolute_momentum]

        if raw_momentum.empty:
            return {}

        df = pd.DataFrame(raw_momentum, columns=["raw_momentum"])
        log_returns = np.log(price_slice / price_slice.shift(1)).dropna()

        if self.use_volatility_control:
            volatility = log_returns[df.index].std() * np.sqrt(252)
            vol_threshold = volatility.quantile(self.max_volatility_percentile)
            df = df[volatility[df.index] <= vol_threshold]
            if df.empty:
                return {}

        df["base_score"] = (df["raw_momentum"].rank(pct=True) - 0.5) * 2.0

        if self.use_quality_filter:
            df["max_drawdown"] = price_slice[df.index].apply(
                self._calculate_max_drawdown
            )
            df["drawdown_penalty"] = np.exp(
                -self.drawdown_penalty_factor * df["max_drawdown"]
            )
            df["score"] = df["base_score"] * df["drawdown_penalty"]
        else:
            df["score"] = df["base_score"]

        df["score"] = df["score"].clip(-1.0, 1.0)

        df["final_percentile"] = df["score"].rank(method="average", pct=True)
        dist_from_median = (df["final_percentile"] - 0.5).abs()
        conds = [
            dist_from_median > self.high_confidence_threshold,
            dist_from_median > self.medium_confidence_threshold,
        ]
        choices = [self.high_confidence_value, self.medium_confidence_value]
        df["confidence"] = np.select(
            conds, choices, default=self.default_confidence_value
        )
        final_df = df[df["confidence"] >= self.min_confidence].copy()

        if final_df.empty:
            return {}

        return {
            symbol: RawAlphaSignal(
                symbol=symbol,
                score=float(np.clip(row.score, -1.0, 1.0)),
                confidence=float(np.clip(row.confidence, 0.3, 0.95)),
                expected_value=0.0,
                components={
                    "momentum_raw_momentum": float(row.raw_momentum),
                    "momentum_base_score": float(row.base_score),
                    "momentum_max_drawdown": float(row.get("max_drawdown", 0)),
                    "momentum_drawdown_penalty": float(
                        row.get("drawdown_penalty", 1.0)
                    ),
                    "momentum_dyn_formation": formation_period,
                    "momentum_dyn_skip": skip_period,
                },
            )
            for symbol, row in final_df.iterrows()
        }
