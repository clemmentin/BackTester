import logging
from collections import deque
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

import config
from .alpha_normalization import AlphaNormalizer
from .market_detector import MarketDetector, MarketState, MacroRegime, MarketRegime
from .reversal_alpha import ReversalAlphaModule
from .price_alpha import PriceAlphaModule
from .liquidity_alpha import LiquidityAlphaModule
from strategy.contracts import RawAlphaSignal, RawAlphaSignalDict
from .momentum_alpha import MomentumAlphaModule


class AlphaSource(Enum):

    REVERSAL = "reversal"
    PRICE = "price"
    LIQUIDITY = "liquidity"
    MOMENTUM = "momentum"


class AlphaEngine:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.all_params = kwargs.copy()

        # Initialize factor modules
        self.reversal_module = ReversalAlphaModule(**self.all_params)
        self.price_module = PriceAlphaModule(**self.all_params)
        self.liquidity_module = LiquidityAlphaModule(**self.all_params)
        self.momentum_module = MomentumAlphaModule(**self.all_params)
        self.market_detector = MarketDetector(**self.all_params)

        engine_params = self.all_params.get("alpha_engine", {})

        # Base weights - calibrated based on long-term historical IC performance
        self.base_weights = {
            AlphaSource.REVERSAL: engine_params.get("reversal_weight", 0.35),
            AlphaSource.PRICE: engine_params.get("price_weight", 0.10),
            AlphaSource.LIQUIDITY: engine_params.get("liquidity_weight", 0.15),
            AlphaSource.MOMENTUM: engine_params.get("momentum_weight", 0.40),
        }

        total_base = sum(self.base_weights.values())
        self.base_weights = {k: v / total_base for k, v in self.base_weights.items()}

        # IC monitoring configuration
        ic_params = self.all_params.get("ic_monitoring", {})
        self.ic_enabled = ic_params.get("enabled", True)
        self.ic_lookback = ic_params.get("lookback_period", 180)
        self.fwd_return_period = ic_params.get("forward_return_period", 20)
        self.ic_smoothing_alpha = ic_params.get("smoothing_alpha", 0.20)
        self.ic_threshold = ic_params.get("ic_threshold", 0.02)
        self.ic_strong_threshold = ic_params.get("ic_strong_threshold", 0.10)

        # Historical data for IC calculation
        self.factor_scores_history = deque(
            maxlen=self.ic_lookback + self.fwd_return_period + 10
        )
        self.prices_history = deque(
            maxlen=self.ic_lookback + self.fwd_return_period + 10
        )
        self.smoothed_ic = {source: 0.0 for source in AlphaSource}
        self.current_weights = self.base_weights.copy()

        # Signal combination strategy
        self.combination_mode = engine_params.get("combination_mode", "smart_weighted")
        self.conflict_threshold = engine_params.get("conflict_threshold", 0.6)

        # Elite signal protection prioritizes the best factor if its signal is strong
        self.elite_factor = AlphaSource.REVERSAL
        self.elite_confidence_threshold = engine_params.get(
            "elite_confidence_threshold", 0.75
        )

        # Weight constraints
        self.min_weight = engine_params.get("min_factor_weight", 0.10)
        self.max_weight = engine_params.get("max_factor_weight", 0.70)

        # Quality filtering
        quality_params = engine_params.get("signal_quality", {})
        self.quality_filter_enabled = quality_params.get("enabled", True)
        self.min_score_threshold = quality_params.get("min_score", 0.15)
        self.min_confidence_threshold = quality_params.get("min_confidence", 0.30)

        # Normalization
        self.normalizer = AlphaNormalizer(**kwargs)
        self.normalization_enabled = engine_params.get("normalization_enabled", True)

        self.logger.info(
            f"AlphaEngine initialized: "
            f"Reversal={self.base_weights[AlphaSource.REVERSAL]:.0%}, "
            f"Price={self.base_weights[AlphaSource.PRICE]:.0%}, "
            f"Liquidity={self.base_weights[AlphaSource.LIQUIDITY]:.0%} | "
            f"Mode={self.combination_mode}, "
            f"Weights=[{self.min_weight:.0%}, {self.max_weight:.0%}]"
        )

    def generate_alpha_signals(
        self,
        market_data: pd.DataFrame,
        symbols: List[str],
        timestamp: pd.Timestamp,
        fundamental_data: Optional[Dict[str, Dict]] = None,
        macro_data: Optional[pd.DataFrame] = None,
        market_state: Optional[MarketState] = None,
    ) -> RawAlphaSignalDict:
        """
        Main function to generate alpha signals for a given timestamp.
        """
        if market_state is None:
            market_state = self.market_detector.detect_market_state(
                market_data, timestamp, symbols, macro_data=macro_data
            )

        factor_signals = self._generate_raw_factor_signals(
            market_data, timestamp, market_state.regime.value
        )

        self._update_history_for_ic(timestamp, factor_signals, market_data)

        if self.ic_enabled:
            self._update_ic_scores(timestamp)

        self.current_weights = self._get_dynamic_weights(market_state)

        combined_signals = self._combine_signals_smart(
            factor_signals, symbols, market_state
        )

        if self.normalization_enabled and combined_signals:
            combined_signals = self.normalizer.normalize_factors(
                combined_signals, market_data
            )

        ic_summary = ", ".join(
            [f"{s.name}={self.smoothed_ic[s]:.3f}" for s in AlphaSource]
        )
        weight_summary = ", ".join(
            [f"{s.name[0]}={self.current_weights[s]:.2f}" for s in AlphaSource]
        )

        self.logger.info(
            f"Generated {len(combined_signals)} signals | Weights: {weight_summary} | IC: {ic_summary}"
        )

        return combined_signals

    def _get_dynamic_weights(
        self, market_state: MarketState
    ) -> Dict[AlphaSource, float]:
        return self.base_weights

    def _combine_signals_smart(
        self,
        factor_signals: Dict[AlphaSource, Dict],
        symbols: List[str],
        market_state: MarketState,
    ) -> RawAlphaSignalDict:
        final_signals = {}
        for source, signals in factor_signals.items():
            if signals:
                for symbol, signal in signals.items():
                    # If a symbol is signaled by multiple factors, the last one seen wins.
                    # This is acceptable because the crucial ranking happens later in the normalizer.
                    final_signals[symbol] = signal

        return final_signals

    def _generate_raw_factor_signals(
        self, market_data: pd.DataFrame, timestamp: pd.Timestamp, regime: str
    ) -> Dict[AlphaSource, Dict]:
        today_data = market_data.loc[
            market_data.index.get_level_values("timestamp") <= timestamp
        ]
        prices_df = today_data["close"].unstack(level="symbol")
        opens_df = today_data["open"].unstack(level="symbol")
        volumes_df = today_data["volume"].unstack(level="symbol")

        factor_signals = {
            AlphaSource.REVERSAL: self.reversal_module.calculate_batch_reversal_trend_signals(
                prices_df, volumes_df, timestamp, regime
            ),
            AlphaSource.PRICE: self.price_module.calculate_batch_price_signals(
                prices_df, opens_df, volumes_df, timestamp, regime
            ),
            AlphaSource.LIQUIDITY: self.liquidity_module.calculate_batch_liquidity_signals(
                prices_df, volumes_df, timestamp, regime
            ),
            AlphaSource.MOMENTUM: self.momentum_module.calculate_batch_momentum_signals(
                prices_df, volumes_df, timestamp, regime
            ),
        }
        return factor_signals

    def _update_ic_scores(self, timestamp: pd.Timestamp):
        """Update Information Coefficient (IC) scores for each factor."""
        required_len = self.fwd_return_period + 1
        if len(self.prices_history) < required_len:
            return

        signal_date_idx = -required_len
        past_ts, past_prices = self.prices_history[signal_date_idx]
        _, past_scores = self.factor_scores_history[signal_date_idx]
        current_ts, current_prices = self.prices_history[-1]

        if current_ts != timestamp:
            self.logger.warning(
                f"Timestamp mismatch in IC calc. History: {current_ts}, Current: {timestamp}"
            )
            return

        common_price_symbols = past_prices.keys() & current_prices.keys()
        fwd_returns = {
            sym: (current_prices[sym] / past_prices[sym]) - 1.0
            for sym in common_price_symbols
            if past_prices.get(sym, 0) > 0
        }

        for source in AlphaSource:
            if source not in past_scores or not past_scores[source]:
                continue

            common_symbols = past_scores[source].keys() & fwd_returns.keys()
            if len(common_symbols) < 15:
                continue

            try:
                scores = np.array([past_scores[source][sym] for sym in common_symbols])
                returns = np.array([fwd_returns[sym] for sym in common_symbols])

                if np.std(scores) < 1e-6 or np.std(returns) < 1e-6:
                    continue

                ic, _ = stats.spearmanr(scores, returns)
                ic = 0.0 if np.isnan(ic) else ic

                prev_ic = self.smoothed_ic[source]
                self.smoothed_ic[source] = (
                    self.ic_smoothing_alpha * ic
                    + (1 - self.ic_smoothing_alpha) * prev_ic
                )
            except (KeyError, IndexError) as e:
                self.logger.warning(f"Could not calculate IC for {source.name}: {e}")

    def _update_history_for_ic(
        self,
        timestamp: pd.Timestamp,
        factor_signals: Dict[AlphaSource, Dict],
        market_data: pd.DataFrame,
    ):
        """Store current signals and prices for future IC calculation."""
        if self.prices_history and self.prices_history[-1][0] == timestamp:
            return

        timestamp_mask = market_data.index.get_level_values("timestamp") == timestamp
        if not timestamp_mask.any():
            return

        current_prices = (
            market_data.loc[timestamp_mask, "close"].groupby("symbol").last().to_dict()
        )
        if not current_prices:
            return

        current_scores = {
            source: {sym: sig.score for sym, sig in signals.items()}
            for source, signals in factor_signals.items()
            if signals
        }

        self.prices_history.append((timestamp, current_prices))
        self.factor_scores_history.append((timestamp, current_scores))
