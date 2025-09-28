# strategies/alpha/alpha_engine.py
import hashlib
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .market_detector import MarketDetector, MarketRegime, MarketState
from .momentum_alpha import MomentumAlphaModule
from .technical_alpha import AlphaSignal, AlphaSignalType, TechnicalAlphaModule

_worker_data = {}


def init_worker(
    preprocessed_data: Dict,
    adjusted_weights: Dict,
    market_state: MarketState,
    timestamp: pd.Timestamp,
    risk_on_symbols: set,
    engine_params: Dict,
):
    """
    Initializer for each worker process. It runs ONCE per process and loads all
    necessary data into the worker's global scope, avoiding repeated serialization.
    """
    global _worker_data
    _worker_data["preprocessed_data"] = preprocessed_data
    _worker_data["adjusted_weights"] = adjusted_weights
    _worker_data["market_state"] = market_state
    _worker_data["timestamp"] = timestamp
    _worker_data["risk_on_symbols"] = risk_on_symbols
    _worker_data["engine_params"] = engine_params

    # Modules are instantiated inside the worker to be process-safe.
    _worker_data["momentum_module"] = MomentumAlphaModule(**engine_params)
    _worker_data["technical_module"] = TechnicalAlphaModule(**engine_params)

    # Optional: uncomment to verify that workers are initializing.
    # print(f"Worker process {os.getpid()} has been initialized.")


def _worker_compute_batch_signals(
    symbol_batch: List[str],
) -> Dict[str, "CompositeAlphaSignal"]:
    """
    This is the core function executed by each worker process. It's a top-level
    function that gets its data from the global `_worker_data` storage.
    """
    global _worker_data
    # Retrieve all necessary data from the worker's global storage
    preprocessed_data = _worker_data["preprocessed_data"]
    adjusted_weights = _worker_data["adjusted_weights"]
    market_state = _worker_data["market_state"]
    timestamp = _worker_data["timestamp"]
    risk_on_symbols = _worker_data["risk_on_symbols"]
    engine_params = _worker_data["engine_params"]
    momentum_module = _worker_data["momentum_module"]
    technical_module = _worker_data["technical_module"]

    batch_signals = {}

    # --- Alpha Calculation Logic ---
    momentum_data = {
        s: preprocessed_data[s]
        for s in symbol_batch
        if s in preprocessed_data and s in risk_on_symbols
    }
    technical_data = {
        s: preprocessed_data[s] for s in symbol_batch if s in preprocessed_data
    }

    momentum_signals = (
        momentum_module.calculate_batch_momentum_signals(
            momentum_data, timestamp, market_state.regime.value
        )
        if momentum_data
        else {}
    )
    technical_signals = (
        technical_module.calculate_batch_alpha_signals(technical_data, timestamp)
        if technical_data
        else {}
    )

    # --- Signal Combination Logic ---
    for symbol in symbol_batch:
        symbol_signals = {}
        if symbol in momentum_signals:
            symbol_signals[AlphaSource.MOMENTUM] = momentum_signals[symbol]
        if symbol in technical_signals:
            symbol_signals[AlphaSource.TECHNICAL] = technical_signals[symbol]

        if symbol_signals:
            composite_signal = _create_composite_signal_for_worker(
                symbol,
                symbol_signals,
                adjusted_weights,
                market_state,
                timestamp,
                engine_params,
            )
            if composite_signal:
                batch_signals[symbol] = composite_signal
    return batch_signals


def _create_composite_signal_for_worker(
    symbol: str,
    signals_dict: Dict,
    weights: Dict,
    market_state: MarketState,
    timestamp: pd.Timestamp,
    params: Dict,
) -> Optional["CompositeAlphaSignal"]:
    """Helper function for signal combination inside a worker."""
    total_score, total_confidence, total_weight = 0.0, 0.0, 0.0
    for source, signal in signals_dict.items():
        weight = weights.get(source, 0.0)
        if weight > 0:
            total_score += signal.score * weight
            total_confidence += signal.confidence * weight
            total_weight += weight

    if total_weight == 0:
        return None

    final_score = total_score / total_weight
    final_confidence = total_confidence / total_weight

    min_conf = params.get("min_composite_confidence", 0.3)
    min_score = params.get("min_composite_score", 0.2)

    action = "hold"
    if final_confidence >= min_conf:
        if final_score > min_score:
            action = "long"
        elif final_score < -min_score:
            action = "exit"

    # Simplified but robust position sizing logic
    size_multiplier = np.clip(final_confidence * abs(final_score) * 2.0, 0.3, 1.5)

    return CompositeAlphaSignal(
        symbol=symbol,
        final_score=np.clip(final_score, -1, 1),
        final_confidence=np.clip(final_confidence, 0, 1),
        sources=signals_dict,
        weights={k: v for k, v in weights.items() if k in signals_dict},
        market_state=market_state,
        timestamp=timestamp,
        action=action,
        position_size_multiplier=size_multiplier,
    )


# ============================================================================
# AlphaEngine Supporting Enums and Dataclasses
# ============================================================================


class AlphaSource(Enum):
    MOMENTUM = "momentum"
    TECHNICAL = "technical"
    COMPOSITE = "composite"


@dataclass
class CompositeAlphaSignal:
    symbol: str
    final_score: float
    final_confidence: float
    sources: Dict[AlphaSource, AlphaSignal]
    weights: Dict[AlphaSource, float]
    market_state: MarketState
    timestamp: pd.Timestamp
    action: str
    position_size_multiplier: float


# ============================================================================
# AlphaEngine Class
# ============================================================================


class AlphaEngine:
    """
    Optimized alpha signal fusion engine with improved performance.
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Store all kwargs to pass to workers
        self.all_params = kwargs.copy()

        # Initialize modules for the main process (e.g., for sequential runs or market detection)
        self.technical_module = TechnicalAlphaModule(**self.all_params)
        self.momentum_module = MomentumAlphaModule(**self.all_params)
        self.market_detector = MarketDetector(**self.all_params)

        self.base_weights = {
            AlphaSource.MOMENTUM: kwargs.get("momentum_weight", 0.40),
            AlphaSource.TECHNICAL: kwargs.get("technical_weight", 0.35),
            AlphaSource.COMPOSITE: kwargs.get("composite_weight", 0.25),
        }
        total_weight = sum(self.base_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.base_weights = {
                k: v / total_weight for k, v in self.base_weights.items()
            }

        self.risk_on_symbols = set(kwargs.get("risk_on_symbols", []))

        # Parallel processing settings
        self.use_parallel = kwargs.get("use_parallel", True)
        self.max_workers = kwargs.get(
            "max_workers", max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
        )
        self.batch_size = kwargs.get("batch_size", 100)
        self.parallel_symbol_threshold = kwargs.get("parallel_symbol_threshold", 200)

        self.last_signals = {}
        self.signal_history = []

    def generate_alpha_signals(
        self, market_data: pd.DataFrame, symbols: List[str], timestamp: pd.Timestamp
    ) -> Dict[str, CompositeAlphaSignal]:

        market_state = self.market_detector.detect_market_state(
            market_data, timestamp, symbols
        )
        preprocessed_data = self._preprocess_market_data(
            market_data, symbols, timestamp
        )

        if self.use_parallel and len(symbols) >= self.parallel_symbol_threshold:
            self.logger.debug(
                f"High workload ({len(symbols)} symbols), engaging PARALLEL alpha engine."
            )
            final_signals = self._parallel_compute_signals(
                preprocessed_data, symbols, timestamp, market_state
            )
        else:
            if self.use_parallel:
                self.logger.debug(
                    f"Low workload ({len(symbols)} symbols), using SEQUENTIAL alpha engine to avoid overhead."
                )
            final_signals = self._sequential_compute_signals(
                preprocessed_data, symbols, timestamp, market_state
            )

        final_signals = self._apply_risk_filters(final_signals, market_state)

        self.last_signals = final_signals
        self._update_signal_history_batch(final_signals)
        return final_signals

    def _parallel_compute_signals(
        self,
        preprocessed_data: Dict,
        symbols: List[str],
        timestamp: pd.Timestamp,
        market_state: MarketState,
    ) -> Dict:
        """
        Manages the parallel computation of signals using a ProcessPoolExecutor.
        """
        composite_signals = {}
        adjusted_weights = self._adjust_weights_for_regime(market_state.regime)

        # Arguments for the worker initializer function
        init_args = (
            preprocessed_data,
            adjusted_weights,
            market_state,
            timestamp,
            self.risk_on_symbols,
            self.all_params,  # Pass all config params to the worker
        )

        with ProcessPoolExecutor(
            max_workers=self.max_workers, initializer=init_worker, initargs=init_args
        ) as executor:
            symbol_batches = [
                symbols[i : i + self.batch_size]
                for i in range(0, len(symbols), self.batch_size)
            ]

            # Submit the top-level worker function, NOT a class method
            futures = [
                executor.submit(_worker_compute_batch_signals, batch)
                for batch in symbol_batches
            ]

            for future in as_completed(futures):
                try:
                    composite_signals.update(
                        future.result(timeout=120)
                    )  # Increased timeout
                except Exception as e:
                    self.logger.error(
                        f"A worker process encountered an error: {e}", exc_info=True
                    )
        return composite_signals

    def _sequential_compute_signals(
        self,
        preprocessed_data: Dict,
        symbols: List[str],
        timestamp: pd.Timestamp,
        market_state: MarketState,
    ) -> Dict:

        self.logger.info("Running alpha calculation sequentially.")
        adjusted_weights = self._adjust_weights_for_regime(market_state.regime)

        init_worker(
            preprocessed_data,
            adjusted_weights,
            market_state,
            timestamp,
            self.risk_on_symbols,
            self.all_params,
        )
        all_signals = _worker_compute_batch_signals(symbols)
        return all_signals

    @lru_cache(maxsize=16)
    def _adjust_weights_for_regime(
        self, regime: MarketRegime
    ) -> Dict[AlphaSource, float]:
        """
        _adjust_weights_for_regime is a cached method that adjusts alpha weights based on the market regime.
        """
        # This can be expanded with more complex logic later
        return self.base_weights

    def _preprocess_market_data(
        self, market_data: pd.DataFrame, symbols: List[str], timestamp: pd.Timestamp
    ) -> Dict:
        """
        Preprocesses and caches market data for all symbols to be used by workers.
        """
        max_lookback = 252
        preprocessed = {}
        for symbol in symbols:
            try:
                if symbol in market_data.index.get_level_values("symbol"):
                    full_symbol_data = market_data.xs(symbol, level="symbol")
                    symbol_data_up_to_ts = full_symbol_data[
                        full_symbol_data.index <= timestamp
                    ]
                    symbol_data = symbol_data_up_to_ts.tail(max_lookback)
                    if len(symbol_data) >= 20:  # Basic data check
                        preprocessed[symbol] = {
                            "prices": symbol_data["close"].values,
                            "volumes": symbol_data.get(
                                "volume", pd.Series(np.zeros(len(symbol_data)))
                            ).values,
                            "highs": symbol_data.get(
                                "high", symbol_data["close"]
                            ).values,
                            "lows": symbol_data.get("low", symbol_data["close"]).values,
                        }
            except Exception as e:
                self.logger.debug(f"Could not preprocess data for {symbol}: {e}")
            return preprocessed

    def _apply_risk_filters(self, signals: Dict, market_state: MarketState) -> Dict:
        """
        Applies final risk filters, such as limiting the number of positions.
        """
        # Example: Limit max number of long positions based on market regime
        max_positions = {
            MarketRegime.CRISIS: 3,
            MarketRegime.BEAR: 5,
            MarketRegime.VOLATILE: 7,
        }.get(
            market_state.regime, 12
        )  # Default max positions

        long_signals = {s: sig for s, sig in signals.items() if sig.action == "long"}

        if len(long_signals) > max_positions:
            # Sort by a combined score of confidence and score, then trim
            sorted_longs = sorted(
                long_signals.items(),
                key=lambda item: item[1].final_confidence * item[1].final_score,
                reverse=True,
            )
            kept_symbols = {item[0] for item in sorted_longs[:max_positions]}

            # Set action to 'hold' for signals that were trimmed
            for symbol, signal in signals.items():
                if signal.action == "long" and symbol not in kept_symbols:
                    signal.action = "hold"
                    self.logger.debug(f"Risk filter trimmed long signal for {symbol}.")
        return signals

    def _update_signal_history_batch(self, signals: Dict[str, CompositeAlphaSignal]):
        """Updates the internal signal history for later analysis."""
        # This method can be expanded for diagnostics
        pass
