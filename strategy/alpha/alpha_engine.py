import logging
import json
from collections import deque
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from .alpha_normalization import AlphaNormalizer
from .market_detector import MarketDetector, MarketState, MacroRegime
from .reversal_alpha import ReversalAlphaModule
from .price_alpha import PriceAlphaModule
from .liquidity_alpha import LiquidityAlphaModule
from strategy.contracts import RawAlphaSignal, RawAlphaSignalDict
from .momentum_alpha import MomentumAlphaModule
from .bayesian_ev_calculator import BayesianEVCalculator, SignalOutcome


def _calculate_alpha_task(module_instance, method_name, source_enum, *args):
    """Wrapper function to execute an alpha module's calculation method.
    Returns a tuple containing the source enum and the calculated signals.

    Args:
      module_instance:
      method_name:
      source_enum:
      *args:

    Returns:

    """
    try:
        calculation_method = getattr(module_instance, method_name)
        result = calculation_method(*args)
        return (source_enum, result)
    except Exception as e:
        logging.error(f"Error in parallel alpha task for {source_enum.name}: {e}")
        return (source_enum, {})


class AlphaSource(Enum):
    """ """

    REVERSAL = "reversal"
    PRICE = "price"
    LIQUIDITY = "liquidity"
    MOMENTUM = "momentum"


class AlphaEngine:
    """ """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.all_params = kwargs.copy()

        self.executor = ProcessPoolExecutor(max_workers=1)

        self.reversal_module = ReversalAlphaModule(**self.all_params)
        self.price_module = PriceAlphaModule(**self.all_params)
        self.liquidity_module = LiquidityAlphaModule(**self.all_params)
        self.momentum_module = MomentumAlphaModule(**self.all_params)
        self.market_detector = MarketDetector(**self.all_params)

        engine_params = self.all_params.get("alpha_engine", {})

        self.base_weights = engine_params.get(
            "base_weights",
            {
                "reversal": 0.45,
                "liquidity": 0.45,
                "price": 0.10,
            },
        )

        total_base = sum(self.base_weights.values())
        self.base_weights = {k: v / total_base for k, v in self.base_weights.items()}

        self.momentum_confirmer_sensitivity = engine_params.get(
            "momentum_confirmer_sensitivity", 0.15
        )

        ic_params = self.all_params.get("ic_monitoring", {})
        self.ic_enabled = ic_params.get("enabled", True)
        self.ic_lookback = ic_params.get("lookback_period", 180)
        self.fwd_return_period = ic_params.get("forward_return_period", 20)
        self.ic_smoothing_alpha = ic_params.get("smoothing_alpha", 0.20)
        self.ic_dynamic_weights = ic_params.get("use_dynamic_weights", True)
        self.ic_weight_sensitivity = ic_params.get("weight_sensitivity", 1.19)

        self.deviation_sensitivity = {
            AlphaSource.REVERSAL: np.array([-0.2, -0.3, -0.1, 0.0, 0.0, 0.3, 0.0, 0.0]),
        }
        self.deviation_weight_sensitivity = engine_params.get(
            "deviation_sensitivity", 0.5
        )

        self.all_data = self.all_params.get("all_processed_data")
        self.bayesian_ev = BayesianEVCalculator(**self.all_params)
        self.active_signals: Dict[str, Dict] = {}
        self.signal_evaluation_period = self.all_params.get(
            "signal_evaluation_period", 20
        )
        self._last_generated_signals_cache = {}

        self.factor_scores_history = deque(
            maxlen=self.ic_lookback + self.fwd_return_period + 10
        )
        self.prices_history = deque(
            maxlen=self.ic_lookback + self.fwd_return_period + 10
        )
        self.smoothed_ic = {source: 0.0 for source in AlphaSource}
        self.current_weights = self.base_weights.copy()

        self.combination_mode = engine_params.get("combination_mode", "smart_weighted")
        self.min_weight = engine_params.get("min_factor_weight", 0.10)
        self.max_weight = engine_params.get("max_factor_weight", 0.70)

        self.normalizer = AlphaNormalizer(**kwargs)
        self.normalization_enabled = engine_params.get("normalization_enabled", True)
        self.deviation_weight_sensitivity = kwargs.get("deviation_sensitivity", 0.5)
        self.logger.info(
            f"AlphaEngine initialized with parallel execution support ({os.cpu_count()} workers)."
        )

    def shutdown(self):
        """Shuts down the process pool executor."""
        self.logger.info("Shutting down AlphaEngine's process pool...")
        self.executor.shutdown()

    def save_bayesian_priors(self, filepath: str):
        """

        Args:
          filepath: str:

        Returns:

        """
        try:
            priors_data = self.bayesian_ev.export_priors()
            with open(filepath, "w") as f:
                json.dump(priors_data, f, indent=4)
            self.logger.info(f"Bayesian priors saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save Bayesian priors: {e}")

    def load_bayesian_priors(self, filepath: str):
        """

        Args:
          filepath: str:

        Returns:

        """
        try:
            with open(filepath, "r") as f:
                priors_data = json.load(f)
            self.bayesian_ev.import_priors(priors_data)
            self.logger.info(f"Bayesian priors loaded from {filepath}")
        except FileNotFoundError:
            self.logger.warning(
                f"Bayesian priors file not found at {filepath}, starting with default priors."
            )
        except Exception as e:
            self.logger.error(f"Failed to load Bayesian priors: {e}")

    def record_signal_entry(
        self,
        symbol: str,
        entry_price: float,
        timestamp: pd.Timestamp,
        spy_entry_price: Optional[float] = None,
    ):
        """

        Args:
          symbol: str:
          entry_price: float:
          timestamp: pd.Timestamp:
          spy_entry_price: Optional[float]:  (Default value = None)

        Returns:

        """
        if symbol in self._last_generated_signals_cache:
            signal_data_at_generation = self._last_generated_signals_cache[symbol]
            entry_market_state = self.market_detector.detect_market_state(
                self.all_data, timestamp
            )
            entry_regime = entry_market_state.regime.value.lower()

            # 2. Recalculate the Expected Value using the entry-time market state
            ic_val = float(
                self.smoothed_ic.get(signal_data_at_generation["factor_source"], 0.0)
            )

            ev_at_entry, diagnostics_at_entry = (
                self.bayesian_ev.calculate_expected_value(
                    signal_score=signal_data_at_generation["score"],
                    signal_confidence=signal_data_at_generation["confidence"],
                    factor_source=signal_data_at_generation["factor_source"],
                    regime=entry_regime,
                    factor_ic=ic_val,
                )
            )

            # 3. Store both the original EV (for analysis) and the new EV (for learning)
            self.active_signals[symbol] = {
                "entry_price": entry_price,
                "entry_timestamp": timestamp,
                "signal_data": signal_data_at_generation,  # Original signal data from time t
                "market_regime_at_entry": entry_regime,  # The correct regime
                "factor_source": signal_data_at_generation["factor_source"],
                "predicted_ev_at_entry": ev_at_entry,  # The correctly-timed EV for learning
                "spy_entry_price": spy_entry_price,
            }
            self.logger.debug(f"Recorded entry for {symbol} at ${entry_price:.2f}")

    def evaluate_signal_outcomes(
        self,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp,
        market_data: Optional[pd.DataFrame] = None,
    ):
        """

        Args:
          current_prices: Dict[str:
          float]:
          timestamp: pd.Timestamp:
          market_data: Optional[pd.DataFrame]:  (Default value = None)

        Returns:

        """
        completed_symbols = []
        for symbol, active_signal in self.active_signals.items():
            days_held = (timestamp - active_signal["entry_timestamp"]).days

            if days_held >= self.signal_evaluation_period:
                if symbol in current_prices:
                    exit_price = current_prices[symbol]
                    entry_price = active_signal["entry_price"]
                    actual_return = (exit_price / entry_price) - 1.0

                    spy_return = 0.0
                    use_spy_benchmark = True

                    if market_data is not None:
                        try:
                            spy_data = market_data.xs("SPY", level="symbol")
                            entry_timestamp = active_signal["entry_timestamp"]
                            spy_entry = spy_data.loc[
                                spy_data.index <= entry_timestamp, "close"
                            ].iloc[-1]
                            spy_exit = spy_data.loc[
                                spy_data.index <= timestamp, "close"
                            ].iloc[-1]
                            spy_return = (spy_exit / spy_entry) - 1.0
                        except (KeyError, IndexError, ValueError):
                            use_spy_benchmark = False
                    else:
                        use_spy_benchmark = False

                    MIN_SUCCESS_RETURN = 0.001
                    if use_spy_benchmark:
                        was_winner = actual_return > max(spy_return, MIN_SUCCESS_RETURN)
                        is_strong_failure = (actual_return < spy_return) and (
                            actual_return < 0.0
                        )
                    else:
                        was_winner = actual_return > MIN_SUCCESS_RETURN
                        is_strong_failure = actual_return < 0.0

                    signal_data = active_signal["signal_data"]
                    ev_for_learning = active_signal["predicted_ev_at_entry"]
                    regime_for_learning = active_signal["market_regime_at_entry"]
                    outcome = SignalOutcome(
                        symbol=symbol,
                        timestamp=timestamp,
                        factor_source=active_signal["factor_source"],
                        regime=regime_for_learning,
                        entry_score=signal_data["score"],
                        entry_confidence=signal_data["confidence"],
                        predicted_ev=ev_for_learning,
                        was_winner=was_winner,
                        actual_return=actual_return,
                        hold_period=days_held,
                        market_return=spy_return if use_spy_benchmark else None,
                        is_strong_failure=is_strong_failure,
                    )
                    self.bayesian_ev.update_with_outcome(outcome)
                    completed_symbols.append(symbol)
                    self._save_outcome_for_ml(outcome)

        for symbol in completed_symbols:
            if symbol in self.active_signals:
                del self.active_signals[symbol]

    def generate_alpha_signals(
        self,
        market_data: pd.DataFrame,
        symbols: List[str],
        timestamp: pd.Timestamp,
        fundamental_data: Optional[Dict[str, Dict]] = None,
        macro_data: Optional[pd.DataFrame] = None,
        market_state: Optional[MarketState] = None,
    ) -> Dict[AlphaSource, RawAlphaSignalDict]:
        """

        Args:
          market_data: pd.DataFrame:
          symbols: List[str]:
          timestamp: pd.Timestamp:
          fundamental_data: Optional[Dict[str:
          Dict]]:  (Default value = None)
          macro_data: Optional[pd.DataFrame]:  (Default value = None)
          market_state: Optional[MarketState]:  (Default value = None)

        Returns:

        """
        if market_state is None:
            market_state = self.market_detector.detect_market_state(
                market_data, timestamp, symbols, macro_data=macro_data
            )

        factor_signals = self._generate_raw_factor_signals(
            market_data, timestamp, market_state
        )

        # (MODIFIED) Apply momentum as a confirmer to Reversal and Liquidity signals.
        factor_signals = self._apply_momentum_confirmation(factor_signals)

        # (MODIFIED) Remove Momentum from the primary factor signals after it has been used for confirmation.
        if AlphaSource.MOMENTUM in factor_signals:
            del factor_signals[AlphaSource.MOMENTUM]

        self._update_history_for_ic(timestamp, factor_signals, market_data)
        if self.ic_enabled:
            self._update_ic_scores(timestamp)

        self.current_weights = self._get_dynamic_weights(market_state)
        self._calculate_expected_values(factor_signals, market_state)

        self._last_generated_signals_cache.clear()
        for source, signals in factor_signals.items():
            if signals:
                for symbol, signal in signals.items():
                    self._last_generated_signals_cache[symbol] = {
                        "score": signal.score,
                        "confidence": signal.confidence,
                        "expected_value": signal.expected_value,
                        "factor_source": source.name.lower(),
                        "market_regime": market_state.regime.value.lower(),
                    }

        ic_summary = ", ".join(
            [
                f"{s}={self.smoothed_ic.get(s, 0.0):.3f}"
                for s in self.base_weights.keys()
            ]
        )
        self.logger.info(
            f"Generated factor signals: "
            + ", ".join(
                [
                    f"{src.name}:{len(sig) if sig else 0}"
                    for src, sig in factor_signals.items()
                ]
            )
            + f" | IC: {ic_summary}"
        )
        return factor_signals

    def _apply_momentum_confirmation(
        self, factor_signals: Dict[AlphaSource, RawAlphaSignalDict]
    ) -> Dict[AlphaSource, RawAlphaSignalDict]:
        """(MODIFIED) Use Momentum signals to confirm/reject Reversal and Liquidity signals
        by adjusting their confidence scores.

        Args:
          factor_signals: Dict[AlphaSource:
          RawAlphaSignalDict]:

        Returns:

        """
        momentum_signals = factor_signals.get(AlphaSource.MOMENTUM, {})
        if not momentum_signals:
            return factor_signals

        self.logger.debug(
            f"Applying momentum confirmation with {len(momentum_signals)} momentum signals."
        )

        for source_to_confirm in [AlphaSource.REVERSAL, AlphaSource.LIQUIDITY]:
            if source_to_confirm in factor_signals:
                for symbol, signal in factor_signals[source_to_confirm].items():
                    if symbol in momentum_signals:
                        momentum_signal = momentum_signals[symbol]
                        momentum_score = momentum_signal.score
                        signal_score = signal.score

                        # If momentum and signal directions align, boost confidence. If not, penalize.
                        # The adjustment is proportional to the momentum score's magnitude.
                        alignment = np.sign(momentum_score * signal_score)
                        confidence_adjustment = (
                            abs(momentum_score)
                            * self.momentum_confirmer_sensitivity
                            * alignment
                        )

                        original_confidence = signal.confidence
                        signal.confidence = float(
                            np.clip(signal.confidence + confidence_adjustment, 0.0, 1.0)
                        )
                        self.logger.debug(
                            f"Symbol {symbol} ({source_to_confirm.name}): "
                            f"Original confidence {original_confidence:.2f}, "
                            f"Momentum score {momentum_score:.2f}, "
                            f"Adjustment {confidence_adjustment:.2f}, "
                            f"New confidence {signal.confidence:.2f}."
                        )
        return factor_signals

    def _get_dynamic_weights(
        self, market_state: MarketState
    ) -> Dict[AlphaSource, float]:
        """

        Args:
          market_state: MarketState:

        Returns:

        """
        if not self.ic_dynamic_weights or not self.ic_enabled:
            return self.base_weights

        tilts = {}
        for src in self.base_weights.keys():  # (MODIFIED) Iterate over new base weights
            ic_val = self.smoothed_ic.get(src, 0.0)
            ic_mult = 1.0 + self.ic_weight_sensitivity * ic_val
            deviation_mult = 1.0
            if (
                market_state.deviation_vector is not None
                and src in self.deviation_sensitivity
            ):
                sensitivity = self.deviation_sensitivity[src]
                deviation_effect = np.dot(market_state.deviation_vector, sensitivity)
                deviation_mult = (
                    1.0 + self.deviation_weight_sensitivity * deviation_effect
                )
            combined_mult = np.clip(ic_mult * deviation_mult, 0.5, 1.5)
            tilts[src] = combined_mult

        raw = {
            src: self.base_weights.get(src, 0.0) * tilts.get(src, 1.0)
            for src in self.base_weights.keys()
        }
        total = sum(raw.values())
        if total <= 0:
            return self.base_weights

        normalized = {src: raw[src] / total for src in self.base_weights.keys()}
        clipped = {
            src: float(np.clip(normalized[src], self.min_weight, self.max_weight))
            for src in self.base_weights.keys()
        }
        clip_sum = sum(clipped.values())
        if clip_sum <= 0:
            return self.base_weights

        final = {src: clipped[src] / clip_sum for src in self.base_weights.keys()}
        return final

    def _generate_raw_factor_signals(
        self,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_state: MarketState,
    ) -> Dict[AlphaSource, Dict]:
        """MODIFIED: Implements data slicing before parallel execution to reduce IPC overhead.

        Args:
          market_data: pd.DataFrame:
          timestamp: pd.Timestamp:
          market_state: MarketState:

        Returns:

        """
        today_data = market_data.loc[
            market_data.index.get_level_values("timestamp") <= timestamp
        ]

        # --- OPTIMIZATION: Pre-slicing data to reduce IPC overhead ---
        MAX_REQUIRED_LOOKBACK = 252  # A safe, generous lookback (approx. 1 year)

        all_timestamps = today_data.index.get_level_values("timestamp").unique()
        if len(all_timestamps) > MAX_REQUIRED_LOOKBACK:
            required_start_date = all_timestamps[-MAX_REQUIRED_LOOKBACK]
            data_slice = today_data[
                today_data.index.get_level_values("timestamp") >= required_start_date
            ]
        else:
            data_slice = today_data

        prices_df = data_slice["close"].unstack(level="symbol")
        opens_df = data_slice["open"].unstack(level="symbol")
        volumes_df = data_slice["volume"].unstack(level="symbol")
        regime = market_state.regime.value

        tasks = [
            (
                self.reversal_module,
                "calculate_batch_reversal_trend_signals",
                AlphaSource.REVERSAL,
                prices_df,
                volumes_df,
                timestamp,
                market_state,
            ),
            (
                self.price_module,
                "calculate_batch_price_signals",
                AlphaSource.PRICE,
                prices_df,
                opens_df,
                volumes_df,
                timestamp,
                regime,
            ),
            (
                self.liquidity_module,
                "calculate_batch_liquidity_signals",
                AlphaSource.LIQUIDITY,
                prices_df,
                volumes_df,
                timestamp,
                regime,
            ),
            (
                self.momentum_module,
                "calculate_batch_momentum_signals",
                AlphaSource.MOMENTUM,
                prices_df,
                volumes_df,
                timestamp,
                market_state,
            ),
        ]

        future_to_source = {
            self.executor.submit(_calculate_alpha_task, *task): task[2]
            for task in tasks
        }

        factor_signals = {}
        for future in as_completed(future_to_source):
            try:
                source_enum, result_data = future.result()
                factor_signals[source_enum] = result_data
            except Exception as e:
                source = future_to_source[future]
                self.logger.error(f"A task for alpha source {source.name} failed: {e}")
                factor_signals[source] = {}

        return factor_signals

    def _update_ic_scores(self, timestamp: pd.Timestamp):
        """

        Args:
          timestamp: pd.Timestamp:

        Returns:

        """
        required_len = self.fwd_return_period + 1
        if len(self.prices_history) < required_len:
            return

        past_ts, past_prices = self.prices_history[-required_len]
        _, past_scores = self.factor_scores_history[-required_len]
        current_ts, current_prices = self.prices_history[-1]

        if current_ts != timestamp:
            return

        common_price_symbols = past_prices.keys() & current_prices.keys()
        fwd_returns = {
            sym: (current_prices[sym] / past_prices[sym]) - 1.0
            for sym in common_price_symbols
            if past_prices.get(sym, 0) > 0
        }

        for (
            source
        ) in self.base_weights.keys():  # (MODIFIED) Iterate over new base weights
            if source not in past_scores or not past_scores[source]:
                continue

            common_symbols = past_scores[source].keys() & fwd_returns.keys()
            if len(common_symbols) < 15:
                continue

            scores = np.array([past_scores[source][sym] for sym in common_symbols])
            returns = np.array([fwd_returns[sym] for sym in common_symbols])

            if np.std(scores) < 1e-6 or np.std(returns) < 1e-6:
                continue

            ic, _ = stats.spearmanr(scores, returns)
            ic = 0.0 if np.isnan(ic) else ic
            prev_ic = self.smoothed_ic.get(source, 0.0)
            self.smoothed_ic[source] = (self.ic_smoothing_alpha * ic) + (
                1 - self.ic_smoothing_alpha
            ) * prev_ic

    def _update_history_for_ic(
        self,
        timestamp: pd.Timestamp,
        factor_signals: Dict[AlphaSource, Dict],
        market_data: pd.DataFrame,
    ):
        """

        Args:
          timestamp: pd.Timestamp:
          factor_signals: Dict[AlphaSource:
          Dict]:
          market_data: pd.DataFrame:

        Returns:

        """
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

    def _calculate_expected_values(
        self, factor_signals: Dict[AlphaSource, Dict], market_state: MarketState
    ) -> None:
        """

        Args:
          factor_signals: Dict[AlphaSource:
          Dict]:
          market_state: MarketState:

        Returns:

        """
        regime = market_state.regime.value.lower()
        for source, signals in factor_signals.items():
            if not signals:
                continue

            ic_val = float(self.smoothed_ic.get(source, 0.0))
            factor_source_str = source.name.lower()

            for symbol, signal in signals.items():
                ev, diagnostics = self.bayesian_ev.calculate_expected_value(
                    signal_score=signal.score,
                    signal_confidence=signal.confidence,
                    factor_source=factor_source_str,
                    regime=regime,
                    factor_ic=ic_val,
                    components=signal.components,
                )
                signal.expected_value = ev
                setattr(
                    signal,
                    "expected_loss",
                    diagnostics.get("expected_loss", self.bayesian_ev.base_stop_loss),
                )

    def _save_outcome_for_ml(self, outcome: "SignalOutcome"):
        """

        Args:
          outcome: "SignalOutcome":

        Returns:

        """
        outcome_dict = {
            k: v for k, v in outcome.__dict__.items() if not isinstance(v, pd.Timestamp)
        }
        outcome_dict["timestamp"] = outcome.timestamp.isoformat()

        filepath = "./data/ml_training_data.csv"
        df_new = pd.DataFrame([outcome_dict])

        if not os.path.isfile(filepath):
            df_new.to_csv(filepath, index=False)
        else:
            df_new.to_csv(filepath, mode="a", header=False, index=False)

    def finalize_learning(self):
        """ """
        self.logger.info("Finalizing AlphaEngine learning...")
        if hasattr(self.bayesian_ev, "finalize_learning"):
            self.bayesian_ev.finalize_learning()
        else:
            self.logger.warning(
                "BayesianEVCalculator does not have a finalize_learning method."
            )
