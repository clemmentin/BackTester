import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from strategy.contracts import RawAlphaSignal, RawAlphaSignalDict


class SignalQualityFilter:
    """ """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        quality_config = kwargs.get("signal_quality", {})

        # --- Core Quality Thresholds ---
        self.enabled = quality_config.get("enable_quality_filter", True)
        self.min_score_threshold = quality_config.get("min_score_threshold", 0.12)
        self.min_confidence_threshold = quality_config.get(
            "min_confidence_threshold", 0.30
        )

        # Bayesian EV filter is a key risk control.
        self.min_expected_value = quality_config.get("min_expected_value", 0.002)

        # --- Component Quality Check ---
        self.require_strong_component = quality_config.get(
            "require_strong_component", True
        )
        self.strong_component_threshold = quality_config.get(
            "strong_component_threshold", 0.25
        )
        self.strong_component_thresholds = quality_config.get(
            "strong_component_thresholds",
            {"reversal": 0.25, "momentum": 0.20, "price": 0.15, "liquidity": 0.10},
        )

        # --- Outlier Detection ---
        self.enable_outlier_detection = quality_config.get(
            "enable_outlier_detection", True
        )
        self.outlier_score_threshold = quality_config.get(
            "outlier_score_threshold", 0.95
        )
        self.outlier_confidence_threshold = quality_config.get(
            "outlier_confidence_threshold", 0.15
        )

        # --- Regime-based adaptive thresholds ---
        self.enable_regime_adaptation = quality_config.get(
            "enable_regime_adaptation", True
        )
        self.regime_threshold_adjustments = {
            "crisis": {"score_mult": 0.70, "conf_mult": 0.85},
            "bear": {"score_mult": 0.80, "conf_mult": 0.90},
            "volatile": {"score_mult": 0.85, "conf_mult": 0.90},
            "normal": {"score_mult": 1.00, "conf_mult": 1.00},
            "bull": {"score_mult": 1.10, "conf_mult": 1.05},
            "strong_bull": {"score_mult": 1.15, "conf_mult": 1.10},
        }
        self.logger.info(
            f"SignalQualityFilter Initialized: min_score={self.min_score_threshold:.2f}, "
            f"min_confidence={self.min_confidence_threshold:.2f}, min_ev={self.min_expected_value:.4f}"
        )

    def filter_signals(
        self, signals: RawAlphaSignalDict, regime: str = "NORMAL"
    ) -> Tuple[RawAlphaSignalDict, Dict]:
        """Filters signals based on a series of quality checks.

        Args:
          signals: RawAlphaSignalDict: 
          regime: str:  (Default value = "NORMAL")

        Returns:

        """
        if not self.enabled:
            return signals, {"filtered": 0, "total": len(signals)}

        adjusted_thresholds = self._get_regime_thresholds(regime)
        filtered_signals = {}
        filter_stats = {
            "total": len(signals),
            "passed_all": 0,
            "filtered_weak_score": 0,
            "filtered_low_confidence": 0,
            "filtered_low_ev": 0,
            "filtered_no_strong_component": 0,
            "filtered_outlier": 0,
        }

        for symbol, signal in signals.items():
            # Filter 1: Minimum score
            if abs(signal.score) < adjusted_thresholds["min_score"]:
                filter_stats["filtered_weak_score"] += 1
                continue

            # Filter 2: Minimum confidence
            if signal.confidence < adjusted_thresholds["min_confidence"]:
                filter_stats["filtered_low_confidence"] += 1
                continue

            # Filter 3: Minimum Bayesian Expected Value (Critical)
            if signal.expected_value <= self.min_expected_value:
                filter_stats["filtered_low_ev"] += 1
                continue

            # Filter 4: Strong component check
            if self.require_strong_component and not self._has_strong_component(
                signal, regime
            ):
                filter_stats["filtered_no_strong_component"] += 1
                continue

            # Filter 5: Outlier detection
            if self.enable_outlier_detection and self._is_outlier(signal):
                filter_stats["filtered_outlier"] += 1
                continue

            # MODIFICATION: Removed minimum agreement filter as it's not applicable for single-factor sleeves.

            filtered_signals[symbol] = signal
            filter_stats["passed_all"] += 1

        return filtered_signals, filter_stats

    def _get_regime_thresholds(self, regime: str) -> Dict[str, float]:
        """Get score and confidence thresholds adjusted for the current market regime.

        Args:
          regime: str: 

        Returns:

        """
        regime_key = regime.lower()
        adjustments = self.regime_threshold_adjustments.get(
            regime_key, {"score_mult": 1.0, "conf_mult": 1.0}
        )
        if not self.enable_regime_adaptation:
            adjustments = {"score_mult": 1.0, "conf_mult": 1.0}
        return {
            "min_score": self.min_score_threshold * adjustments["score_mult"],
            "min_confidence": self.min_confidence_threshold * adjustments["conf_mult"],
        }

    def _has_strong_component(
        self, signal: RawAlphaSignal, regime: str = "normal"
    ) -> bool:
        """Check if at least one underlying component of the signal is strong enough.
        MODIFIED: Simplified by removing weighted_evidence_fallback for single-factor sleeve context.

        Args:
          signal: RawAlphaSignal: 
          regime: str:  (Default value = "normal")

        Returns:

        """
        if not signal.components:
            return True

        regime_key = (regime or "normal").lower()
        adjustments = self.regime_threshold_adjustments.get(
            regime_key, {"score_mult": 1.0}
        )
        strong_mult = (
            adjustments["score_mult"] if self.enable_regime_adaptation else 1.0
        )

        for key, value in signal.components.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Infer factor from component key
                factor_name = next(
                    (f for f in self.strong_component_thresholds if f in key.lower()),
                    None,
                )
                if factor_name:
                    base_thr = self.strong_component_thresholds.get(
                        factor_name, self.strong_component_threshold
                    )
                    thr = max(0.10, base_thr * strong_mult)
                    if abs(value) >= thr:
                        return True
        return False

    def _is_outlier(self, signal: RawAlphaSignal) -> bool:
        """Detects outlier signals that are likely erroneous (e.g., max score with min confidence).

        Args:
          signal: RawAlphaSignal: 

        Returns:

        """
        if (
            abs(signal.score) > self.outlier_score_threshold
            and signal.confidence < self.outlier_confidence_threshold
        ):
            return True
        if abs(signal.score) == 1.0 and signal.confidence < 0.70:
            return True
        if signal.components:
            for value in signal.components.values():
                if isinstance(value, (int, float)) and (
                    np.isnan(value) or np.isinf(value)
                ):
                    return True
        return False

    # MODIFICATION: Removed _has_minimum_agreement and related helper methods as they are not applicable.
