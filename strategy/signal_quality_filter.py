import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from strategy.contracts import RawAlphaSignal, RawAlphaSignalDict


class SignalQualityFilter:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        quality_config = kwargs.get("signal_quality", {})

        self.enabled = quality_config.get("enable_quality_filter", True)

        # === Core Quality Thresholds ===
        # Absolute strength: eliminate clearly weak signals
        self.min_score_threshold = quality_config.get("min_score_threshold", 0.12)

        # Confidence: eliminate unreliable signals
        self.min_confidence_threshold = quality_config.get(
            "min_confidence_threshold", 0.30
        )

        # === Component Quality Check ===
        # Require at least one factor to show strong conviction
        self.require_strong_component = quality_config.get(
            "require_strong_component", True
        )
        self.strong_component_threshold = quality_config.get(
            "strong_component_threshold", 0.20
        )

        # === Outlier Detection ===
        # Catch extreme scores that may be data errors
        self.enable_outlier_detection = quality_config.get(
            "enable_outlier_detection", True
        )
        self.outlier_score_threshold = quality_config.get(
            "outlier_score_threshold", 0.95
        )
        self.outlier_confidence_threshold = quality_config.get(
            "outlier_confidence_threshold", 0.15
        )

        # === Agreement Check (Optional) ===
        # Ensure signal isn't driven by single weak factor when others disagree
        self.require_minimum_agreement = quality_config.get(
            "require_minimum_agreement", True
        )
        self.min_agreeing_components = quality_config.get("min_agreeing_components", 1)

        # === Regime-based adaptive thresholds ===
        self.enable_regime_adaptation = quality_config.get(
            "enable_regime_adaptation", True
        )
        self.regime_threshold_adjustments = {
            "crisis": {"score_mult": 0.70, "conf_mult": 0.85},  # More lenient in crisis
            "bear": {"score_mult": 0.80, "conf_mult": 0.90},
            "volatile": {"score_mult": 0.85, "conf_mult": 0.90},
            "normal": {"score_mult": 1.00, "conf_mult": 1.00},
            "bull": {"score_mult": 1.10, "conf_mult": 1.05},  # More strict in bull
            "strong_bull": {"score_mult": 1.15, "conf_mult": 1.10},
        }

        self.logger.info(
            f"SignalQualityFilter (Refactored): "
            f"min_score={self.min_score_threshold:.2f}, "
            f"min_confidence={self.min_confidence_threshold:.2f}, "
            f"strong_component={self.require_strong_component}, "
            f"outlier_detection={self.enable_outlier_detection}"
        )
        self.logger.info(
            "NOTE: Divergence filtering REMOVED (high divergence = factor complementarity)"
        )

    def filter_signals(
        self, signals: RawAlphaSignalDict, regime: str = "NORMAL"
    ) -> Tuple[RawAlphaSignalDict, Dict]:
        if not self.enabled:
            return signals, {"filtered": 0, "total": len(signals)}

        # Get regime-adjusted thresholds
        adjusted_thresholds = self._get_regime_thresholds(regime)

        filtered_signals = {}
        filter_stats = {
            "total": len(signals),
            "passed_all": 0,
            "filtered_weak_score": 0,
            "filtered_low_confidence": 0,
            "filtered_no_strong_component": 0,
            "filtered_outlier": 0,
            "filtered_insufficient_agreement": 0,
            "regime": regime,
            "score_threshold": adjusted_thresholds["min_score"],
            "conf_threshold": adjusted_thresholds["min_confidence"],
        }

        for symbol, signal in signals.items():
            # === Filter 1: Minimum absolute score ===
            if abs(signal.score) < adjusted_thresholds["min_score"]:
                filter_stats["filtered_weak_score"] += 1
                self.logger.debug(
                    f"{symbol}: Filtered weak score {signal.score:.3f} "
                    f"< {adjusted_thresholds['min_score']:.3f}"
                )
                continue

            # === Filter 2: Minimum confidence ===
            if signal.confidence < adjusted_thresholds["min_confidence"]:
                filter_stats["filtered_low_confidence"] += 1
                self.logger.debug(
                    f"{symbol}: Filtered low confidence {signal.confidence:.3f} "
                    f"< {adjusted_thresholds['min_confidence']:.3f}"
                )
                continue

            # === Filter 3: Strong component check ===
            if self.require_strong_component:
                if not self._has_strong_component(signal):
                    filter_stats["filtered_no_strong_component"] += 1
                    self.logger.debug(
                        f"{symbol}: Filtered - no strong individual component "
                        f"(components: {signal.components})"
                    )
                    continue

            # === Filter 4: Outlier detection ===
            if self.enable_outlier_detection:
                if self._is_outlier(signal):
                    filter_stats["filtered_outlier"] += 1
                    self.logger.warning(
                        f"{symbol}: Filtered outlier - score={signal.score:.3f}, "
                        f"conf={signal.confidence:.3f} (potential data error)"
                    )
                    continue

            # === Filter 5: Minimum agreement check ===
            if self.require_minimum_agreement:
                if not self._has_minimum_agreement(signal):
                    filter_stats["filtered_insufficient_agreement"] += 1
                    self.logger.debug(
                        f"{symbol}: Filtered insufficient agreement "
                        f"(components: {signal.components})"
                    )
                    continue

            # === Signal passed all filters ===
            filtered_signals[symbol] = signal
            filter_stats["passed_all"] += 1

        filter_rate = 1 - (len(filtered_signals) / len(signals)) if signals else 0

        self.logger.info(
            f"Quality filter ({regime}): {len(filtered_signals)}/{len(signals)} passed "
            f"({filter_rate:.1%} filtered) - "
            f"thresholds: score>{adjusted_thresholds['min_score']:.2f}, "
            f"conf>{adjusted_thresholds['min_confidence']:.2f}"
        )

        if filter_stats["filtered_outlier"] > 0:
            self.logger.warning(
                f"Detected {filter_stats['filtered_outlier']} outliers (potential data issues)"
            )

        return filtered_signals, filter_stats

    def _get_regime_thresholds(self, regime: str) -> Dict[str, float]:
        """
        Get regime-adjusted quality thresholds.

        Logic:
        - Crisis/volatile markets: lower thresholds (we need signals even if weaker)
        - Bull markets: higher thresholds (can afford to be more selective)
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

    def _has_strong_component(self, signal: RawAlphaSignal) -> bool:
        if not hasattr(signal, "components") or not signal.components:
            # No components available, assume it's fine
            return True

        # Look for alpha factor components (ignore metadata)
        alpha_factors = [
            "momentum",
            "reversal",
            "price",
            "technical",
            "quality",
            "value",
        ]

        for key, value in signal.components.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Check if this is an alpha factor
                if any(factor in key.lower() for factor in alpha_factors):
                    if abs(value) >= self.strong_component_threshold:
                        return True

        return False

    def _is_outlier(self, signal: RawAlphaSignal) -> bool:
        # Check 1: Extreme score with very low confidence
        if abs(signal.score) > self.outlier_score_threshold:
            if signal.confidence < self.outlier_confidence_threshold:
                return True

        # Check 2: Perfect score (clipping) with modest confidence
        if abs(signal.score) == 1.0:
            if signal.confidence < 0.70:
                return True

        # Check 3: Impossible combinations (NaN/Inf in components)
        if hasattr(signal, "components") and signal.components:
            for key, value in signal.components.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        self.logger.warning(
                            f"Invalid component value: {key}={value} for {signal.symbol}"
                        )
                        return True

        return False

    def _has_minimum_agreement(self, signal: RawAlphaSignal) -> bool:
        if not hasattr(signal, "components") or not signal.components:
            return True

        final_direction = np.sign(signal.score)
        if final_direction == 0:
            return False  # Zero score should have been filtered earlier

        # Count factors agreeing with final direction
        # Only count meaningful factors (exclude metadata)
        alpha_factors = [
            "momentum",
            "reversal",
            "price",
            "technical",
            "quality",
            "value",
        ]

        agreeing = 0
        disagreeing = 0
        total = 0

        for key, value in signal.components.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not np.isnan(value)
            ):
                if any(factor in key.lower() for factor in alpha_factors):
                    total += 1
                    factor_direction = np.sign(value)

                    if factor_direction == final_direction:
                        agreeing += 1
                    elif factor_direction == -final_direction:
                        disagreeing += 1

        # Need at least min_agreeing_components in agreement
        if total < 2:
            return True  # Can't have disagreement with only 1 factor

        # Pass if we have minimum agreement
        if agreeing >= self.min_agreeing_components:
            return True

        # Special case: if only 1 factor agrees but it's VERY strong
        # and others are weak, that's acceptable
        if agreeing == 1 and total > 1:
            # Check if the agreeing factor is much stronger than disagreeing ones
            agreeing_strengths = []
            disagreeing_strengths = []

            for key, value in signal.components.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and not np.isnan(value)
                ):
                    if any(factor in key.lower() for factor in alpha_factors):
                        if np.sign(value) == final_direction:
                            agreeing_strengths.append(abs(value))
                        elif np.sign(value) == -final_direction:
                            disagreeing_strengths.append(abs(value))

            if agreeing_strengths and disagreeing_strengths:
                max_agreeing = max(agreeing_strengths)
                max_disagreeing = max(disagreeing_strengths)

                # If agreeing factor is >3x stronger, accept it
                if max_agreeing > 3.0 * max_disagreeing:
                    return True

        return False

    def get_signal_quality_metrics(self, signal: RawAlphaSignal) -> Dict:
        """
        Get detailed quality metrics for a signal (for analysis/debugging).

        Returns:
            Dict with various quality metrics and filter pass/fail status
        """
        has_strong = self._has_strong_component(signal)
        is_outlier = self._is_outlier(signal)
        has_agreement = self._has_minimum_agreement(signal)

        # Calculate component statistics
        component_stats = self._calculate_component_statistics(signal)

        return {
            "symbol": signal.symbol,
            "score": signal.score,
            "confidence": signal.confidence,
            "abs_score": abs(signal.score),
            # Quality checks
            "has_strong_component": has_strong,
            "is_outlier": is_outlier,
            "has_minimum_agreement": has_agreement,
            # Component statistics
            "num_components": component_stats["count"],
            "avg_component_strength": component_stats["avg_strength"],
            "max_component_strength": component_stats["max_strength"],
            "component_std": component_stats["std"],
            # Filter pass/fail
            "passes_score_filter": abs(signal.score) >= self.min_score_threshold,
            "passes_confidence_filter": signal.confidence
            >= self.min_confidence_threshold,
            "passes_strong_component_filter": has_strong
            or not self.require_strong_component,
            "passes_outlier_filter": not is_outlier
            or not self.enable_outlier_detection,
            "passes_agreement_filter": has_agreement
            or not self.require_minimum_agreement,
            # Overall quality score
            "overall_quality": self._compute_overall_quality(signal, component_stats),
        }

    def _calculate_component_statistics(self, signal: RawAlphaSignal) -> Dict:
        """Calculate statistics about signal components."""
        if not hasattr(signal, "components") or not signal.components:
            return {
                "count": 0,
                "avg_strength": 0.0,
                "max_strength": 0.0,
                "std": 0.0,
            }

        alpha_factors = [
            "momentum",
            "reversal",
            "price",
            "technical",
            "quality",
            "value",
        ]
        component_values = []

        for key, value in signal.components.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if any(factor in key.lower() for factor in alpha_factors):
                    component_values.append(value)

        if not component_values:
            return {
                "count": 0,
                "avg_strength": 0.0,
                "max_strength": 0.0,
                "std": 0.0,
            }

        abs_values = [abs(v) for v in component_values]

        return {
            "count": len(component_values),
            "avg_strength": np.mean(abs_values),
            "max_strength": np.max(abs_values),
            "std": np.std(component_values),
        }

    def _compute_overall_quality(
        self, signal: RawAlphaSignal, component_stats: Dict
    ) -> float:
        # Component 1: Confidence
        confidence_component = signal.confidence * 0.40

        # Component 2: Score strength
        score_strength = min(abs(signal.score) / 0.60, 1.0)  # Normalize to 0.60 = max
        score_component = score_strength * 0.30

        # Component 3: Max component strength
        max_strength = min(component_stats["max_strength"] / 0.60, 1.0)
        strength_component = max_strength * 0.20

        # Component 4: Component count bonus
        count_bonus = min(component_stats["count"] / 3.0, 1.0)
        count_component = count_bonus * 0.10

        overall = (
            confidence_component
            + score_component
            + strength_component
            + count_component
        )

        return np.clip(overall, 0.0, 1.0)

    def rank_signals_by_quality(
        self, signals: RawAlphaSignalDict
    ) -> List[Tuple[str, float]]:
        quality_scores = []

        for symbol, signal in signals.items():
            component_stats = self._calculate_component_statistics(signal)
            quality = self._compute_overall_quality(signal, component_stats)
            quality_scores.append((symbol, quality))

        # Sort by quality (descending)
        quality_scores.sort(key=lambda x: x[1], reverse=True)

        return quality_scores


def apply_signal_quality_filter(
    signals: RawAlphaSignalDict, config_params: Dict, regime: str = "NORMAL"
) -> RawAlphaSignalDict:
    filter_module = SignalQualityFilter(**config_params)
    filtered_signals, stats = filter_module.filter_signals(signals, regime)

    return filtered_signals
