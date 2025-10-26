import os
import json
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from scipy import stats
from pathlib import Path
import pandas as pd


@dataclass
class BayesianPrior:
    """Stores Bayesian prior parameters for a factor-regime combination."""

    alpha: float = 15.0
    beta: float = 10.0
    mean_gain: float = 0.04
    variance_gain: float = 0.015
    mean_loss: float = -0.015
    variance_loss: float = 0.005
    pseudocount: float = 12.0
    last_update: Optional[pd.Timestamp] = None
    total_signals: int = 0


@dataclass
class SignalOutcome:
    """Records the outcome of a signal for Bayesian updating."""

    symbol: str
    timestamp: pd.Timestamp
    factor_source: str
    regime: str
    entry_score: float
    entry_confidence: float
    predicted_ev: float
    was_winner: bool
    actual_return: float
    hold_period: int
    market_return: Optional[float] = None
    is_strong_failure: bool = False


class BayesianEVCalculator:
    """Calculates Bayesian Expected Value with persistent learning and adaptive rates."""

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

        ev_config = kwargs.get("bayesian_ev", {})
        self.enabled = ev_config.get("enabled", True)
        self.use_regime_specific_priors = ev_config.get(
            "use_regime_specific_priors", True
        )
        self.prior_strength = ev_config.get("prior_strength", 20.0)

        self.gain_volatility_penalty = ev_config.get("gain_volatility_penalty", 0.25)

        self.base_learning_rate = ev_config.get("learning_rate", 0.05)
        self.min_observations_for_update = ev_config.get(
            "min_observations_for_update", 10
        )

        # MODIFIED: Added adaptive learning rate parameters.
        self.use_dynamic_learning_rate = ev_config.get(
            "use_dynamic_learning_rate", True
        )
        self.error_learning_rate_multiplier = ev_config.get(
            "error_learning_rate_multiplier", 2.5
        )
        self.max_learning_rate = ev_config.get("max_learning_rate", 0.5)

        self.use_ewma_learning = ev_config.get("use_ewma_learning", True)
        self.outcome_history_length = ev_config.get("outcome_history_length", 500)
        self.outcome_history: deque = deque(maxlen=self.outcome_history_length)

        self.enable_persistence = ev_config.get("enable_persistence", True)
        self.persistence_dir = ev_config.get(
            "persistence_dir", "./data/bayesian_priors"
        )
        self.auto_save_frequency = ev_config.get("auto_save_frequency", 100)
        self._update_counter = 0
        self._last_save_counter = 0

        self.priors: Dict[str, Dict[str, BayesianPrior]] = defaultdict(dict)
        self._initialize_priors()
        if self.enable_persistence:
            self._load_priors_from_disk()

        self.ic_influence_on_prior = ev_config.get("ic_influence_on_prior", 0.25)
        self.use_credible_intervals = ev_config.get("use_credible_intervals", True)
        self.credible_interval_level = ev_config.get("credible_interval_level", 0.80)

        self.base_stop_loss = ev_config.get("base_stop_loss", 0.02)
        self.dynamic_stop_loss = ev_config.get("dynamic_stop_loss", True)

        self.confidence_to_alpha_scaler = ev_config.get(
            "confidence_to_alpha_scaler", 5.0
        )
        self.signal_to_gain_scaler = ev_config.get("signal_to_gain_scaler", 0.10)
        self.signal_to_weight_scaler = ev_config.get("signal_to_weight_scaler", 10.0)

        self.risk_aversion_factor = ev_config.get("risk_aversion_factor", 1.0)
        self.knowledge_decay_rate = ev_config.get("knowledge_decay_rate", 0.005)
        self.logger.info(
            f"BayesianEVCalculator initialized: "
            f"gain_vol_penalty={self.gain_volatility_penalty}, "
            f"EWMA_learning={'ON' if self.use_ewma_learning else 'OFF'}, "
            f"persistence={'ON' if self.enable_persistence else 'OFF'}"
        )

    def _initialize_priors(self):
        factors = ["reversal", "price", "liquidity"]
        regimes = ["crisis", "bear", "volatile", "normal", "bull", "strong_bull"]
        base_priors = {
            "reversal": {
                "crisis": BayesianPrior(
                    alpha=20,
                    beta=7,
                    mean_gain=0.18,
                    variance_gain=0.02,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "bear": BayesianPrior(
                    alpha=16,
                    beta=9,
                    mean_gain=0.08,
                    variance_gain=0.015,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "volatile": BayesianPrior(
                    alpha=18,
                    beta=8,
                    mean_gain=0.15,
                    variance_gain=0.02,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "normal": BayesianPrior(
                    alpha=16,
                    beta=9,
                    mean_gain=0.04,
                    variance_gain=0.015,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "bull": BayesianPrior(
                    alpha=15,
                    beta=10,
                    mean_gain=0.07,
                    variance_gain=0.01,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "strong_bull": BayesianPrior(
                    alpha=14,
                    beta=11,
                    mean_gain=0.025,
                    variance_gain=0.01,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
            },
            "price": {
                "crisis": BayesianPrior(
                    alpha=10,
                    beta=10,
                    mean_gain=0.01,
                    variance_gain=0.025,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "bear": BayesianPrior(
                    alpha=15,
                    beta=10,
                    mean_gain=0.05,
                    variance_gain=0.018,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "volatile": BayesianPrior(
                    alpha=10,
                    beta=10,
                    mean_gain=0.01,
                    variance_gain=0.015,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "normal": BayesianPrior(
                    alpha=14,
                    beta=11,
                    mean_gain=0.035,
                    variance_gain=0.015,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "bull": BayesianPrior(
                    alpha=9,
                    beta=11,
                    mean_gain=0.005,
                    variance_gain=0.012,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "strong_bull": BayesianPrior(
                    alpha=13,
                    beta=12,
                    mean_gain=0.025,
                    variance_gain=0.01,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
            },
            "liquidity": {
                "crisis": BayesianPrior(
                    alpha=19,
                    beta=6,
                    mean_gain=0.14,
                    variance_gain=0.025,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "bear": BayesianPrior(
                    alpha=16,
                    beta=9,
                    mean_gain=0.045,
                    variance_gain=0.015,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "volatile": BayesianPrior(
                    alpha=15,
                    beta=10,
                    mean_gain=0.04,
                    variance_gain=0.015,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "normal": BayesianPrior(
                    alpha=16,
                    beta=9,
                    mean_gain=0.045,
                    variance_gain=0.015,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "bull": BayesianPrior(
                    alpha=17,
                    beta=8,
                    mean_gain=0.05,
                    variance_gain=0.012,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
                "strong_bull": BayesianPrior(
                    alpha=16,
                    beta=9,
                    mean_gain=0.04,
                    variance_gain=0.012,
                    mean_loss=-0.015,
                    pseudocount=12.0,
                ),
            },
        }
        for factor in factors:
            for regime in regimes:
                if factor in base_priors and regime in base_priors[factor]:
                    self.priors[factor][regime] = base_priors[factor][regime]
                else:
                    self.priors[factor][regime] = BayesianPrior()

    def _load_priors_from_disk(self):
        save_path = Path(self.persistence_dir) / "priors_latest.json"
        if not save_path.exists():
            self.logger.info(
                "No saved priors found. Starting with default initialization."
            )
            return
        try:
            with open(save_path, "r") as f:
                data = json.load(f)
            self.import_priors(data["priors"])
            metadata = data.get("metadata", {})
            self.logger.info(
                f" Loaded learned priors from previous backtests:\n"
                f"  - Total historical updates: {metadata.get('total_updates', 0)}\n"
                f"  - Previous backtest count: {metadata.get('backtest_count', 0)}\n"
                f"  - Last saved: {metadata.get('last_save_time', 'Unknown')}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load priors from {save_path}: {e}")
            self.logger.info("Using default initialization instead.")

    def _save_priors_to_disk(self, force: bool = False):
        if not self.enable_persistence:
            return
        updates_since_save = self._update_counter - self._last_save_counter
        if not force and updates_since_save < self.auto_save_frequency:
            return
        try:
            save_dir = Path(self.persistence_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            prior_dict = self.export_priors()
            save_path = save_dir / "priors_latest.json"
            if save_path.exists():
                with open(save_path, "r") as f:
                    old_data = json.load(f)
                old_metadata = old_data.get("metadata", {})
                total_updates = old_metadata.get("total_updates", 0)
                backtest_count = old_metadata.get("backtest_count", 0)
            else:
                total_updates = 0
                backtest_count = 0
            save_data = {
                "priors": prior_dict,
                "metadata": {
                    "total_updates": total_updates + self._update_counter,
                    "backtest_count": backtest_count + 1,
                    "last_save_time": pd.Timestamp.now().isoformat(),
                    "learning_rate": self.base_learning_rate,
                },
            }
            with open(save_path, "w") as f:
                json.dump(save_data, f, indent=2)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            versioned_path = save_dir / f"priors_v{timestamp}.json"
            with open(versioned_path, "w") as f:
                json.dump(save_data, f, indent=2)
            self._last_save_counter = self._update_counter
            self.logger.info(f" Saved learned priors to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save priors: {e}")

    def finalize_learning(self):
        self.logger.info(
            f"Finalizing backtest learning. Total updates this run: {self._update_counter}"
        )
        self._save_priors_to_disk(force=True)

    def calculate_expected_value(
        self,
        signal_score: float,
        signal_confidence: float,
        factor_source: str,
        regime: str,
        factor_ic: float = 0.0,
        components: Optional[Dict] = None,
    ) -> Tuple[float, Dict]:
        if not self.enabled:
            return self._simple_ev(signal_score, signal_confidence), {}

        prior = self._get_prior(factor_source, regime)
        win_rate, win_rate_lower, win_rate_upper = self._calculate_posterior_win_rate(
            prior, signal_confidence, factor_ic
        )

        raw_expected_gain = self._calculate_expected_gain(
            prior, signal_score, signal_confidence, factor_ic
        )
        gain_stdev = np.sqrt(prior.variance_gain)
        gain_volatility_penalty = self.gain_volatility_penalty * gain_stdev

        expected_gain = raw_expected_gain - gain_volatility_penalty
        expected_gain = max(0, expected_gain)

        expected_loss = self._calculate_expected_loss(
            prior, signal_score, signal_confidence
        )

        raw_ev = (win_rate * expected_gain) - ((1.0 - win_rate) * abs(expected_loss))

        ev = raw_ev
        if self.risk_aversion_factor != 1.0:
            total_variance = (win_rate * prior.variance_gain) + (
                (1 - win_rate) * prior.variance_loss
            )
            risk_penalty = self.risk_aversion_factor * np.sqrt(total_variance)
            ev -= risk_penalty

        diagnostics = {
            "win_rate": win_rate,
            "win_rate_lower": win_rate_lower,
            "win_rate_upper": win_rate_upper,
            "raw_expected_gain": raw_expected_gain,
            "gain_vol_penalty": gain_volatility_penalty,
            "final_expected_gain": expected_gain,
            "expected_loss": expected_loss,
            "raw_ev": raw_ev,
            "risk_adjusted_ev": ev,
            "prior_alpha": prior.alpha,
            "prior_beta": prior.beta,
            "prior_strength": prior.pseudocount,
            "observations": prior.total_signals,
        }
        return ev, diagnostics

    def _get_prior(self, factor_source: str, regime: str) -> BayesianPrior:
        regime_lower = regime.lower()
        if factor_source in self.priors and regime_lower in self.priors[factor_source]:
            return self.priors[factor_source][regime_lower]
        if factor_source in self.priors and "normal" in self.priors[factor_source]:
            return self.priors[factor_source]["normal"]
        return BayesianPrior()

    def _calculate_posterior_win_rate(
        self, prior: BayesianPrior, signal_confidence: float, factor_ic: float
    ) -> Tuple[float, float, float]:
        alpha, beta = prior.alpha, prior.beta
        ic_adjustment = self.ic_influence_on_prior * factor_ic * self.prior_strength
        if ic_adjustment > 0:
            alpha += ic_adjustment
        else:
            beta += abs(ic_adjustment)

        alpha += signal_confidence * self.confidence_to_alpha_scaler
        posterior_alpha, posterior_beta = alpha, beta
        mean_win_rate = posterior_alpha / (posterior_alpha + posterior_beta)

        if self.use_credible_intervals:
            lower_p = (1 - self.credible_interval_level) / 2
            upper_p = 1 - lower_p
            lower_bound = stats.beta.ppf(lower_p, posterior_alpha, posterior_beta)
            upper_bound = stats.beta.ppf(upper_p, posterior_alpha, posterior_beta)
        else:
            lower_bound, upper_bound = mean_win_rate, mean_win_rate

        mean_win_rate = float(np.clip(mean_win_rate, 0.35, 0.80))
        lower_bound = float(np.clip(lower_bound, 0.30, mean_win_rate))
        upper_bound = float(np.clip(upper_bound, mean_win_rate, 0.85))
        return mean_win_rate, lower_bound, upper_bound

    def _calculate_expected_gain(
        self,
        prior: BayesianPrior,
        signal_score: float,
        signal_confidence: float,
        factor_ic: float,
    ) -> float:
        prior_mean, prior_weight = prior.mean_gain, prior.pseudocount
        signal_implied_gain = (
            abs(signal_score) * signal_confidence * self.signal_to_gain_scaler
        )
        signal_implied_gain *= 1.0 + factor_ic
        signal_weight = (
            abs(signal_score) * signal_confidence * self.signal_to_weight_scaler
        )

        total_weight = prior_weight + signal_weight
        posterior_mean = (
            (prior_weight * prior_mean) + (signal_weight * signal_implied_gain)
        ) / total_weight
        return max(posterior_mean, 0.0)

    def _calculate_expected_loss(
        self, prior: BayesianPrior, signal_score: float, signal_confidence: float
    ) -> float:
        if not self.dynamic_stop_loss:
            return self.base_stop_loss

        prior_loss = abs(prior.mean_loss)
        confidence_factor = 0.7 + (signal_confidence * 0.6)
        score_factor = 0.8 + (abs(signal_score) * 0.4)
        dynamic_stop = self.base_stop_loss * confidence_factor * score_factor
        expected_loss = 0.6 * dynamic_stop + 0.4 * prior_loss
        return float(np.clip(expected_loss, 0.01, 0.05))

    def update_with_outcome(self, outcome: SignalOutcome):
        prior = self._get_prior(outcome.factor_source, outcome.regime)
        self.outcome_history.append(outcome)
        if prior.total_signals < self.min_observations_for_update:
            prior.total_signals += 1
            return

        # MODIFIED: Adaptive Learning Rate Calculation based on prediction surprise.
        current_learning_rate = self.base_learning_rate
        if self.use_dynamic_learning_rate:
            # A prediction error occurs if the outcome contradicts a confident prediction.
            is_prediction_error = (
                outcome.predicted_ev > 0 and not outcome.was_winner
            ) or (outcome.predicted_ev < 0 and outcome.was_winner)
            if is_prediction_error:
                # Greatly increase learning rate for surprising outcomes.
                confidence_factor = 1.0 + outcome.entry_confidence
                adjusted_lr = (
                    self.base_learning_rate
                    * self.error_learning_rate_multiplier
                    * confidence_factor
                )
                current_learning_rate = min(adjusted_lr, self.max_learning_rate)
            elif outcome.was_winner and not outcome.is_strong_failure:
                # Slightly decrease learning rate for expected outcomes.
                current_learning_rate *= 0.5

            if outcome.is_strong_failure:
                # Also learn faster from strong, unexpected failures.
                current_learning_rate = max(
                    current_learning_rate, self.base_learning_rate * 1.5
                )

        if self.use_ewma_learning:
            decay_factor = 1.0 - self.knowledge_decay_rate
            prior.alpha *= decay_factor
            prior.beta *= decay_factor
            total_strength = prior.alpha + prior.beta
            current_win_rate = prior.alpha / total_strength
            observation = 1.0 if outcome.was_winner else 0.0
            new_win_rate = (
                1 - current_learning_rate
            ) * current_win_rate + current_learning_rate * observation
            prior.alpha = new_win_rate * total_strength
            prior.beta = (1.0 - new_win_rate) * total_strength
            if outcome.was_winner:
                prior.mean_gain = (
                    1 - current_learning_rate
                ) * prior.mean_gain + current_learning_rate * outcome.actual_return
                squared_error_gain = (outcome.actual_return - prior.mean_gain) ** 2
                prior.variance_gain = (
                    1 - current_learning_rate
                ) * prior.variance_gain + current_learning_rate * squared_error_gain
            else:
                prior.mean_loss = (
                    1 - current_learning_rate
                ) * prior.mean_loss + current_learning_rate * outcome.actual_return
                squared_error_loss = (outcome.actual_return - prior.mean_loss) ** 2
                prior.variance_loss = (
                    1 - current_learning_rate
                ) * prior.variance_loss + current_learning_rate * squared_error_loss
        else:  # Original conjugate update logic (kept for reference)
            if outcome.was_winner:
                prior.alpha += current_learning_rate
            else:
                prior.beta += current_learning_rate

        prior.total_signals += 1
        prior.last_update = outcome.timestamp
        self._update_counter += 1
        if self.enable_persistence:
            self._save_priors_to_disk()

        self.logger.debug(
            f"Updated prior for {outcome.factor_source}/{outcome.regime}: "
            f"win_rate={prior.alpha / (prior.alpha + prior.beta):.3f}, "
            f"mean_gain={prior.mean_gain:.4f}, lr={current_learning_rate:.3f}"
        )

    def _simple_ev(self, score: float, confidence: float) -> float:
        win_rate = 0.45 + (confidence * 0.25)
        expected_gain = abs(score) * confidence * 0.05
        expected_loss = self.base_stop_loss
        return (win_rate * expected_gain) - ((1.0 - win_rate) * expected_loss)

    def get_prior_statistics(self) -> pd.DataFrame:
        rows = []
        for factor, regimes in self.priors.items():
            for regime, prior in regimes.items():
                win_rate = prior.alpha / (prior.alpha + prior.beta)
                rows.append(
                    {
                        "factor": factor,
                        "regime": regime,
                        "win_rate": win_rate,
                        "mean_gain": prior.mean_gain,
                        "mean_loss": prior.mean_loss,
                        "variance_gain": prior.variance_gain,
                        "variance_loss": prior.variance_loss,
                        "observations": prior.total_signals,
                        "last_update": prior.last_update,
                    }
                )
        return pd.DataFrame(rows)

    def import_priors(self, priors_data: Dict):
        self.priors = defaultdict(dict)
        for factor, regimes in priors_data.items():
            for regime, data in regimes.items():
                last_update_ts = (
                    pd.Timestamp(data["last_update"])
                    if data.get("last_update")
                    else None
                )
                self.priors[factor][regime] = BayesianPrior(
                    alpha=data.get("alpha", 15.0),
                    beta=data.get("beta", 10.0),
                    mean_gain=data.get("mean_gain", 0.04),
                    variance_gain=data.get("variance_gain", 0.015),
                    mean_loss=data.get("mean_loss", -0.015),
                    variance_loss=data.get("variance_loss", 0.005),
                    pseudocount=data.get("pseudocount", 12.0),
                    total_signals=data.get("total_signals", 0),
                    last_update=last_update_ts,
                )
        self.logger.info(
            f"Successfully imported priors for {len(priors_data)} factors."
        )

    def export_priors(self) -> Dict:
        export_dict = {}
        for factor, regimes in self.priors.items():
            export_dict[factor] = {}
            for regime, prior in regimes.items():
                export_dict[factor][regime] = {
                    "alpha": prior.alpha,
                    "beta": prior.beta,
                    "mean_gain": prior.mean_gain,
                    "variance_gain": prior.variance_gain,
                    "mean_loss": prior.mean_loss,
                    "variance_loss": prior.variance_loss,
                    "pseudocount": prior.pseudocount,
                    "total_signals": prior.total_signals,
                    "last_update": (
                        prior.last_update.isoformat() if prior.last_update else None
                    ),
                }
        return export_dict
