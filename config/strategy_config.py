import importlib
import logging

# ============================================================================
# STRATEGY SELECTION
# ============================================================================
CURRENT_STRATEGY = "HYBRID_DUAL_ALPHA"

# Walk-forward optimization settings
WFO_ENABLED = False
WFO_TRAIN_PERIOD = 5  # years
WFO_TEST_PERIOD = 1  # year

# ============================================================================
# HYBRID DUAL ALPHA STRATEGY - SINGLE SOURCE OF TRUTH
# ============================================================================

HYBRID_DUAL_PARAMS = {
    # =================== REBALANCE & SIZING LOGIC ===================
    "rebalance_frequency": "weekly",
    "tactical_mode": {
        "enabled": True,
        "strategic_rebalance_day": 0,
        "tactical_entry_enabled": False,
        "tactical_min_ev_threshold": 0.015,
        "tactical_min_confidence_threshold": 0.85,
        "tactical_cash_deployment_pct": 0.40,
    },
    "min_ev_percentile_filter": 0.23,
    "sizing_ev_to_confidence_ratio": 0.71,
    "bull_market_leverage": 1.07,
    # =================== ALPHA ENGINE & FACTORS ===================
    "alpha_engine": {
        "combination_mode": "smart_weighted",
        "normalization_enabled": True,
        "normalization_method": "rank",
        "winsorize_percentile": 0.02,
        "neutralize_beta": True,
        "base_weights": {
            "reversal": 0.45,
            "liquidity": 0.45,
            "price": 0.10,
        },
        "momentum_confirmer_sensitivity": 0.15,
        "deviation_sensitivity": 0.5,
        "ic_weight_sensitivity": 1.19,
    },
    "ic_monitoring": {
        "enabled": True,
        "lookback_period": 180,
        "forward_return_period": 20,
        "smoothing_alpha": 0.20,
        "use_dynamic_weights": True,
        "weight_sensitivity": 1.18,
    },
    "bayesian_ev": {
        "enabled": True,
        "use_ewma_learning": True,
        "use_dynamic_learning_rate": True,
        "prior_strength": 50.0,
        "learning_rate": 0.02,
        "error_learning_rate_multiplier": 2.5,
        "max_learning_rate": 0.4,
        "knowledge_decay_rate": 0.005,
        "gain_volatility_penalty": 0.28,
        "risk_aversion_factor": 0.78,
        "confidence_to_alpha_scaler": 5.0,
        "signal_to_gain_scaler": 0.16,
        "signal_to_weight_scaler": 16.0,
        "enable_persistence": False,
        "persistence_dir": "./data/bayesian_priors",
    },
    "signal_evaluation_period": 20,
    # --- Factor Parameters  ---
    "reversal_alpha_params": {
        # Dynamic lookbacks are handled in code, fixed values removed
        "extreme_loser_threshold": -0.12,
        "volume_surge_threshold": 1.15,
        "min_confidence": 0.30,
        "min_avg_volume": 300_000,
        "min_dollar_volume": 15_000_000,
        "regime_lookback_weights": {
            "CRISIS": {"short": 0.80, "long": 0.20},
            "VOLATILE": {"short": 0.75, "long": 0.25},
            "BEAR": {"short": 0.65, "long": 0.35},
            "NORMAL": {"short": 0.50, "long": 0.50},
            "BULL": {"short": 0.40, "long": 0.60},
            "STRONG_BULL": {"short": 0.30, "long": 0.70},
        },
        "regime_config": {
            "CRISIS": {"strength": 2.0, "threshold_mult": 1.3},
            "BEAR": {"strength": 1.5, "threshold_mult": 1.1},
            "NORMAL": {"strength": 0.9, "threshold_mult": 1.15},
            "BULL": {"strength": 0.5, "threshold_mult": 1.4},
            "STRONG_BULL": {"strength": 0.3, "threshold_mult": 1.8},
        },
    },
    "price_alpha_params": {
        "enable_high_certainty_mode": True,
        "min_confidence": 0.35,
        "min_score_threshold": 0.40,
        # RSI settings
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "rsi_short_period": 5,
        "rsi_long_period": 28,
        "rsi_short_oversold": 35,
        "rsi_long_oversold": 45,
        "rsi_short_overbought": 65,
        "rsi_long_overbought": 55,
        # Bollinger Bands settings
        "bb_period": 20,
        "bb_std": 2.0,
        "bb_deviation_threshold": 1.0,
        # Confidence & Scoring
        "base_confidence_on_signal": 0.65,
        "volume_confirmation_multiplier": 1.5,
        "confidence_volume_bonus": 0.15,
        "alignment_short_weight": 0.5,
        "alignment_long_weight": 0.3,
        "quality_denominator": 50.0,
        "bb_penetration_mult": 0.15,
        "base_oversold_score": 0.5,
        "base_overbought_score": -0.5,
        "quality_mult": 0.3,
    },
    "liquidity_alpha_params": {
        "volume_lookback": 20,
        "amihud_lookback": 20,
        "min_confidence": 0.30,
        "score_threshold": 0.12,
        "min_avg_dollar_volume": 1_000_000,
        "volume_ratio_filter_threshold": 2.5,
        "regime_multipliers": {
            "crisis": 1.5,
            "bear": 1.1,
            "normal": 1.0,
            "bull": 0.95,
            "strong_bull": 0.75,
        },
        "high_liquidity_percentile": 0.90,
        "low_liquidity_percentile": 0.10,
        "confidence_base": 0.35,
        "confidence_extremeness_mult": 0.80,
        "confidence_extreme_threshold": 0.85,
        "confidence_low_threshold": 0.15,
        "confidence_extreme_bonus": 0.15,
    },
    "momentum_alpha_params": {
        # Dynamic lookbacks are handled in code, fixed values removed
        "min_absolute_momentum": 0.02,
        "min_confidence": 0.25,
        "use_volatility_control": True,
        "max_volatility_percentile": 0.75,
        "use_quality_filter": True,
        "drawdown_penalty_factor": 10.0,
        "high_confidence_threshold": 0.35,
        "medium_confidence_threshold": 0.25,
        "high_confidence_value": 0.75,
        "medium_confidence_value": 0.60,
        "default_confidence_value": 0.40,
    },
    # =================== MARKET STATE & RISK SCALING ===================
    "market_detector_params": {
        "garch_lookback_days": 1000,
        "garch_search_reps": 10,
        "garch_high_vol_threshold": 0.7,
        "prototype_enabled": True,
        "transition_lambda": 1.24,
    },
}

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

OPTIMIZATION_PARAMS = {
    "hybrid_dual_alpha": {
        "signal_to_gain_scaler": (0.10, 0.25),
        "transition_lambda": (0.75, 1.75),
        "min_ev_percentile_filter": (0.15, 0.30),
        "tactical_min_ev_threshold": (0.01, 0.03),
        "tactical_min_confidence_threshold": (0.75, 0.90),
        "atr_multiplier": (2.0, 3.5),
        "lpr_profit_threshold": (0.30, 0.60),
        "cvar_target": (0.015, 0.03),
        "max_drawdown_threshold": (0.20, 0.30),
        "signal_evaluation_period": (5, 30),
    }
}

# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

STRATEGY_REGISTRY = {
    "HYBRID_DUAL_ALPHA": "strategy.hybrid_dual_alpha_strategy:HybridDualAlphaStrategy",
}


def get_strategy_class(strategy_name: str):
    """Get strategy class from registry

    Args:
      strategy_name: str:

    Returns:

    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    module_path, class_name = STRATEGY_REGISTRY[strategy_name].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_current_strategy_params():
    """Get parameters for current strategy"""
    if CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        return HYBRID_DUAL_PARAMS
    return {}


def validate_config():
    """Validate strategy configuration"""
    errors = []

    if CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        params = get_current_strategy_params()
        engine_params = params.get("alpha_engine", {})
        base_weights = engine_params.get("base_weights", {})

        if base_weights:
            weight_sum = sum(base_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                errors.append(
                    f"Alpha engine base_weights must sum to 1.0, but got {weight_sum}"
                )
        else:
            errors.append("Alpha engine 'base_weights' not defined.")

    return errors, []


# Auto-validate on import
if __name__ != "__main__":
    errors, warnings = validate_config()
    if warnings:
        for warning in warnings:
            logging.warning(f"Config warning: {warning}")
    if errors:
        error_message = "Strategy config validation failed:\n  - " + "\n  - ".join(
            errors
        )
        raise ValueError(error_message)
