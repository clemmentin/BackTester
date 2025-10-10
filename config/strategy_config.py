import importlib
import logging

# ============================================================================
# STRATEGY SELECTION
# ============================================================================
CURRENT_STRATEGY = "HYBRID_DUAL_ALPHA"
WFO_ENABLED = False
WFO_TRAIN_PERIOD = 5
WFO_TEST_PERIOD = 1

# ============================================================================
# ALPHA UNIFIED STRATEGY (Legacy)
# ============================================================================
ALPHA_UNIFIED_PARAMS = {
    "momentum_weight": 0.50,
    "technical_weight": 0.30,
    "composite_weight": 0.20,
    "min_alpha_score": 0.10,
    "min_confidence": 0.20,
    "momentum_short": 5,
    "momentum_medium": 15,
    "momentum_long": 45,
    "min_momentum_threshold": -0.15,
    "volatility_scaling": True,
    "mean_reversion_window": 60,
    "volume_lookback": 10,
    "efficiency_window": 20,
    "trend_window": 50,
    "regime_lookback_short": 15,
    "regime_lookback_medium": 40,
    "regime_lookback_long": 200,
    "bull_threshold": 0.05,
    "crisis_threshold": -0.15,
    "enable_early_warning": True,
}

# ============================================================================
# HYBRID DUAL ALPHA STRATEGY (3-Factor Model)
# ============================================================================

HYBRID_DUAL_PARAMS = {
    # === INTELLIGENT ALPHA ENGINE (4-Factor Model) ===
    "alpha_engine": {
        # Base weights for the four alpha factors. Must sum to 1.0.
        "reversal_weight": 0.25,
        "price_weight": 0.25,
        "liquidity_weight": 0.20,
        "momentum_weight": 0.30,
        # Method for combining signals from different factors.
        "combination_mode": "smart_weighted",
        # Threshold to identify conflicting signals between factors.
        "conflict_threshold": 0.6,
        # Min/max constraints for dynamically adjusted factor weights.
        "min_factor_weight": 0.01,
        "max_factor_weight": 0.99,
        # Configuration for the initial signal quality filter.
        "signal_quality": {
            "enabled": True,
            "min_score": 0.43,
            "min_confidence": 0.43,
        },
        # Configuration for cross-sectional normalization of alpha scores.
        "normalization_enabled": True,
        "normalization_method": "rank",  # 'rank' is robust to outliers.
    },
    # === IC-BASED DYNAMIC WEIGHTING ===
    # Dynamically adjusts alpha factor weights based on recent performance (Information Coefficient).
    "ic_monitoring": {
        "enabled": True,
        "lookback_period": 180,  # How many days of history to use for IC calculation.
        "forward_return_period": 20,  # The future period to correlate signals against (e.g., 20 days).
        "smoothing_alpha": 0.20,  # Smoothing factor for the IC scores (higher is more reactive).
        "ic_threshold": 0.02,  # Minimum IC for a factor to be considered effective.
        "ic_strong_threshold": 0.10,  # IC level to consider a factor's performance as strong.
    },
    # === REVERSAL ALPHA PARAMETERS ===
    # Generates signals for assets that have performed poorly and are expected to bounce back.
    "reversal_alpha_params": {
        "reversal_lookback": 35,  # Lookback period to identify 'loser' stocks.
        "extreme_loser_threshold": -0.12,  # Return threshold to be considered an extreme loser.
        "volume_surge_threshold": 1.2,  # Volume multiplier indicating capitulation.
        "min_avg_volume": 300_000,  # Minimum average daily shares traded.
        "min_dollar_volume": 15_000_000,  # Minimum average daily dollar volume.
        "min_confidence": 0.30,  # Minimum confidence required for a reversal signal.
    },
    # === PRICE ALPHA PARAMETERS ===
    # Generates mean-reversion signals based on technical indicators like RSI and Bollinger Bands.
    "price_alpha_params": {
        "rsi_period": 21,
        "rsi_oversold": 24,  # RSI level below which a stock is considered oversold (buy signal).
        "rsi_overbought": 76,  # RSI level above which a stock is considered overbought (sell signal).
        "bb_period": 20,  # Lookback period for Bollinger Bands.
        "bb_std": 2.0,  # Number of standard deviations for Bollinger Bands.
        "min_confidence": 0.30,
        "min_score_threshold": 0.15,
    },
    # === LIQUIDITY ALPHA PARAMETERS ===
    # Generates signals based on volume and liquidity anomalies.
    "liquidity_alpha_params": {
        "volume_lookback": 20,
        "volume_spike_threshold": 1.5,  # Volume increase factor to be considered a significant spike.
        "use_amihud": True,  # Enable Amihud illiquidity measure.
        "amihud_lookback": 20,
        "use_obv": True,  # Enable On-Balance Volume indicator.
        "obv_signal_period": 10,
        "min_avg_dollar_volume": 1_000_000,  # Minimum dollar volume to be included in calculations.
        "min_confidence": 0.30,
    },
    # === MOMENTUM ALPHA PARAMETERS ===
    # Generates signals for assets that have performed well and are expected to continue.
    "momentum_alpha_params": {
        "formation_period": 130,  # Lookback period to measure momentum (approx. 6 months).
        "skip_period": 21,  # Period to skip after formation to avoid short-term reversals.
        "min_absolute_momentum": 0.05,  # Minimum return over the formation period to be considered.
        "min_confidence": 0.30,
        "use_volatility_control": True,  # If true, filter out the most volatile stocks.
        "max_volatility_percentile": 0.80,  # Exclude the top 20% most volatile stocks.
        "use_quality_filter": True,  # If true, penalize momentum signals with high drawdowns.
        "high_dd_penalty": 0.25,  # Penalty applied for high drawdown.
        "max_dd_threshold": 0.20,  # Drawdown level above which the penalty is applied.
    },
    # === MARKET DETECTOR ===
    # Detects the overall market regime (e.g., Bull, Bear, Crisis) to adapt the strategy.
    "market_detector_params": {
        "lookback_short": 20,
        "lookback_medium": 50,
        "lookback_long": 200,
        "bull_threshold": 0.03,
        "crisis_threshold": -0.15,
        "enable_macro_enhancement": True,  # Use macroeconomic data (e.g., VIX, yield curve) to refine detection.
        "macro_indicators_subset": [
            "T10Y2Y",
            "VIXCLS",
            "UNRATE",
        ],  # Key macro indicators to use.
        "macro_thresholds": {
            "yield_curve_inversion": -0.20,  # Spread below which recession risk increases.
            "vix_crisis_level": 35,  # VIX level indicating market crisis.
            "unemployment_spike": 0.50,  # 3-month increase in unemployment rate indicating economic weakness.
        },
    },
    # === POSITION SIZING ===
    # Defines how portfolio weights are calculated and constrained.
    "position_sizing": {
        "weighting_scheme": "stratified_score_weighted",
        "min_position_weight": 0.01,  # Minimum allowed weight for any single position.
        "min_trade_size_dollars": 4000,  # Minimum trade size in dollars to reduce small, inefficient trades.
        "rebalance_threshold": 0.10,  # Minimum weight change required to trigger a rebalance for an existing position.
        "consensus_multiplier": 1.2,  # Weight bonus for stocks selected by both Aggressive and Conservative modules.
    },
    # === CONSERVATIVE MODULE ===
    # Selects candidate stocks based on defensive characteristics.
    "conservative_module": {
        "conservative_min_confidence": 0.45,  # Higher confidence threshold for more reliable, defensive signals.
    },
    # === AGGRESSIVE MODULE ===
    # Selects candidate stocks based on growth and momentum characteristics.
    "aggressive_module": {
        "aggressive_min_confidence": 0.30,  # Lower confidence threshold to capture more growth opportunities.
    },
}

# ============================================================================
# OPTIMIZATION PARAMETERS (for WFO)
# ============================================================================
OPTIMIZATION_PARAMS = {
    "alpha_unified": {
        "momentum_weight": [0.30, 0.35, 0.40, 0.45],
        "technical_weight": [0.25, 0.30, 0.35, 0.40],
        "momentum_short": [8, 10, 12, 15],
        "momentum_medium": [15, 20, 25, 30],
        "momentum_long": [40, 50, 60, 80],
    },
    "hybrid_dual_alpha": {
        "reversal_lookback": (15, 40),
        "rsi_oversold": (20, 40),
        "rsi_overbought": (60, 80),
        "formation_period": (80, 180),
        "min_score_threshold": (0.3, 0.6),
        "min_confidence_threshold": (0.4, 0.7),
    },
}

# ============================================================================
# STRATEGY REGISTRY
# ============================================================================
STRATEGY_REGISTRY = {
    "ALPHA_UNIFIED": "strategy.alpha_strategy:AlphaUnifiedStrategy",
    "HYBRID_DUAL_ALPHA": "strategy.hybrid_dual_alpha_strategy:HybridDualAlphaStrategy",
}


def get_strategy_class(strategy_name: str):
    """Get strategy class from registry."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    module_path, class_name = STRATEGY_REGISTRY[strategy_name].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_current_strategy_params():
    """Get parameters for current strategy."""
    if CURRENT_STRATEGY == "ALPHA_UNIFIED":
        return ALPHA_UNIFIED_PARAMS
    elif CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        return HYBRID_DUAL_PARAMS
    return {}


def validate_config():
    """Validate strategy configuration."""
    errors = []
    warnings = []

    if CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        config = HYBRID_DUAL_PARAMS

        engine = config.get("alpha_engine", {})
        weight_sum = (
            engine.get("liquidity_weight", 0)
            + engine.get("reversal_weight", 0)
            + engine.get("price_weight", 0)
            + engine.get("momentum_weight", 0)
        )
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"Alpha engine weights must sum to 1.0, got {weight_sum}")

    return errors, warnings


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
