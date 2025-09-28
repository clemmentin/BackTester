# strategy_config.py - Strategy configuration module
import importlib
import logging

from . import general_config as g_cfg

# ============================================================================
# STRATEGY SELECTION
# ============================================================================
CURRENT_STRATEGY = "HYBRID_DUAL_ALPHA"  # Options: "ALPHA_UNIFIED" | "HYBRID_DUAL_ALPHA"

# Walk-Forward Optimization Settings
WFO_ENABLED = False
WFO_TRAIN_PERIOD = 5  # Years
WFO_TEST_PERIOD = 1  # Years

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

# Alpha Unified Strategy Parameters
ALPHA_UNIFIED_PARAMS = {
    # Core alpha engine weights (must sum to 1.0)
    "momentum_weight": 0.50,
    "technical_weight": 0.30,
    "composite_weight": 0.20,
    # Signal thresholds
    "min_alpha_score": 0.10,
    "min_confidence": 0.20,
    # Momentum calculation windows
    "momentum_short": 5,
    "momentum_medium": 15,
    "momentum_long": 45,
    "min_momentum_threshold": -0.15,
    "volatility_scaling": True,
    # Technical indicators
    "mean_reversion_window": 60,
    "volume_lookback": 10,
    "efficiency_window": 20,
    "trend_window": 50,
    # Market regime detection
    "regime_lookback_short": 15,
    "regime_lookback_medium": 40,
    "regime_lookback_long": 200,
    "bull_threshold": 0.05,
    "crisis_threshold": -0.15,
    "enable_early_warning": True,
}

# Hybrid Dual Alpha Strategy Parameters
HYBRID_DUAL_PARAMS = {
    # === Allocation Controller ===
    "base_conservative_allocation": 0.30,
    "base_aggressive_allocation": 0.70,
    "allocation_adaptation_enabled": True,
    "target_max_utilization": 0.95,
    "min_target_weight": 0.02,
    "weight_to_score_multiplier": 10,
    "position_change_cooldown": 3,
    # Dynamic allocation map by market regime
    "dynamic_allocation_map": {
        "crisis": {"conservative": 0.90, "aggressive": 0.10},
        "bear": {"conservative": 0.75, "aggressive": 0.25},
        "volatile": {"conservative": 0.60, "aggressive": 0.40},
        "recovery": {"conservative": 0.40, "aggressive": 0.60},
        "normal": {"conservative": 0.30, "aggressive": 0.70},
        "bull": {"conservative": 0.15, "aggressive": 0.85},
        "strong_bull": {"conservative": 0.05, "aggressive": 0.95},
    },
    # === Shared Alpha Engine ===
    "momentum_weight": 0.40,
    "technical_weight": 0.35,
    "composite_weight": 0.25,
    "min_alpha_score": 0.30,
    "min_confidence": 0.40,
    # === Conservative Module ===
    "conservative_alpha_weight": 0.25,
    "conservative_min_confidence": 0.40,
    "conservative_max_positions": 3,
    "conservative_momentum_windows": [20, 60, 120],
    "conservative_position_weights": [0.45, 0.35, 0.20],
    "conservative_momentum_weights": [0.2, 0.3, 0.5],
    "conservative_warning_threshold": 2,
    "conservative_warning_multiplier": 0.5,
    # === Aggressive Module ===
    "aggressive_alpha_weight": 0.40,
    "aggressive_min_confidence": 0.25,
    "aggressive_max_positions": 12,
    "aggressive_max_single_position": 0.15,
    "aggressive_momentum_windows": [5, 10, 20],
    "aggressive_momentum_weights": [0.45, 0.3, 0.25],
    "aggressive_sizing_blend_factor": 0.5,
    # Aggressive regime multipliers
    "aggressive_regime_multipliers": {
        "strong_bull": 1.5,
        "bull": 1.3,
        "recovery": 1.1,
        "normal": 1.0,
        "volatile": 0.7,
        "bear": 0.6,
        "crisis": 0.4,
    },
    # Aggressive risk multipliers
    "aggressive_risk_multipliers": {
        "low": 1.1,
        "medium": 1.0,
        "high": 0.7,
        "extreme": 0.4,
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

        'base_conservative_allocation': (0.15, 0.65),
        'min_target_weight': (0.005, 0.04),
        'weight_to_score_multiplier': (5, 18),

        'min_alpha_score': (0.15, 0.55),
        'min_confidence': (0.25, 0.65),

        'conservative_alpha_weight': (0.12, 0.38),
        'conservative_min_confidence': (0.25, 0.60),
        'conservative_max_positions': (2, 6),
        'conservative_warning_threshold': (1, 4),

        'aggressive_alpha_weight': (0.25, 0.55),
        'aggressive_min_confidence': (0.15, 0.40),
        'aggressive_max_positions': (6, 18),
        'aggressive_max_single_position': (0.06, 0.20),
        'aggressive_sizing_blend_factor': (0.25, 0.75),

        'momentum_weight': (0.30, 0.55),
        'technical_weight': (0.25, 0.50),
        'composite_weight': (0.15, 0.40),

        'target_max_utilization': (0.85, 0.98),
        'position_change_cooldown': (1, 7),
    }
}

# ============================================================================
# STRATEGY REGISTRY
# ============================================================================
STRATEGY_REGISTRY = {
    "ALPHA_UNIFIED": "strategy.alpha_strategy:AlphaUnifiedStrategy",
    "HYBRID_DUAL_ALPHA": "strategy.hybrid_dual_alpha_strategy:HybridDualAlphaStrategy",
}


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================
def create_strategy(events_queue, symbol_list, **kwargs):
    """Factory function for creating strategy instances."""
    strategy_name = kwargs.pop("name", CURRENT_STRATEGY)

    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{strategy_name}' not found in STRATEGY_REGISTRY")

    # Dynamic import
    module_path, class_name = STRATEGY_REGISTRY[strategy_name].split(":")
    module = importlib.import_module(module_path)
    StrategyClass = getattr(module, class_name)

    # Select parameters
    if strategy_name == "ALPHA_UNIFIED":
        params = ALPHA_UNIFIED_PARAMS.copy()
    elif strategy_name == "HYBRID_DUAL_ALPHA":
        params = HYBRID_DUAL_PARAMS.copy()
    else:
        params = {}

    # Merge with overrides and add global symbols
    params.update(kwargs)
    params["risk_on_symbols"] = g_cfg.RISK_ON_SYMBOLS
    params["risk_off_symbols"] = g_cfg.RISK_OFF_SYMBOLS

    logging.info(f"Creating strategy instance: {strategy_name}")
    return StrategyClass(events_queue, symbol_list, **params)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_current_strategy_params():
    """Get parameters for current strategy."""
    if CURRENT_STRATEGY == "ALPHA_UNIFIED":
        return ALPHA_UNIFIED_PARAMS
    elif CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        return HYBRID_DUAL_PARAMS
    return {}


def get_optimization_params():
    """Get optimization parameters for current strategy."""
    return OPTIMIZATION_PARAMS.get(CURRENT_STRATEGY.lower(), {})


def update_strategy_param(param_name, new_value):
    """Update a specific strategy parameter."""
    if param_name in ALPHA_UNIFIED_PARAMS:
        ALPHA_UNIFIED_PARAMS[param_name] = new_value
        return True
    return False


def print_config_summary():
    """Print strategy configuration summary."""
    print("=" * 60)
    print("Strategy Configuration Summary")
    print("=" * 60)
    print(f"Current Strategy: {CURRENT_STRATEGY}")
    print(f"WFO Status: {'Enabled' if WFO_ENABLED else 'Disabled'}")

    if CURRENT_STRATEGY == "ALPHA_UNIFIED":
        p = ALPHA_UNIFIED_PARAMS
        print(f"\nAlpha Weights:")
        print(f"  Momentum: {p['momentum_weight']:.1%}")
        print(f"  Technical: {p['technical_weight']:.1%}")
        print(f"  Composite: {p['composite_weight']:.1%}")
    elif CURRENT_STRATEGY == "HYBRID_DUAL_ALPHA":
        p = HYBRID_DUAL_PARAMS
        print(f"\nBase Allocation:")
        print(f"  Conservative: {p['base_conservative_allocation']:.1%}")
        print(f"  Aggressive: {p['base_aggressive_allocation']:.1%}")

    print("=" * 60)


def validate_config():
    """Validate strategy configuration."""
    errors = []
    warnings = []

    # Validate ALPHA_UNIFIED weights
    weight_sum = (
        ALPHA_UNIFIED_PARAMS["momentum_weight"]
        + ALPHA_UNIFIED_PARAMS["technical_weight"]
        + ALPHA_UNIFIED_PARAMS["composite_weight"]
    )
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"Alpha weights must sum to 1.0, got {weight_sum}")

    # Validate momentum windows
    if not (
        ALPHA_UNIFIED_PARAMS["momentum_short"]
        < ALPHA_UNIFIED_PARAMS["momentum_medium"]
        < ALPHA_UNIFIED_PARAMS["momentum_long"]
    ):
        errors.append("Momentum windows must be in ascending order")

    return errors, warnings


# ============================================================================
# AUTO-VALIDATION
# ============================================================================
if __name__ == "__main__":
    errors, warnings = validate_config()

    if errors:
        print("Strategy Configuration Errors:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("Strategy Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    logger = logging.getLogger(__name__)
    logger.info(f"Strategy configuration loaded: {CURRENT_STRATEGY}")
    logger.info(f"WFO status: {'Enabled' if WFO_ENABLED else 'Disabled'}")
