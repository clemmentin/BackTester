# trading_parameters.py - Execution, Portfolio and Risk configuration

# ============================================================================
# TRADING EXECUTION PARAMETERS
# ============================================================================
TRADING_PARAMS = {
    # Position sizing configuration
    "position_sizing": {
        "target_utilization": 0.95,
        "min_utilization": 0.80,
        "max_positions": 10
        ,
        "min_positions": 3,
        "position_buffer": 0.02,
        "rebalance_threshold": 0.10,
        "position_size_method": "signal_weighted",  # 'equal_weight' | 'signal_weighted' | 'risk_parity'
    },
    # Strategy execution rules
    "strategy_execution": {
        "min_holding_days": 5,
        "max_holding_days": 365,
        "signal_score_threshold": 1.0,
        "entry_stagger_days": 1,
        "exit_stagger_days": 1,
        "rebalance_frequency": "weekly",  # 'daily' | 'weekly' | 'biweekly' | 'monthly'
    },
    # Market regime detection thresholds
    "market_detection": {
        "crisis_drawdown_threshold": -0.30,
        "crisis_volatility_threshold": 0.35,
        "bear_drawdown_threshold": -0.15,
        "bear_volatility_threshold": 0.25,
        "trend_confirmation_multiplier": 1.02,
        "regime_transition_days": 3,
    },
    # Early warning system
    "early_warning": {
        "enabled": True,
        "market_breadth_threshold": 0.6,
        "momentum_divergence_threshold": 0.30,
        "volatility_spike_multiplier": 4.0,
        "warning_persistence_days": 5,
        "warning_decay_rate": 0.8,
    },
    # Dynamic position sizing
    "dynamic_sizing": {
        "enabled": True,
        "base_volatility_target": 0.15,
        "min_position_multiplier": 0.5,
        "max_position_multiplier": 2.0,
        "volatility_window": 20,
        "volatility_smoothing": 0.3,
    },
    # Transaction costs
    "transaction_costs": {
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
        "min_commission": 1.0,
        "market_impact_factor": 0.0001,
    },
}

# ============================================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================================
RISK_PARAMS = {
    # Portfolio exposure limits
    "portfolio_limits": {
        "max_sector_exposure": 0.30,
        "max_single_position": 0.12,
        "min_single_position": 0.02,
        "max_correlation": 0.80,
        "concentration_penalty": 0.05,
        "max_leverage": 2.0,
    },
    # Drawdown control
    "drawdown_control": {
        "max_drawdown_threshold": 0.25,
        "drawdown_reduction_rate": 0.9,
        "recovery_threshold": 0.90,
        "cooldown_days": 3,
        "emergency_drawdown_threshold": 0.30,
    },
    # Stop loss configuration
    "stop_loss": {
        "enabled": True,
        "fixed_stop_loss": 0.10,
        "trailing_stop_activation": 0.10,
        "trailing_stop_distance": 0.06,
        "time_stop_days": 60,
        "use_atr_stops": True,
        "atr_stop_multiplier": 2.0,
        "volatility_adjusted": True,
        "vol_adjustment_factor": 1.5,
        # Tiered stop levels
        "tiered_stop_levels": {
            "level_1": {"threshold": -0.05, "action": "reduce_half"},
            "level_2": {"threshold": -0.08, "action": "exit_all"},
        },
        # Profit protection levels
        "profit_protection": {
            "level_1": {"profit": 0.08, "protect": 0.05},
            "level_2": {"profit": 0.15, "protect": 0.10},
            "level_3": {"profit": 0.25, "protect": 0.18},
        },
        # Market regime adjustments
        "market_regime_adjustments": {
            "bull": 1.4,
            "normal": 1.0,
            "bear": 0.8,
            "crisis": 0.6,
        },
    },
}

# ============================================================================
# MARKET REGIME CONFIGURATION
# ============================================================================
MARKET_REGIME = {
    # Basic thresholds
    "vix_threshold": 20,
    "trend_lookback": 60,
    "volatility_regime_threshold": 1.3,
    "enable_regime_filter": True,
    # Crisis/Bear thresholds (linked to trading params)
    "crisis_drawdown": TRADING_PARAMS["market_detection"]["crisis_drawdown_threshold"],
    "crisis_volatility": TRADING_PARAMS["market_detection"][
        "crisis_volatility_threshold"
    ],
    "bear_drawdown": TRADING_PARAMS["market_detection"]["bear_drawdown_threshold"],
    "bear_volatility": TRADING_PARAMS["market_detection"]["bear_volatility_threshold"],
    # Regime indicator weights
    "regime_indicators": {
        "trend_weight": 0.4,
        "volatility_weight": 0.3,
        "breadth_weight": 0.2,
        "sentiment_weight": 0.1,
    },
    # Position multipliers by regime
    "regime_position_multipliers": {
        "BULL": 1.4,
        "BEAR": 0.8,
        "VOLATILE": 0.5,
        "SIDEWAYS": 0.9,
        "CRISIS": 0.5,
    },
}

# ============================================================================
# MONEY MANAGEMENT CONFIGURATION
# ============================================================================
MONEY_MANAGEMENT = {
    # Position sizing
    "kelly_fraction": 0.25,
    "max_allocation_per_trade": RISK_PARAMS["portfolio_limits"]["max_single_position"],
    "cash_reserve_pct": 1 - TRADING_PARAMS["position_sizing"]["target_utilization"],
    "rebalance_threshold": TRADING_PARAMS["position_sizing"]["rebalance_threshold"],
    # Risk limits
    "daily_var_limit": 0.03,
    "monthly_risk_budget": 0.08,
    # Profit taking rules
    "profit_taking_rules": {
        "partial_profit_at": RISK_PARAMS["stop_loss"]["trailing_stop_activation"],
        "partial_profit_pct": 0.33,
        "trailing_profit_activation": 0.20,
    },
    # Loss management rules
    "loss_management_rules": {
        "scale_in_enabled": False,
        "avg_down_enabled": False,
        "max_loss_per_day": 0.05,
        "max_loss_per_week": 0.10,
    },
}

# ============================================================================
# STRATEGY-SPECIFIC PARAMETERS (Optional overrides)
# ============================================================================
STRATEGY_SPECIFIC = {
    "momentum": {
        "window_short": 20,
        "window_medium": 60,
        "window_long": 120,
        "weight_short": 0.3,
        "weight_medium": 0.4,
        "weight_long": 0.3,
        "minimum_momentum_score": -0.15,
        "trend_filter_window": 200,
        "use_trend_filter": True,
    },
    "rotation": {
        "hold_top_n": 6,
        "minimum_assets": 3,
        "momentum_decay_factor": 0.95,
        "ranking_smoothing_days": 5,
    },
    "volatility": {
        "buy_threshold": 25.0,
        "sell_threshold": 45.0,
        "vol_ma_period": 20,
        "vol_std_multiplier": 1.5,
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_param(category, subcategory, param_name, default=None):
    """Get parameter value with fallback to default."""
    config_map = {
        "TRADING_PARAMS": TRADING_PARAMS,
        "RISK_PARAMS": RISK_PARAMS,
        "STRATEGY_SPECIFIC": STRATEGY_SPECIFIC,
        "MARKET_REGIME": MARKET_REGIME,
        "MONEY_MANAGEMENT": MONEY_MANAGEMENT,
    }

    try:
        return (
            config_map.get(category, {}).get(subcategory, {}).get(param_name, default)
        )
    except Exception:
        return default


def validate_parameters():
    """Validate parameter consistency across configurations."""
    errors = []
    warnings = []

    # Check utilization
    target_util = TRADING_PARAMS["position_sizing"]["target_utilization"]
    if target_util > 0.98:
        errors.append(
            f"Target utilization {target_util:.1%} too high - insufficient cash buffer"
        )

    # Check position counts
    min_pos = TRADING_PARAMS["position_sizing"]["min_positions"]
    max_pos = TRADING_PARAMS["position_sizing"]["max_positions"]
    if min_pos > max_pos:
        errors.append(f"Min positions ({min_pos}) > Max positions ({max_pos})")

    # Check drawdown thresholds
    max_dd = RISK_PARAMS["drawdown_control"]["max_drawdown_threshold"]
    emergency_dd = RISK_PARAMS["drawdown_control"]["emergency_drawdown_threshold"]
    if emergency_dd <= max_dd:
        errors.append(
            f"Emergency DD ({emergency_dd:.1%}) must be > Max DD ({max_dd:.1%})"
        )

    # Warnings
    if max_dd > 0.30:
        warnings.append(f"Max drawdown threshold {max_dd:.1%} may be too high")

    if TRADING_PARAMS["position_sizing"]["min_positions"] < 3:
        warnings.append("Less than 3 positions may reduce diversification")

    return errors, warnings


# ============================================================================
# AUTO-VALIDATION
# ============================================================================
if __name__ == "__main__":
    errors, warnings = validate_parameters()

    if errors:
        print("Trading Parameter Errors:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("Trading Parameter Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
