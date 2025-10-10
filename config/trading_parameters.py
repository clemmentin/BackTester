# ============================================================================
# TRADING EXECUTION PARAMETERS
# ============================================================================
TRADING_PARAMS = {
    # NEW: UNIFIED POSITION MANAGEMENT
    "position_management": {
        "min_total_positions": 5,
        "max_total_positions": 20,
        "regime_allocation_pct": {
            "strong_bull": {"conservative": 0.00, "aggressive": 1.00},
            "bull": {"conservative": 0.10, "aggressive": 0.90},
            "normal": {"conservative": 0.30, "aggressive": 0.80},
            "recovery": {"conservative": 0.50, "aggressive": 0.50},
            "volatile": {"conservative": 0.60, "aggressive": 0.40},
            "bear": {"conservative": 0.70, "aggressive": 0.30},
            "crisis": {"conservative": 0.90, "aggressive": 0.10},
        },
    },
    # POSITION SIZING & CONSTRUCTION
    "position_sizing": {
        "target_utilization": 0.98,
        "position_buffer": 0.02,
        "rebalance_threshold": 0.10,
        "weighting_scheme": "stratified_score_weighted",
        "min_position_weight": 0.01,
        "consensus_multiplier": 1.2,
        "conservative_only_multiplier": 1.0,
        "aggressive_only_multiplier": 0.9,
    },
    # STRATEGY EXECUTION RULES
    "strategy_execution": {
        "min_holding_days": 20,
        "max_holding_days": 180,
        "rebalance_frequency": "weekly",
    },
    # TRANSACTION COSTS
    "transaction_costs": {
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
        "min_commission": 1.0,
    },
    # DYNAMIC SIZING (Volatility-based)
    "dynamic_sizing": {
        "enabled": False,
        "base_volatility_target": 0.15,
        "min_position_multiplier": 0.5,
        "max_position_multiplier": 2.0,
        "volatility_window": 20,
    },
}

# ============================================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================================
RISK_PARAMS = {
    # PORTFOLIO EXPOSURE MANAGEMENT
    "exposure_management": {
        "min_exposure_pct": 0.60,
        "max_exposure_pct": 1.00,
    },
    # PORTFOLIO CONSTRUCTION LIMITS
    "portfolio_limits": {
        "max_single_position": 0.18,
        "min_single_position": 0.02,
    },
    # DRAWDOWN CONTROL
    "drawdown_control": {
        "max_drawdown_threshold": 0.25,
        "emergency_drawdown_threshold": 0.30,
        "drawdown_lookback_days": 252,
    },
    # Portfolio-Specific Volatility
    "portfolio_volatility_control": {
        "enabled": True,  #
        "portfolio_volatility_target": 0.15,
        "portfolio_volatility_lookback": 60,
        "portfolio_volatility_max_scaling": 1.5,
        "portfolio_volatility_min_scaling": 0.5,
    },
    # Continuous Risk Scaling
    "continuous_risk_scaling": {
        "enabled": True,
        "drawdown_weight": 0.50,
        "market_regime_weight": 0.30,
        "portfolio_volatility_weight": 0.20,
    },
    # Risk budget levels (adjusted by Layer 3: Risk Manager)
    "risk_budget_levels": {
        "strong_bull": {
            "target_exposure": 1.5,
            "max_positions_factor": 1.2,
        },
        "normal": {
            "target_exposure": 0.98,
            "max_positions_factor": 1.0,
        },
        "cautious": {
            "target_exposure": 0.93,
            "max_positions_factor": 0.9,
        },
        "recovery": {
            "target_exposure": 0.88,
            "max_positions_factor": 0.8,
        },
        "sideways": {
            "target_exposure": 0.60,
            "max_positions_factor": 0.6,
        },
        "defensive": {
            "target_exposure": 0.50,
            "max_positions_factor": 0.5,
        },
        "emergency": {
            "target_exposure": 0.25,
            "max_positions_factor": 0.25,
        },
    },
    # Sideways market detection (price-based only)
    "sideways_detection": {
        "enabled": True,
        "lookback": 30,
        "low_trend_threshold": 0.05,
        "vol_min": 0.008,
        "vol_max": 0.025,
    },
    # Stop loss configuration
    "stop_loss": {
        "enabled": True,
        "fixed_stop_loss": 0.10,
        "trailing_stop_activation": 0.12,
        "trailing_stop_distance": 0.08,
        "profit_protection": {
            "enabled": True,
            "levels": [
                {"profit": 0.20, "action": "trail", "trail_pct": 0.15},
                {"profit": 0.40, "action": "trail", "trail_pct": 0.12},
                {"profit": 0.60, "action": "trail", "trail_pct": 0.10},
            ],
        },
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
        # Regime-based adjustments (references strategy_config.py regimes)
        "market_regime_adjustments": {
            "strong_bull": 1.4,
            "bull": 1.2,
            "normal": 1.0,
            "recovery": 0.9,
            "volatile": 0.7,
            "bear": 0.7,
            "crisis": 0.5,
        },
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_param(category, subcategory, param_name, default=None):
    """Get parameter with fallback."""
    config_map = {
        "TRADING_PARAMS": TRADING_PARAMS,
        "RISK_PARAMS": RISK_PARAMS,
    }
    try:
        return (
            config_map.get(category, {}).get(subcategory, {}).get(param_name, default)
        )
    except Exception:
        return default


def validate_parameters():
    """Validate parameter consistency."""
    errors = []
    warnings = []

    # Utilization check
    target_util = TRADING_PARAMS["position_sizing"]["target_utilization"]
    if target_util > 0.98:
        errors.append(f"Target utilization {target_util:.1%} too high")

    # Position count check
    min_pos = TRADING_PARAMS["position_management"]["min_total_positions"]
    max_pos = TRADING_PARAMS["position_management"]["max_total_positions"]
    if min_pos > max_pos:
        errors.append(f"Min positions ({min_pos}) > Max positions ({max_pos})")

    # Drawdown thresholds
    max_dd = RISK_PARAMS["drawdown_control"]["max_drawdown_threshold"]
    emergency_dd = RISK_PARAMS["drawdown_control"]["emergency_drawdown_threshold"]
    if emergency_dd <= max_dd:
        errors.append(f"Emergency DD must be > Max DD")

    # Warnings
    if max_dd > 0.30:
        warnings.append(f"Max drawdown {max_dd:.1%} may be too high")
    if min_pos < 3:
        warnings.append("Less than 3 positions reduces diversification")

    return errors, warnings


# Auto-validate
if __name__ == "__main__":
    errors, warnings = validate_parameters()
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
