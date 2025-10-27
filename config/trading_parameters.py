# === CORE TRADING PARAMETERS ===
TRADING_PARAMS = {
    "transaction_costs": {
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,  # Fallback for when model data is unavailable
        "min_commission": 1.0,
        "enable_volume_slippage_model": True,
        "slippage_model_k": 0.1,
    },
    "position_sizing": {
        "min_trade_size_dollars": 2000.0,
        "min_trade_size_shares": 20,
    },
}

# === CORE RISK PARAMETERS ===
RISK_PARAMS = {
    "portfolio_limits": {
        "max_single_position": 0.20,  # Max weight for any single stock
        "min_single_position": 0.02,  # Min weight to avoid tiny, inefficient positions
    },
    "position_management": {
        "min_total_positions": 5,  # Minimum number of stocks to hold for diversification
        "max_total_positions": 20,  # Maximum number of stocks to hold
    },
    "drawdown_control": {
        "max_drawdown_threshold": 0.25,  # A significant drawdown level that triggers risk reduction
        "emergency_drawdown_threshold": 0.30,  # An emergency level for drastic action
        "drawdown_lookback_days": 252,
    },
    "cvar_control": {
        "enabled": True,
        "cvar_target": 0.02,  # Target a max 2% average loss on the worst 5% of days
        "cvar_lookback": 60,
        "cvar_confidence_level": 0.95,
        "cvar_min_scaling": 0.3,
        "cvar_max_scaling": 1.2,
    },
    "stop_loss": {
        "enabled": True,
        "fixed_stop_loss": 0.10,
        "trailing_stop_activation": 0.12,
        "trailing_stop_distance": 0.08,
        "dynamic_stop": {
            "enabled": True,
            "eloss_weight": 0.5,
            "atr_weight": 0.5,
            "atr_multiplier": 2.92,
            "min_stop_pct": 0.04,
            "max_stop_pct": 0.18,
        },
        "profit_protection": {
            "enabled": True,
            "let_profits_run": {
                "enabled": True,
                "profit_threshold": 0.40,
                "final_trail_pct": 0.10,
            },
            "levels": [
                {"profit": 0.20, "action": "trail", "trail_pct": 0.15},
                {"profit": 0.40, "action": "trail", "trail_pct": 0.12},
                {"profit": 0.60, "action": "trail", "trail_pct": 0.10},
            ],
        },
    },
}
