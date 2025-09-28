# Simplified config module - Clean imports without complex management layer
# Import core configurations for easy access
import logging

# Data layer configurations
from .general_config import (ACTIVE_MACRO_INDICATORS, BACKTEST_END_DATE,
                             BACKTEST_START_DATE, INDIVIDUAL_STOCKS,
                             INITIAL_CAPITAL, RISK_OFF_SYMBOLS,
                             RISK_ON_SYMBOLS, SYMBOLS)
# Strategy layer configurations
from .strategy_config import (ALPHA_UNIFIED_PARAMS, CURRENT_STRATEGY,
                              HYBRID_DUAL_PARAMS, OPTIMIZATION_PARAMS,
                              WFO_ENABLED, WFO_TEST_PERIOD, WFO_TRAIN_PERIOD,
                              create_strategy, print_config_summary)
# Execution layer configurations
from .trading_parameters import (MARKET_REGIME, MONEY_MANAGEMENT, RISK_PARAMS,
                                 TRADING_PARAMS, get_param,
                                 validate_parameters)


def validate_all_configs():
    """Run validation across all config modules"""
    print("Running configuration validation...")
    errors, warnings = validate_parameters()  # from trading_parameters.py

    # Can add validation from other modules here if needed in the future

    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors and not warnings:
        print("All configurations seem valid.")

    return len(errors) == 0


# Auto-validate on import
if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    is_valid = validate_all_configs()
    if is_valid:
        logger.info("Configuration system loaded successfully")
    else:
        logger.warning("Configuration system loaded with validation issues")

    logger.info(f"Current strategy: {CURRENT_STRATEGY}")
    logger.info(f"Total symbols: {len(SYMBOLS)}")
    logger.info(f"WFO enabled: {WFO_ENABLED}")
