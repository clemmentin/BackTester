from .general_config import (
    SYMBOLS,
    RISK_ON_SYMBOLS,
    RISK_OFF_SYMBOLS,
    INITIAL_CAPITAL,
    BACKTEST_END_DATE,
    BACKTEST_START_DATE,
    WARMUP_PERIOD_DAYS,
    GLOBAL_RANDOM_SEED,
)

# Import from strategy_config.py
from .strategy_config import (
    HYBRID_DUAL_PARAMS,
    ALPHA_UNIFIED_PARAMS,
    WFO_ENABLED,
    WFO_TEST_PERIOD,
    WFO_TRAIN_PERIOD,
    get_current_strategy_params,
    CURRENT_STRATEGY,
)

# Import from trading_parameters.py
from .trading_parameters import RISK_PARAMS, TRADING_PARAMS


class ConfigRegistry:
    """
    Central registry for accessing configuration parameters.
    Automatically routes requests to the correct config file.
    """

    @staticmethod
    def get_strategy_param(section: str, key: str = None, default=None):
        strategy_params = get_current_strategy_params()
        if section not in strategy_params:
            return default if key else {}
        section_data = strategy_params[section]
        if key is None:
            return section_data
        return section_data.get(key, default)

    @staticmethod
    def get_trading_param(
        category: str, section: str = None, key: str = None, default=None
    ):
        config_map = {"RISK_PARAMS": RISK_PARAMS, "TRADING_PARAMS": TRADING_PARAMS}
        if category not in config_map:
            return default
        category_data = config_map[category]
        if section is None:
            return category_data
        if section not in category_data:
            return default if key else {}
        section_data = category_data[section]
        if key is None:
            return section_data
        return section_data.get(key, default)

    @staticmethod
    def get_param_with_override(kwargs: dict, section: str, key: str, default):
        if key in kwargs:
            return kwargs[key]
        if section in kwargs:
            section_dict = kwargs[section]
            if isinstance(section_dict, dict) and key in section_dict:
                return section_dict[key]
        value = ConfigRegistry.get_strategy_param(section, key)
        if value is not None:
            return value
        for category in ["RISK_PARAMS", "TRADING_PARAMS"]:
            value = ConfigRegistry.get_trading_param(category, section, key)
            if value is not None:
                return value
        return default


config = ConfigRegistry()

get_strategy_param = config.get_strategy_param
get_trading_param = config.get_trading_param
get_param_with_override = config.get_param_with_override

__all__ = [
    # General parameters
    "SYMBOLS",
    "DEFENSIVE_QUALITY_STOCKS",
    "DEFENSIVE_SAFE_HAVENS",
    "RISK_ON_SYMBOLS",
    "RISK_OFF_SYMBOLS",
    "INITIAL_CAPITAL",
    "BACKTEST_START_DATE",
    "BACKTEST_END_DATE",
    "WARMUP_PERIOD_DAYS",
    "GLOBAL_RANDOM_SEED",
    # Strategy parameters
    "HYBRID_DUAL_PARAMS",
    "ALPHA_UNIFIED_PARAMS",
    "CURRENT_STRATEGY",
    "WFO_ENABLED",
    # Trading and risk parameters
    "RISK_PARAMS",
    "TRADING_PARAMS",
    # Functions
    "get_current_strategy_params",
    "get_strategy_param",
    "get_trading_param",
    "get_param_with_override",
    # Registry instance
    "config",
]
