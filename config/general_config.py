import os

# ============================================================================
# TRADING UNIVERSE CONFIGURATION
# ============================================================================

# Core momentum stocks - high profit factor performers
CORE_MOMENTUM_UNIVERSE = [
    "DDOG",
    "CRWD",
    "FTNT",
    "F",
    "GM",
    "TGT",
    "SNOW",
    "COP",
    "LRCX",
    "ETN",
    "WM",
    "CAT",
    "BKNG",
    "ISRG",
    "VRTX",
    "EQIX",
    "NOW",
    "ASML",
    "PLD",
    "CMG",
    "AMT",
    "AVGO",
    "NVDA",
    "TSLA",
    "AMD",
    "LOW",
    "DE",
    "INTU",
    "NFLX",
    "PFE",
    "AXP",
    "BAC",
    "GS",
    "GE",
    "SBUX",
    "TJX",
    "GILD",
    "JPM",
    "VOO",
]

# Cyclical stocks for mean-reversion strategies
SELECT_CYCLICAL_REVERSAL = [
    "AAPL",
    "AMZN",
    "MSFT",
    "GOOGL",
    "UNH",
    "JNJ",
    "V",
    "MA",
    "HD",
    "LLY",
    "PYPL",
    "RTX",
]

# Safe-haven assets for defensive positioning
PRIMARY_RISK_OFF_UNIVERSE = ["GLD", "KO", "PG", "SPY"]

# Auto-derived universes
PRIMARY_RISK_ON_UNIVERSE = list(set(CORE_MOMENTUM_UNIVERSE + SELECT_CYCLICAL_REVERSAL))
RISK_ON_SYMBOLS = list(set(PRIMARY_RISK_ON_UNIVERSE))
RISK_OFF_SYMBOLS = list(set(PRIMARY_RISK_OFF_UNIVERSE))
SYMBOLS = list(set(RISK_ON_SYMBOLS + RISK_OFF_SYMBOLS))
INDIVIDUAL_STOCKS = list(set(RISK_ON_SYMBOLS + RISK_OFF_SYMBOLS) - {"VOO", "GLD"})
# ============================================================================
# DATA FETCHING CONFIGURATION
# ============================================================================

# Backtest date range
DATA_START_DATE = "1990-01-01"
BACKTEST_START_DATE = "2005-01-01"
BACKTEST_END_DATE = "2025-10-22"
WARMUP_PERIOD_DAYS = 365

# Global settings
GLOBAL_RANDOM_SEED = 42
INITIAL_CAPITAL = 100000.0

# Data source flags
ENABLE_FUNDAMENTAL_DATA = False
ENABLE_MACRO_DATA = True

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
PROCESSED_DATA_CACHE_PATH = os.path.join(CACHE_DIR, "all_processed_data.parquet")

# ============================================================================
# PERFORMANCE AND CACHING
# ============================================================================

GARCH_LOOKBACK_DAYS = 1000
BETA_CACHE_MAX_AGE_DAYS = 20
BETA_CACHE_MAX_SIZE = 100

# ============================================================================
# MACRO ECONOMIC INDICATORS
# ============================================================================

MACRO_INDICATORS = {
    "interest_rates": {
        "DFF": "Federal Funds Rate",
        "DGS10": "10-Year Treasury Rate",
        "DGS2": "2-Year Treasury Rate",
        "DGS5": "5-Year Treasury Rate",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "TB3MS": "3-Month Treasury Bill",
    },
    "inflation": {
        "CPIAUCSL": "Consumer Price Index",
        "CPILFESL": "Core CPI",
        "T5YIE": "5-Year Breakeven Inflation",
        "DFEDTARU": "Fed Inflation Target",
    },
    "economic_activity": {
        "GDP": "Gross Domestic Product",
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls",
        "INDPRO": "Industrial Production",
        "HOUST": "Housing Starts",
    },
    "market_sentiment": {
        "VIXCLS": "VIX Volatility Index",
        "WALCL": "Fed Balance Sheet",
        "M2SL": "M2 Money Supply",
    },
}

ACTIVE_MACRO_INDICATORS = {
    **MACRO_INDICATORS["interest_rates"],
    **MACRO_INDICATORS["inflation"],
    **MACRO_INDICATORS["market_sentiment"],
    **MACRO_INDICATORS["economic_activity"],
}

MACRO_INDICATOR_FREQUENCY_MAP = {
    "DFF": "high_frequency",
    "DGS10": "high_frequency",
    "DGS2": "high_frequency",
    "DGS5": "high_frequency",
    "T10Y2Y": "high_frequency",
    "TB3MS": "high_frequency",
    "VIXCLS": "high_frequency",
    "WALCL": "high_frequency",
    "T5YIE": "high_frequency",
    "CPIAUCSL": "monthly",
    "CPILFESL": "monthly",
    "UNRATE": "monthly",
    "PAYEMS": "monthly",
    "INDPRO": "monthly",
    "HOUST": "monthly",
    "M2SL": "monthly",
    "DFEDTARU": "monthly",
    "GDP": "quarterly",
}

# ============================================================================
# DATA VALIDATION
# ============================================================================

MIN_DATA_POINTS = 252
MIN_SYMBOLS_REQUIRED = 10
MAX_MISSING_DATA_PCT = 0.1

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "quantitative_strategy.log"
