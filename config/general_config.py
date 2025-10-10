import os
from datetime import date

CORE_MOMENTUM_UNIVERSE = [
    # S-Tier Winners (Top Profit Generators)
    "DDOG",
    "CRWD",
    "FTNT",
    "F",
    "GM",
    "TGT",
    "SNOW",
    "COP",
    # A-Tier Winners (Consistent Strong Performers)
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
    # Solid Performers with Good Profit Factor
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
    "VOO",  # VOO (like SPY) is kept for benchmark/diversification
]

# Sub-Group 2: SELECT CYCLICAL / REVERSAL UNIVERSE
# These stocks showed moderate success. They are often cyclical and may be good
# candidates for the reversal alpha module, but require careful management.
# This list is heavily curated from the original.
SELECT_CYCLICAL_REVERSAL = [
    "AAPL",  # Profitable, but lower win-rate. Kept for its market influence.
    "AMZN",  # Strong long-term performer.
    "MSFT",  # Core tech holding.
    "GOOGL",  # Core tech holding.
    "UNH",  # Leader in healthcare.
    "JNJ",  # Defensive qualities, moderately positive PnL.
    "V",  # Strong financials performer.
    "MA",
    "HD",
    "LLY",  # Positive PnL, important sector.
    "PYPL",
    "RTX",
]


# --- PRIMARY RISK-OFF UNIVERSE ---
# Drastically simplified. The model is not effective at trading most defensive ETFs.
# We retain only the most essential, uncorrelated safe-haven assets.
PRIMARY_RISK_OFF_UNIVERSE = [
    "GLD",  # Gold: The primary safe haven. Showed positive PnL.
    "KO",  # Coca-Cola: Showed surprisingly high Profit Factor in limited trades.
    "PG",  # Procter & Gamble: Also showed high Profit Factor.
]


# --- DERIVED UNIVERSES ---
# This section automatically combines the lists above. You don't need to change it.
PRIMARY_RISK_ON_UNIVERSE = list(set(CORE_MOMENTUM_UNIVERSE + SELECT_CYCLICAL_REVERSAL))
RISK_ON_SYMBOLS = list(set(PRIMARY_RISK_ON_UNIVERSE))
RISK_OFF_SYMBOLS = list(set(PRIMARY_RISK_OFF_UNIVERSE))
SYMBOLS = list(set(RISK_ON_SYMBOLS + RISK_OFF_SYMBOLS))

# For compatibility with UnifiedDataManager (fetches data for individual stocks)
INDIVIDUAL_STOCKS = list(set(RISK_ON_SYMBOLS + RISK_OFF_SYMBOLS) - {"VOO", "GLD"})
# ============================================================================
# DATA FETCHING CONFIGURATION
# ============================================================================

# Date ranges
DATA_START_DATE = "1990-01-01"  # Historical data fetch start
BACKTEST_START_DATE = "2000-01-01"  # Strategy backtest start
BACKTEST_END_DATE = "2025-08-01"  # Strategy backtest end
WARMUP_PERIOD_DAYS = 252  # Technical indicator warmup
GLOBAL_RANDOM_SEED = 42  # For reproducibility
INITIAL_CAPITAL = 100000.0  # Initial capital for backtests

# Data source toggles
ENABLE_FUNDAMENTAL_DATA = False  # Deprecated for 3-factor model
ENABLE_MACRO_DATA = True  # NEW: Enable for enhanced market detection

# Cache and storage
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
PROCESSED_DATA_CACHE_PATH = os.path.join(CACHE_DIR, "all_processed_data.parquet")

# ============================================================================
# PERFORMANCE AND CACHING
# ============================================================================

GARCH_LOOKBACK_DAYS = 1000  # Number of days for fitting the GARCH model.
BETA_CACHE_MAX_AGE_DAYS = (
    20  # How many days to keep a calculated beta value before re-calculating.
)
BETA_CACHE_MAX_SIZE = 100  # The maximum number of beta sets to store in memory.

# ============================================================================
# MACRO ECONOMIC INDICATORS (Data Layer Only)
# ============================================================================

# Define available macro indicators organized by category
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

# Flatten for fetcher (all available indicators)
ACTIVE_MACRO_INDICATORS = {
    **MACRO_INDICATORS["interest_rates"],
    **MACRO_INDICATORS["inflation"],
    **MACRO_INDICATORS["market_sentiment"],
    **MACRO_INDICATORS["economic_activity"],
}

# Update frequency for MacroDataFetcher
MACRO_INDICATOR_FREQUENCY_MAP = {
    # High frequency (daily/weekly)
    "DFF": "high_frequency",
    "DGS10": "high_frequency",
    "DGS2": "high_frequency",
    "DGS5": "high_frequency",
    "T10Y2Y": "high_frequency",
    "TB3MS": "high_frequency",
    "VIXCLS": "high_frequency",
    "WALCL": "high_frequency",
    "T5YIE": "high_frequency",
    # Monthly
    "CPIAUCSL": "monthly",
    "CPILFESL": "monthly",
    "UNRATE": "monthly",
    "PAYEMS": "monthly",
    "INDPRO": "monthly",
    "HOUST": "monthly",
    "M2SL": "monthly",
    "DFEDTARU": "monthly",
    # Quarterly
    "GDP": "quarterly",
}

# ============================================================================
# DATA VALIDATION
# ============================================================================

MIN_DATA_POINTS = 252  # Minimum trading days for valid backtest
MIN_SYMBOLS_REQUIRED = 10  # Minimum symbols needed
MAX_MISSING_DATA_PCT = 0.1  # Max allowable missing data

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "quantitative_strategy.log"
