# General configuration - Data layer settings
# Asset definitions, data sources, and system-wide constants

import os
from datetime import date

RISK_ON_SYMBOLS = [
    # Broad Market
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM",  # Russell 2000 (Small Caps)
    # Key Sectors
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLE",  # Energy
    "XLI",  # Industrials
]

RISK_OFF_SYMBOLS = [
    # Bonds
    "IEF",  # 7-10 Year Treasury Bond ETF
    # Commodities & Currencies
    "GLD",  # Gold
    # Defensive Equities
    "USMV",  # USA Minimum Volatility ETF
]
INDIVIDUAL_STOCKS = [
    # --- Technology & AI Leaders ---
    "NVDA",  # Unquestioned AI leader
    "MSFT",  # Cloud & Enterprise AI
    "AAPL",  # Consumer Tech Ecosystem
    "GOOGL",  # Search & Digital Ads
    "AMZN",  # E-commerce & Cloud (AWS)
    "ASML",  # Semiconductor Equipment Monopoly
    'META',  # Social Media & AI
    'TSLA',  # Electric Vehicles & Energy
    # --- Semiconductor Ecosystem (Deepened) ---
    "ASML",  # Photolithography Monopoly
    "LRCX",  # Lam Research: Wafer Fabrication Equipment
    "TSM",  # Taiwan Semiconductor: Leading-edge Chip Manufacturing
    "AMD",  # Advanced Micro Devices: Key competitor in CPU/GPU
    # --- Financial & Industrial Blue Chips ---
    "JPM",  # Top US Bank
    # --- Industrials, Aerospace & Logistics (Diversified) ---
    "GE",  # GE Aerospace: Aircraft Engines
    "UPS",  # United Parcel Service: Global Logistics Leader
    # --- Health Care & Consumer Staples ---
    "LLY",  # Pharma Giant (for Eli Lilly) - NOTE: LLY is a better choice than others now
    "COST",  # Premier Warehouse Retailer
    # --- Energy ---
    "XOM",  # Supermajor Energy Producer
    'COST', 'WMT', 'PG', 'KO', 'PEP', 'MDLZ', 'CL'  # Consumer Staples Giants
]


# Combined symbol list for data fetching
SYMBOLS = list(set(RISK_ON_SYMBOLS + RISK_OFF_SYMBOLS + INDIVIDUAL_STOCKS))

# Conservative and Aggressive symbol groups for strategy allocation
CONSERVATIVE_SYMBOLS = RISK_OFF_SYMBOLS
AGGRESSIVE_INDIVIDUAL_STOCKS = INDIVIDUAL_STOCKS

# === Date and Time Settings ===
DATA_START_DATE = "2000-01-01"  # Historical data fetch start
BACKTEST_START_DATE = "2010-01-01"  # Strategy backtest start
WARMUP_PERIOD_DAYS = 252
BACKTEST_END_DATE = "2025-08-01"  # Strategy backtest end
INITIAL_CAPITAL = 100000.0  # Starting capital for backtests

# === Cache and Storage Configuration ===
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
PROCESSED_DATA_CACHE_PATH = os.path.join(CACHE_DIR, "all_processed_data.parquet")

# === Macro Economic Indicators Configuration ===
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

# Flatten macro indicators for data fetching
ACTIVE_MACRO_INDICATORS = {
    **MACRO_INDICATORS["interest_rates"],
    **MACRO_INDICATORS["inflation"],
    **MACRO_INDICATORS["market_sentiment"],
    **MACRO_INDICATORS["economic_activity"],
}

# === Data Ingestion Settings ===
DATA_INGESTION_SETTINGS = {
    "default": {"start_date": "2000-01-01", "end_date": None},
    "yahoo_finance": {"start_date": "2000-01-01", "end_date": None},
    "fred_macro": {
        "start_date": "2000-01-01",
        "end_date": None,
        "indicators": ACTIVE_MACRO_INDICATORS,
    },
}

# Update frequency mapping for different data types
DATA_UPDATE_FREQUENCY = {
    "high_frequency": 30,  # Daily data like rates, VIX
    "monthly": 90,  # Monthly economic data
    "quarterly": 180,  # Quarterly reports like GDP
}

# Macro indicator frequency classification
MACRO_INDICATOR_FREQUENCY_MAP = {
    # High Frequency (daily/weekly updates)
    "DFF": "high_frequency",
    "DGS10": "high_frequency",
    "DGS2": "high_frequency",
    "DGS5": "high_frequency",
    "T10Y2Y": "high_frequency",
    "TB3MS": "high_frequency",
    "VIXCLS": "high_frequency",
    "WALCL": "high_frequency",
    "T5YIE": "high_frequency",
    # Monthly updates
    "CPIAUCSL": "monthly",
    "CPILFESL": "monthly",
    "UNRATE": "monthly",
    "PAYEMS": "monthly",
    "INDPRO": "monthly",
    "HOUST": "monthly",
    "M2SL": "monthly",
    "DFEDTARU": "monthly",
    # Quarterly updates
    "GDP": "quarterly",
}

# === Data Validation Settings ===
MIN_DATA_POINTS = 252  # Minimum trading days for valid backtest
MIN_SYMBOLS_REQUIRED = 10  # Minimum symbols needed to proceed
MAX_MISSING_DATA_PCT = 0.1  # Maximum allowable missing data percentage

# === Logging Configuration ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "quantitative_strategy.log"

if __name__ == "__main__":
    print("--- Running general_config.py for testing ---")
    print(f"Optimized Risk-on ETFs: {len(RISK_ON_SYMBOLS)}")
    print(f"Optimized Risk-off ETFs: {len(RISK_OFF_SYMBOLS)}")
    print(f"Optimized Individual Stocks: {len(INDIVIDUAL_STOCKS)}")
    print(f"General config loaded: {len(SYMBOLS)} total symbols")
    print(
        f"Risk-on assets: {len(RISK_ON_SYMBOLS)}, Risk-off assets: {len(RISK_OFF_SYMBOLS)}"
    )
    print(f"Individual stocks: {len(INDIVIDUAL_STOCKS)}")
    print(f"Macro indicators: {len(ACTIVE_MACRO_INDICATORS)}")
