# data_pipeline/data_fetchers/individual_stock_fetcher.py

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from .base_fetcher import \
    BaseDataFetcher  # MODIFIED: It will inherit from the updated BaseDataFetcher


class IndividualStockFetcher(BaseDataFetcher):
    def __init__(self, cache_dir: Path, db_config: Dict[str, Any]):
        super().__init__(cache_dir, db_config)
        self.logger = logging.getLogger(__name__)
        # Set cache validity to 24 hours for snapshot data
        self.cache_validity_hours = 24

    @property
    def data_type(self) -> str:
        return "individual_stocks"

    def fetch(
        self,
        symbols: List[str],
        data_types: List[str] = None,
        force_refresh: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Main fetch method for individual stock snapshot data.
        Uses file-based caching with a 24-hour validity.
        """
        if data_types is None:
            data_types = ["fundamentals", "ratings"]

        self.logger.info(
            f"Fetching detailed data for {len(symbols)} stocks. Data types: {data_types}"
        )
        all_data = []

        for symbol in symbols:
            symbol_df = None

            # 1. Try to load from cache first (if not forcing refresh)
            if not force_refresh:
                symbol_df = self.load_from_cache(symbol)

            # 2. If no valid cache, fetch from API
            if symbol_df is None:
                self.logger.info(f"Fetching fresh data for {symbol} from API.")
                symbol_data_dict = self._fetch_from_api(symbol, data_types)
                if symbol_data_dict:
                    # Convert dict to a single-row DataFrame
                    symbol_df = pd.DataFrame([symbol_data_dict])
                    self.save_to_cache(symbol_df, symbol)

            if symbol_df is not None and not symbol_df.empty:
                all_data.append(symbol_df)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(
                f"Successfully processed data for {len(combined_df)}/{len(symbols)} stocks."
            )
            return combined_df
        else:
            self.logger.warning(
                "No individual stock data could be fetched or loaded from cache."
            )
            return pd.DataFrame()

    def _fetch_from_api(self, symbol: str, data_types: List[str]) -> Optional[Dict]:
        """Fetches all required data for a single symbol from the yfinance API."""
        try:
            ticker = yf.Ticker(symbol)
            # Use ticker.info as it's the most reliable and contains most fundamental data
            info = ticker.info

            if not info or info.get("quoteType") is None:
                self.logger.warning(
                    f"Could not retrieve valid info for {symbol}. Skipping."
                )
                return None

            symbol_data = {"symbol": symbol, "timestamp": datetime.now()}

            if "fundamentals" in data_types:
                symbol_data.update(self._get_fundamentals(info))

            if "ratings" in data_types:
                # This call is less reliable, handle failures gracefully
                try:
                    symbol_data.update(self._get_analyst_summary(ticker))
                except Exception as e:
                    self.logger.warning(f"Could not fetch ratings for {symbol}: {e}")

            if "earnings" in data_types:
                try:
                    symbol_data.update(self._get_earnings_data(ticker))
                except Exception as e:
                    self.logger.warning(f"Could not fetch earnings for {symbol}: {e}")

            return symbol_data

        except Exception as e:
            self.logger.error(f"Major error fetching API data for {symbol}: {e}")
            return None

    def _get_fundamentals(self, info: Dict) -> Dict:
        """Extracts key fundamental metrics from the ticker.info dictionary."""
        return {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
        }

    def _get_analyst_summary(self, ticker) -> Dict:
        """Gets the latest analyst recommendation."""
        recs = ticker.recommendations
        if recs is not None and not recs.empty:
            latest_rec = recs.iloc[-1]
            return {
                "analyst_rating": latest_rec.get("To Grade"),
                "analyst_firm": latest_rec.get("Firm"),
            }
        return {}

    def _get_earnings_data(self, ticker) -> Dict:
        """Gets recent earnings data."""
        # yfinance earnings data can be sparse. This is a best-effort attempt.
        earnings = ticker.earnings
        if not earnings.empty:
            latest_earnings = earnings.iloc[-1]
            return {
                "latest_quarterly_earnings": latest_earnings.get("Earnings"),
                "latest_quarterly_revenue": latest_earnings.get("Revenue"),
            }
        return {}

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validates the structure of the final DataFrame."""
        return (
            not data.empty and "symbol" in data.columns and "timestamp" in data.columns
        )
