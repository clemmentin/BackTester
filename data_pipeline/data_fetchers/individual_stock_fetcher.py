import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sqlalchemy import text

from .base_fetcher import BaseDataFetcher

# Load environment variables
load_dotenv()


class IndividualStockFetcher(BaseDataFetcher):
    """ """
    def __init__(self, cache_dir: Path, db_config: Dict[str, Any]):
        super().__init__(cache_dir, db_config)
        self.logger = logging.getLogger(__name__)
        self.cache_validity_hours = 24
        self.table_name = "fundamental_data"

        # Load FMP API key from environment
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FMP_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )

        # FMP API base URL
        self.base_url = "https://financialmodelingprep.com/api/v3"

        self.logger.info(
            "IndividualStockFetcher initialized with FMP API and database support"
        )

    @property
    def data_type(self) -> str:
        """ """
        return "individual_stocks"

    def _uses_database(self) -> bool:
        """This fetcher uses the database."""
        return True

    def fetch(
        self,
        symbols: List[str],
        data_types: List[str] = None,
        force_refresh: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Main fetch method for individual stock snapshot data.
        Uses database if available, falls back to file-based caching.

        Args:
          symbols: List[str]: 
          data_types: List[str]:  (Default value = None)
          force_refresh: bool:  (Default value = False)
          **kwargs: 

        Returns:

        """
        if data_types is None:
            data_types = ["fundamentals", "ratios"]

        self.logger.info(
            f"Fetching detailed data for {len(symbols)} stocks using FMP API. "
            f"Data types: {data_types}"
        )
        all_data = []

        for symbol in symbols:
            symbol_df = None

            # 1. Try to load from database first (if not forcing refresh)
            if not force_refresh and self.engine:
                symbol_df = self._load_from_database(symbol)
                if symbol_df is not None:
                    self.logger.debug(f"Loaded {symbol} from database")

            # 2. If no DB data, try cache
            if symbol_df is None and not force_refresh:
                symbol_df = self.load_from_cache(symbol)
                if symbol_df is not None:
                    self.logger.debug(f"Loaded {symbol} from file cache")

            # 3. If still no data, fetch from API
            if symbol_df is None:
                self.logger.info(f"Fetching fresh data for {symbol} from FMP API.")
                symbol_data_dict = self._fetch_from_api(symbol, data_types)
                if symbol_data_dict:
                    symbol_df = pd.DataFrame([symbol_data_dict])
                    # Save to both cache and database
                    self.save_to_cache(symbol_df, symbol)
                    if self.engine:
                        self._save_to_database(symbol_df, symbol)

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

    def _load_from_database(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load fundamental data from database for a symbol.

        Args:
          symbol: str: 

        Returns:

        """
        if not self.engine:
            return None
        try:
            # Get the most recent record for this symbol
            query = text(
                f"SELECT * FROM {self.table_name} WHERE symbol = :symbol "
                f"ORDER BY timestamp DESC LIMIT 1"
            )
            df = pd.read_sql(query, self.engine, params={"symbol": symbol})

            if df.empty:
                return None

            # Check if data is fresh (within cache validity period)
            if "timestamp" in df.columns:
                data_age = datetime.now() - pd.to_datetime(df["timestamp"].iloc[0])
                if data_age > timedelta(hours=self.cache_validity_hours):
                    self.logger.debug(
                        f"Database data for {symbol} is stale ({data_age.days} days old)"
                    )
                    return None

            return df

        except Exception as e:
            self.logger.error(f"Error loading {symbol} from database: {e}")
            return None

    def _save_to_database(self, data: pd.DataFrame, symbol: str):
        """Save fundamental data to database.

        Args:
          data: pd.DataFrame: 
          symbol: str: 

        Returns:

        """
        if not self.engine or data.empty:
            return
        try:
            # Prepare data for insertion
            data_to_save = data.copy()

            # Ensure timestamp column exists
            if "timestamp" not in data_to_save.columns:
                data_to_save["timestamp"] = datetime.now()

            with self.engine.begin() as conn:
                # Delete existing records for this symbol (upsert logic)
                delete_query = text(
                    f"DELETE FROM {self.table_name} WHERE symbol = :symbol"
                )
                conn.execute(delete_query, {"symbol": symbol})

                # Insert new data
                data_to_save.to_sql(
                    self.table_name,
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                )

            self.logger.info(
                f"Successfully saved fundamental data for {symbol} to database"
            )
        except Exception as e:
            self.logger.error(f"Error saving {symbol} to database: {e}")

    def _make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to FMP API.
        
        Args:
            endpoint: API endpoint (e.g., 'profile/AAPL')

        Args:
          endpoint: str: 
          params: Dict:  (Default value = None)

        Returns:
          JSON response as dictionary, or None if request fails

        """
        if params is None:
            params = {}

        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # FMP returns empty list or error dict for invalid symbols
            if isinstance(data, list) and len(data) == 0:
                self.logger.warning(
                    f"No data returned from FMP for endpoint: {endpoint}"
                )
                return None
            if isinstance(data, dict) and "Error Message" in data:
                self.logger.error(f"FMP API error: {data.get('Error Message')}")
                return None

            return data

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {endpoint}: {e}")
            return None
        except ValueError as e:
            self.logger.error(f"Failed to parse JSON response for {endpoint}: {e}")
            return None

    def _fetch_from_api(self, symbol: str, data_types: List[str]) -> Optional[Dict]:
        """Fetches all required data for a single symbol using FMP API.

        Args:
          symbol: str: 
          data_types: List[str]: 

        Returns:

        """
        try:
            # Initialize data dictionary
            symbol_data = {"symbol": symbol, "timestamp": datetime.now()}

            if "fundamentals" in data_types:
                fundamentals = self._get_fundamentals(symbol)
                if fundamentals:
                    symbol_data.update(fundamentals)
                else:
                    self.logger.warning(f"Could not retrieve fundamentals for {symbol}")
                    return None

            if "ratios" in data_types:
                ratios = self._get_ratios(symbol)
                if ratios:
                    symbol_data.update(ratios)

            return symbol_data

        except Exception as e:
            self.logger.error(f"Major error fetching API data for {symbol}: {e}")
            return None

    def _get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Extract fundamental data from FMP API.
        Combines data from multiple endpoints for maximum completeness.

        Args:
          symbol: str: 

        Returns:

        """
        fundamentals = {}

        # 1. Get company profile (market cap, beta, dividend yield, etc.)
        profile_data = self._make_api_request(f"profile/{symbol}")
        if profile_data and isinstance(profile_data, list) and len(profile_data) > 0:
            profile = profile_data[0]

            fundamentals["market_cap"] = profile.get("mktCap")
            fundamentals["beta"] = profile.get("beta")
            fundamentals["dividend_yield"] = (
                profile.get("lastDiv") / profile.get("price")
                if profile.get("lastDiv") and profile.get("price")
                else None
            )
            fundamentals["price"] = profile.get("price")
            fundamentals["company_name"] = profile.get("companyName")
            fundamentals["sector"] = profile.get("sector")
            fundamentals["industry"] = profile.get("industry")

        # 2. Get key metrics (PE, PB, ROE, ROA, etc.)
        metrics_data = self._make_api_request(f"key-metrics/{symbol}", {"limit": 1})
        if metrics_data and isinstance(metrics_data, list) and len(metrics_data) > 0:
            metrics = metrics_data[0]

            fundamentals["pe_ratio"] = metrics.get("peRatio")
            fundamentals["price_to_book"] = metrics.get("pbRatio")
            fundamentals["roe"] = metrics.get("roe")
            fundamentals["roa"] = metrics.get("roa")
            fundamentals["revenue_per_share"] = metrics.get("revenuePerShare")
            fundamentals["free_cash_flow_per_share"] = metrics.get(
                "freeCashFlowPerShare"
            )
            fundamentals["book_value"] = metrics.get("bookValuePerShare")
            fundamentals["tangible_book_value"] = metrics.get(
                "tangibleBookValuePerShare"
            )
            fundamentals["debt_to_equity"] = metrics.get("debtToEquity")
            fundamentals["current_ratio"] = metrics.get("currentRatio")
            fundamentals["quick_ratio"] = metrics.get("quickRatio")

        # 3. Get income statement (revenue, earnings, EBITDA, margins, etc.)
        income_data = self._make_api_request(f"income-statement/{symbol}", {"limit": 1})
        if income_data and isinstance(income_data, list) and len(income_data) > 0:
            income = income_data[0]

            fundamentals["revenue"] = income.get("revenue")
            fundamentals["cogs"] = income.get("costOfRevenue")
            fundamentals["gross_profit"] = income.get("grossProfit")
            fundamentals["operating_income"] = income.get("operatingIncome")
            fundamentals["ebitda"] = income.get("ebitda")
            fundamentals["earnings"] = income.get("netIncome")
            fundamentals["eps"] = income.get("eps")
            fundamentals["eps_diluted"] = income.get("epsdiluted")

            # Calculate margins
            if income.get("revenue"):
                fundamentals["gross_margin"] = (
                    income.get("grossProfit") / income.get("revenue")
                    if income.get("grossProfit")
                    else None
                )
                fundamentals["operating_margin"] = (
                    income.get("operatingIncome") / income.get("revenue")
                    if income.get("operatingIncome")
                    else None
                )
                fundamentals["profit_margin"] = (
                    income.get("netIncome") / income.get("revenue")
                    if income.get("netIncome")
                    else None
                )

        # 4. Get balance sheet (assets, debt, cash, equity, etc.)
        balance_data = self._make_api_request(
            f"balance-sheet-statement/{symbol}", {"limit": 1}
        )
        if balance_data and isinstance(balance_data, list) and len(balance_data) > 0:
            balance = balance_data[0]

            fundamentals["total_assets"] = balance.get("totalAssets")
            fundamentals["total_debt"] = balance.get("totalDebt")
            fundamentals["cash"] = balance.get("cashAndCashEquivalents")
            fundamentals["shareholders_equity"] = balance.get("totalStockholdersEquity")
            fundamentals["total_liabilities"] = balance.get("totalLiabilities")
            fundamentals["current_assets"] = balance.get("totalCurrentAssets")
            fundamentals["current_liabilities"] = balance.get("totalCurrentLiabilities")

        # 5. Get cash flow statement (FCF, operating cash flow, etc.)
        cashflow_data = self._make_api_request(
            f"cash-flow-statement/{symbol}", {"limit": 1}
        )
        if cashflow_data and isinstance(cashflow_data, list) and len(cashflow_data) > 0:
            cashflow = cashflow_data[0]

            fundamentals["operating_cash_flow"] = cashflow.get("operatingCashFlow")
            fundamentals["free_cash_flow"] = cashflow.get("freeCashFlow")
            fundamentals["capex"] = cashflow.get("capitalExpenditure")
            fundamentals["dividends_paid"] = cashflow.get("dividendsPaid")

        # Data quality check: must have at least total_assets and revenue
        if (
            fundamentals.get("total_assets") is None
            and fundamentals.get("revenue") is None
        ):
            self.logger.warning(
                f"Insufficient fundamental data for {symbol}: "
                f"missing both total_assets and revenue"
            )
            return None

        return fundamentals

    def _get_ratios(self, symbol: str) -> Dict:
        """Gets additional financial ratios from FMP.

        Args:
          symbol: str: 

        Returns:

        """
        ratios = {}

        ratios_data = self._make_api_request(f"ratios/{symbol}", {"limit": 1})
        if ratios_data and isinstance(ratios_data, list) and len(ratios_data) > 0:
            ratio = ratios_data[0]

            ratios["pe_ratio_ttm"] = ratio.get("priceEarningsRatio")
            ratios["price_to_sales"] = ratio.get("priceToSalesRatio")
            ratios["price_to_book_ttm"] = ratio.get("priceToBookRatio")
            ratios["price_to_fcf"] = ratio.get("priceToFreeCashFlowsRatio")
            ratios["ev_to_ebitda"] = ratio.get("enterpriseValueMultiple")
            ratios["peg_ratio"] = ratio.get("pegRatio")
            ratios["debt_ratio"] = ratio.get("debtRatio")
            ratios["asset_turnover"] = ratio.get("assetTurnover")
            ratios["inventory_turnover"] = ratio.get("inventoryTurnover")
            ratios["receivables_turnover"] = ratio.get("receivablesTurnover")
            ratios["dividend_payout_ratio"] = ratio.get("dividendPayoutRatio")

        return ratios

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validates the structure of the final DataFrame.

        Args:
          data: pd.DataFrame: 

        Returns:

        """
        return (
            not data.empty and "symbol" in data.columns and "timestamp" in data.columns
        )


# Example usage and testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test configuration
    cache_dir = Path("./cache")
    db_config = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }

    try:
        fetcher = IndividualStockFetcher(cache_dir, db_config)

        # Test with a few symbols
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        result = fetcher.fetch(test_symbols, force_refresh=True)

        if not result.empty:
            print(f"\nSuccessfully fetched data for {len(result)} stocks")
            print(f"Columns: {result.columns.tolist()}")
            print(f"\nSample data:\n{result.head()}")
        else:
            print("No data fetched")

    except Exception as e:
        print(f"Error during test: {e}")
        sys.exit(1)
