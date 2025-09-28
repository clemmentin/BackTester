# data_pipeline/unified_data_manager.py

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Import necessary components from the pipeline
import config

from .data_fetchers.individual_stock_fetcher import IndividualStockFetcher
from .data_fetchers.macro_fetcher import MacroDataFetcher
from .data_fetchers.stock_fetcher import StockDataFetcher
from .prepare_data import \
    prepare_features_for_backtest  # MODIFIED: Import moved here

load_dotenv()


class UnifiedDataManager:
    def __init__(self, config_module: Any = config):
        self.config = config_module
        self.logger = logging.getLogger(__name__)

        self.cache_dir = Path(self.config.general_config.CACHE_DIR)
        self.processed_cache_path = Path(
            self.config.general_config.PROCESSED_DATA_CACHE_PATH
        )
        self.processed_cache_path.parent.mkdir(exist_ok=True, parents=True)

        self.db_config = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
        }

        # Initialize fetchers
        self.stock_fetcher = StockDataFetcher(self.cache_dir, self.db_config)
        self.individual_fetcher = IndividualStockFetcher(self.cache_dir, self.db_config)
        self.macro_fetcher = MacroDataFetcher(self.cache_dir, self.db_config)

        self.logger.info("UnifiedDataManager initialized.")

    def build_feature_set(
        self,
        force_refresh: bool = False,
        skip_features: bool = False,
        start_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        NEW: This is the main method that replaces the old runner.
        It orchestrates fetching, preparation, feature engineering, filtering, and caching.
        """
        self.logger.info("Starting to build the full feature set for backtesting.")

        # Step 1: Fetch all raw data
        raw_data = self._fetch_all_raw_data(force_refresh)
        if not self._validate_raw_data(raw_data):
            self.logger.error("Raw data validation failed. Aborting pipeline.")
            return None

        # Step 2: Prepare the base data structure (MultiIndex DataFrame)
        prepared_data = self._prepare_backtest_structure(raw_data["stocks"])
        if prepared_data is None or prepared_data.empty:
            self.logger.error("Failed to prepare backtest data structure. Aborting.")
            return None

        # Step 3: Perform feature engineering (if not skipped)
        if not skip_features:
            self.logger.info("Starting feature engineering process...")
            final_data = prepare_features_for_backtest(
                prepared_data, raw_data.get("macro"), self.config.general_config
            )
        else:
            self.logger.info("Skipping feature engineering as requested.")
            final_data = prepared_data

        if final_data is None or final_data.empty:
            self.logger.error("Data is empty after feature engineering. Aborting.")
            return None

        # Step 4: Apply final date filtering for the backtest period
        final_data = self._apply_date_filter(final_data)
        if final_data.empty:
            self.logger.error(
                "Data is empty after applying the backtest date filter. Aborting."
            )
            return None

        # Step 5: Save the final processed data to cache
        self._save_to_processed_cache(final_data)

        self.logger.info(
            f"Pipeline completed successfully. Final data shape: {final_data.shape}"
        )
        return final_data

    def _fetch_all_raw_data(
        self, force_refresh: bool, start_date_override: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetches all required raw data from the respective fetchers."""
        self.logger.info(f"Fetching raw data. Force refresh: {force_refresh}")

        # Use extended date range to ensure enough history for indicators
        start_date = (
            start_date_override
            if start_date_override
            else self.config.general_config.DATA_START_DATE
        )
        end_date = self.config.general_config.BACKTEST_END_DATE

        self.logger.info(f"Data fetch period: {start_date} to {end_date}")

        results = {}
        try:
            results["stocks"] = self.stock_fetcher.fetch(
                self.config.general_config.SYMBOLS, start_date, end_date, force_refresh
            )
            results["individual"] = self.individual_fetcher.fetch(
                self.config.general_config.INDIVIDUAL_STOCKS,
                force_refresh=force_refresh,
            )
            results["macro"] = self.macro_fetcher.fetch(
                self.config.general_config.ACTIVE_MACRO_INDICATORS,
                start_date,
                end_date,
                force_refresh,
            )
        except Exception as e:
            self.logger.error(
                f"An error occurred during data fetching: {e}", exc_info=True
            )
            # Ensure keys exist even if fetching fails
            if "stocks" not in results:
                results["stocks"] = pd.DataFrame()
            if "individual" not in results:
                results["individual"] = pd.DataFrame()
            if "macro" not in results:
                results["macro"] = pd.DataFrame()

        return results

    def _validate_raw_data(self, raw_data: Dict[str, pd.DataFrame]) -> bool:
        """Validates that essential raw data was fetched."""
        if "stocks" not in raw_data or raw_data["stocks"].empty:
            self.logger.error("Essential stock price data is missing.")
            return False

        # Log warnings for non-essential missing data
        if "macro" not in raw_data or raw_data["macro"].empty:
            self.logger.warning(
                "Macro data is missing. Proceeding without macro features."
            )
        if "individual" not in raw_data or raw_data["individual"].empty:
            self.logger.warning("Individual stock fundamentals are missing.")

        return True

    def _prepare_backtest_structure(
        self, stock_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Converts the flat stock DataFrame to a MultiIndex DataFrame."""
        try:
            if (
                "timestamp" not in stock_data.columns
                or "symbol" not in stock_data.columns
            ):
                self.logger.error(
                    "Stock data is missing 'timestamp' or 'symbol' columns."
                )
                return None

            stock_data["timestamp"] = pd.to_datetime(stock_data["timestamp"])
            prepared = stock_data.set_index(["timestamp", "symbol"]).sort_index()
            self.logger.info(f"Prepared data structure with shape: {prepared.shape}")
            return prepared
        except Exception as e:
            self.logger.error(f"Error preparing data structure: {e}", exc_info=True)
            return None

    def _apply_date_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filters the DataFrame to the specific backtest date range."""
        start_date = pd.to_datetime(self.config.general_config.BACKTEST_START_DATE)
        end_date = pd.to_datetime(self.config.general_config.BACKTEST_END_DATE)

        log_start_date = pd.Timestamp(start_date).date()
        log_end_date = pd.Timestamp(end_date).date()

        self.logger.info(
            f"Applying final date filter from {log_start_date} to {log_end_date}."
        )

        timestamps = data.index.get_level_values("timestamp")
        mask = (timestamps >= start_date) & (timestamps <= end_date)

        return data[mask]

    def _save_to_processed_cache(self, data: pd.DataFrame):
        """Saves the final, processed DataFrame to the main cache file."""
        try:
            data.to_parquet(self.processed_cache_path)
            self.logger.info(
                f"Saved {len(data)} processed records to cache at {self.processed_cache_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save processed data to cache: {e}")
