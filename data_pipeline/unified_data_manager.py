# data_pipeline/unified_data_manager.py

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

import config
from data_pipeline.data_fetchers.individual_stock_fetcher import IndividualStockFetcher
from data_pipeline.data_fetchers.macro_fetcher import MacroDataFetcher
from data_pipeline.data_fetchers.stock_fetcher import StockDataFetcher
from data_pipeline.prepare_data import prepare_features_for_backtest

load_dotenv()


class UnifiedDataManager:
    """ """
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

        # UPDATED: Conditional initialization based on config
        self.enable_fundamental_data = getattr(
            self.config.general_config, "ENABLE_FUNDAMENTAL_DATA", False
        )
        self.enable_macro_data = getattr(
            self.config.general_config, "ENABLE_MACRO_DATA", False
        )

        if self.enable_fundamental_data:
            self.logger.warning("Fundamental data enabled")
            self.individual_fetcher = IndividualStockFetcher(
                self.cache_dir, self.db_config
            )
        else:
            self.logger.info("Fundamental data disabled ")
            self.individual_fetcher = None

        if self.enable_macro_data:
            self.logger.info("Macro data enabled for enhanced regime detection")
            self.macro_fetcher = MacroDataFetcher(self.cache_dir, self.db_config)
        else:
            self.logger.info("Macro data disabled (regime detection uses price only)")
            self.macro_fetcher = None

        self.logger.info(
            f"UnifiedDataManager initialized: "
            f"stocks=YES, fundamentals={'YES' if self.enable_fundamental_data else 'NO'}, "
            f"macro={'YES' if self.enable_macro_data else 'NO'}"
        )

    def build_feature_set(
        self,
        force_refresh: bool = False,
        skip_features: bool = False,
        start_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """

        Args:
          force_refresh: bool:  (Default value = False)
          skip_features: bool:  (Default value = False)
          start_date: Optional[str]:  (Default value = None)

        Returns:

        """

        # Step 1: Fetch all raw data
        raw_data = self._fetch_all_raw_data(force_refresh, start_date)
        if not self._validate_raw_data(raw_data):
            self.logger.error("Raw data validation failed. Aborting pipeline.")
            return None

        # Step 2: Prepare base structure
        prepared_data = self._prepare_backtest_structure(raw_data["stocks"])
        if prepared_data is None or prepared_data.empty:
            self.logger.error("Failed to prepare backtest data structure.")
            return None

        # Step 3: Feature engineering
        if not skip_features:
            self.logger.info("Starting feature engineering...")
            final_data = prepare_features_for_backtest(
                prepared_data, raw_data.get("macro"), self.config.general_config
            )
        else:
            self.logger.info("Skipping feature engineering - using raw OHLCV only ")
            final_data = prepared_data

        if final_data is None or final_data.empty:
            self.logger.error("Data is empty after processing.")
            return None

        # Step 5: Cache results
        self._save_to_processed_cache(final_data)

        self.logger.info(f"Pipeline completed. Final data shape: {final_data.shape}")
        return final_data

    def _fetch_all_raw_data(
        self, force_refresh: bool, start_date_override: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """

        Args:
          force_refresh: bool: 
          start_date_override: Optional[str]:  (Default value = None)

        Returns:

        """
        self.logger.info(f"Fetching raw data (force_refresh={force_refresh})")

        start_date = (
            start_date_override
            if start_date_override
            else self.config.general_config.DATA_START_DATE
        )
        end_date = self.config.general_config.BACKTEST_END_DATE

        self.logger.info(f"Data period: {start_date} to {end_date}")

        results = {}

        # REQUIRED: Stock OHLCV data
        try:
            results["stocks"] = self.stock_fetcher.fetch(
                self.config.general_config.SYMBOLS, start_date, end_date, force_refresh
            )
            self.logger.info(f"Stock data fetched: {len(results['stocks'])} rows")
        except Exception as e:
            self.logger.error(f" Stock data fetch failed: {e}", exc_info=True)
            results["stocks"] = pd.DataFrame()

        # OPTIONAL: Fundamental data (deprecated)
        if self.enable_fundamental_data and self.individual_fetcher is not None:
            try:
                results["individual"] = self.individual_fetcher.fetch(
                    self.config.general_config.INDIVIDUAL_STOCKS,
                    force_refresh=force_refresh,
                )
                if results["individual"].empty:
                    self.logger.warning(
                        "Fundamental fetcher returned empty (expected for deprecated fetcher)"
                    )
            except Exception as e:
                self.logger.warning(f"Fundamental data fetch failed: {e}")
                results["individual"] = pd.DataFrame()
        else:
            results["individual"] = pd.DataFrame()
            self.logger.debug("Fundamental data skipped (disabled)")

        # OPTIONAL: Macro data
        if self.enable_macro_data and self.macro_fetcher is not None:
            try:
                results["macro"] = self.macro_fetcher.fetch(
                    self.config.general_config.ACTIVE_MACRO_INDICATORS,
                    start_date,
                    end_date,
                    force_refresh,
                )
                self.logger.info(
                    f" Macro data fetched: {len(results['macro'])} rows "
                    f"({len(results['macro'].columns)} indicators)"
                )
            except Exception as e:
                self.logger.warning(f"Macro data fetch failed: {e}")
                results["macro"] = pd.DataFrame()
        else:
            results["macro"] = pd.DataFrame()
            self.logger.debug("Macro data skipped (disabled)")

        return results

    def _validate_raw_data(self, raw_data: Dict[str, pd.DataFrame]) -> bool:
        """

        Args:
          raw_data: Dict[str: 
          pd.DataFrame]: 

        Returns:

        """
        # CRITICAL: Stock data must exist
        if "stocks" not in raw_data or raw_data["stocks"].empty:
            self.logger.error(" Stock price data is MISSING (REQUIRED)")
            return False

        self.logger.info("Stock data validation passed")

        # Optional data - just log status
        if "macro" not in raw_data or raw_data["macro"].empty:
            self.logger.info(
                " Macro data not available (regime detection will use price only)"
            )
        else:
            self.logger.info(
                "Macro data available (can be used for regime enhancement)"
            )

        if "individual" not in raw_data or raw_data["individual"].empty:
            self.logger.debug("Fundamental data not available ")

        return True

    def _prepare_backtest_structure(
        self, stock_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Converts flat stock DataFrame to MultiIndex.

        Args:
          stock_data: pd.DataFrame: 

        Returns:

        """
        try:
            if (
                "timestamp" not in stock_data.columns
                or "symbol" not in stock_data.columns
            ):
                self.logger.error(
                    "Stock data missing required columns: 'timestamp' or 'symbol'"
                )
                return None

            stock_data["timestamp"] = pd.to_datetime(stock_data["timestamp"])
            prepared = stock_data.set_index(["timestamp", "symbol"]).sort_index()

            self.logger.info(
                f"Prepared MultiIndex structure: {prepared.shape} "
                f"({len(prepared.index.get_level_values('symbol').unique())} symbols, "
                f"{len(prepared.index.get_level_values('timestamp').unique())} dates)"
            )

            return prepared
        except Exception as e:
            self.logger.error(f"Error preparing data structure: {e}", exc_info=True)
            return None

    # def _apply_date_filter(self, data: pd.DataFrame) -> pd.DataFrame:
    #    """Filters DataFrame to backtest date range."""
    #    start_date = pd.to_datetime(self.config.general_config.BACKTEST_START_DATE)
    #    end_date = pd.to_datetime(self.config.general_config.BACKTEST_END_DATE)
    #
    #    self.logger.info(
    #        f"Applying date filter: {start_date.date()} to {end_date.date()}"
    #    )
    #
    #    timestamps = data.index.get_level_values("timestamp")
    #    mask = (timestamps >= start_date) & (timestamps <= end_date)
    #
    #    filtered = data[mask]
    #
    #    self.logger.info(
    #        f"Date filter result: {len(filtered)} rows "
    #        f"(removed {len(data) - len(filtered)} rows outside range)"
    #    )
    #
    #    return filtered

    def _save_to_processed_cache(self, data: pd.DataFrame):
        """Saves processed DataFrame to cache.

        Args:
          data: pd.DataFrame: 

        Returns:

        """
        try:
            data.to_parquet(self.processed_cache_path)
            self.logger.info(
                f" Saved {len(data)} rows to cache: {self.processed_cache_path}"
            )
        except Exception as e:
            self.logger.error(f" Failed to save to cache: {e}")
