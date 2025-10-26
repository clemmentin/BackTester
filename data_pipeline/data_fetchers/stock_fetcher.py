import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import pandas as pd
import yfinance as yf
from sqlalchemy import text

from .base_fetcher import BaseDataFetcher


class StockDataFetcher(BaseDataFetcher):
    """ """
    def __init__(self, cache_dir: Path, db_config: Dict[str, Any]):
        super().__init__(cache_dir, db_config)
        self.logger = logging.getLogger(__name__)
        self.table_name = "daily_prices"

    @property
    def data_type(self) -> str:
        """ """
        return "prices"

    def _uses_database(self) -> bool:
        """ """
        return True

    def _get_latest_date_in_db(self, symbol: str) -> Optional[datetime]:
        """

        Args:
          symbol: str: 

        Returns:

        """
        if not self.engine:
            return None
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        f"SELECT MAX(timestamp) FROM {self.table_name} WHERE symbol = :symbol"
                    ),
                    {"symbol": symbol},
                )
                row = result.fetchone()
                return pd.to_datetime(row[0]) if row and row[0] else None
        except Exception as e:
            self.logger.error(f"Error checking latest date for {symbol}: {e}")
            return None

    def _save_to_database(self, data: pd.DataFrame, symbol: str):
        """

        Args:
          data: pd.DataFrame: 
          symbol: str: 

        Returns:

        """
        if not self.engine or data.empty:
            return
        try:
            data_to_save = data[
                [
                    "timestamp",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                ]
            ].copy()
            with self.engine.begin() as conn:
                delete_query = text(
                    f"DELETE FROM {self.table_name} WHERE symbol = :symbol AND timestamp >= :start_date AND timestamp <= :end_date"
                )
                conn.execute(
                    delete_query,
                    {
                        "symbol": symbol,
                        "start_date": data_to_save["timestamp"].min(),
                        "end_date": data_to_save["timestamp"].max(),
                    },
                )
                data_to_save.to_sql(
                    self.table_name,
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=1000,
                )
            self.logger.info(
                f"Successfully saved {len(data_to_save)} records for {symbol}"
            )
        except Exception as e:
            self.logger.error(f"Error saving {symbol} to database: {e}")

    def _load_from_database(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """

        Args:
          symbol: str: 
          start_date: str: 
          end_date: str: 

        Returns:

        """
        if not self.engine:
            return None
        try:
            query = text(
                f"SELECT * FROM {self.table_name} WHERE symbol = :symbol AND timestamp >= :start_date AND timestamp <= :end_date ORDER BY timestamp"
            )
            df = pd.read_sql(
                query,
                self.engine,
                params={
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            return df if not df.empty else None
        except Exception as e:
            self.logger.error(f"Error loading {symbol} from database: {e}")
            return None

    def fetch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """

        Args:
          symbols: List[str]: 
          start_date: str: 
          end_date: str: 
          force_refresh: bool:  (Default value = False)

        Returns:

        """
        all_data = []
        for symbol in symbols:
            self.logger.info(f"Processing {symbol}...")
            data = None
            if force_refresh or not self.engine:
                self.logger.info(
                    f"Force refresh or no DB. Downloading full range for {symbol}."
                )
                data = self._download_and_process(symbol, start_date, end_date)
                if data is not None and not data.empty and self.engine:
                    self._save_to_database(data, symbol)
            else:
                latest_date = self._get_latest_date_in_db(symbol)
                if latest_date:
                    start_dl = latest_date + timedelta(days=1)
                    if start_dl <= pd.to_datetime(end_date):
                        self.logger.info(f"Updating {symbol} from {start_dl.date()}")
                        new_data = self._download_and_process(
                            symbol, start_dl.strftime("%Y-%m-%d"), end_date
                        )
                        if new_data is not None and not new_data.empty:
                            self._save_to_database(new_data, symbol)
                    else:
                        self.logger.info(f"Data for {symbol} is up to date.")
                else:
                    self.logger.info(
                        f"No existing data for {symbol}, downloading full range."
                    )
                    data = self._download_and_process(symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        self._save_to_database(data, symbol)
                data = self._load_from_database(symbol, start_date, end_date)

            if data is not None and not data.empty:
                all_data.append(data)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _download_and_process(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Downloads and standardizes data from yfinance, ensuring end_date is inclusive.

        Args:
          symbol: str: 
          start_date: str: 
          end_date: str: 

        Returns:

        """
        try:
            end_date_dt = pd.to_datetime(end_date)
            end_date_for_api = (end_date_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            self.logger.info(
                f"--> [DIAGNOSTIC] Calling yf.download for {symbol} with: start='{start_date}', end='{end_date_for_api}'"
            )

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date_for_api,  # Use the adjusted end date
                auto_adjust=True,
                progress=False,
            )

            if data.empty:
                self.logger.warning(f"No data returned for {symbol} from yfinance API.")
                return None

            if isinstance(data.columns, pd.MultiIndex):
                self.logger.debug(
                    f"MultiIndex columns detected for {symbol}. Flattening them."
                )
                data.columns = data.columns.get_level_values(0)

            data.reset_index(inplace=True)
            data["symbol"] = symbol
            data.rename(
                columns={
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                inplace=True,
            )
            data["adj_close"] = data["close"]

            # Filter out any data beyond the originally requested end_date
            data = data[data["timestamp"] <= end_date_dt].copy()

            return data[
                [
                    "timestamp",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                ]
            ]
        except Exception as e:
            self.logger.error(f"Download/process failed for {symbol}: {e}")
            return None

    def validate_data(self, data: pd.DataFrame) -> bool:
        """

        Args:
          data: pd.DataFrame: 

        Returns:

        """
        return data is not None and not data.empty and "symbol" in data.columns
