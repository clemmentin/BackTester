# data_pipeline/data_fetchers/macro_fetcher.py

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fredapi import Fred
from sqlalchemy import text

from .base_fetcher import BaseDataFetcher  # Import the updated base class


class MacroDataFetcher(BaseDataFetcher):
    def __init__(self, cache_dir: Path, db_config: Dict[str, Any], **kwargs):
        super().__init__(cache_dir, db_config)
        self.logger = logging.getLogger(__name__)
        self.table_name = "macro_data"

        fred_key = os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=fred_key) if fred_key else None
        if not self.fred:
            self.logger.warning("FRED API key not found.")

        # Default configurations can be passed via kwargs
        self.update_frequency = kwargs.get(
            "update_frequency",
            {"high_frequency": 1, "low_frequency": 30, "quarterly": 90},
        )
        self.indicator_frequency_map = kwargs.get("indicator_frequency_map", {})

    @property
    def data_type(self) -> str:
        return "macro"

    # MODIFIED: Tell the base class that this fetcher needs a database
    def _uses_database(self) -> bool:
        return True

    # The rest of the file can remain as is. For completeness:
    def _check_existing_data(self, series_ids: List[str]) -> Dict:
        if not self.engine:
            return {}
        existing_data = {}
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        f"SELECT series_id, MAX(timestamp) as max_date FROM {self.table_name} WHERE series_id = ANY(:series_ids) GROUP BY series_id"
                    ),
                    {"series_ids": series_ids},
                )
                for row in result:
                    existing_data[row[0]] = {"max_date": row[1]}
        except Exception as e:
            self.logger.error(f"Error checking existing macro data: {e}")
        return existing_data

    def _save_to_database(self, data: pd.DataFrame):
        if not self.engine or data.empty:
            return
        try:
            upsert_query = text(
                "INSERT INTO macro_data (timestamp, series_id, value) VALUES (:timestamp, :series_id, :value) ON CONFLICT (timestamp, series_id) DO UPDATE SET value = EXCLUDED.value;"
            )
            records = data[["timestamp", "series_id", "value"]].to_dict("records")
            with self.engine.begin() as conn:
                conn.execute(upsert_query, records)
            self.logger.info(f"Saved/updated {len(records)} macro records.")
        except Exception as e:
            self.logger.error(f"Error saving macro data: {e}")

    def _load_all_from_database(
        self, series_ids: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        if not self.engine:
            return pd.DataFrame()
        try:
            query = text(
                f"SELECT timestamp, series_id, value FROM {self.table_name} WHERE series_id = ANY(:series_ids) AND timestamp >= :start_date AND timestamp <= :end_date ORDER BY timestamp"
            )
            data = pd.read_sql(
                query,
                self.engine,
                params={
                    "series_ids": list(series_ids),
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            return (
                data.pivot(index="timestamp", columns="series_id", values="value")
                if not data.empty
                else pd.DataFrame()
            )
        except Exception as e:
            self.logger.error(f"Error loading macro data: {e}")
            return pd.DataFrame()

    def fetch(
        self,
        indicators: Dict[str, str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not self.fred:
            return pd.DataFrame()
        if not self.engine or force_refresh:
            return self._download_all_indicators(indicators, start_date, end_date)

        existing = self._check_existing_data(list(indicators.keys()))
        to_download = []
        for series_id in indicators:
            if series_id not in existing:
                to_download.append({"id": series_id, "start": start_date})
            elif self._needs_update(series_id, existing[series_id]):
                to_download.append(
                    {
                        "id": series_id,
                        "start": (
                            existing[series_id]["max_date"] + timedelta(days=1)
                        ).strftime("%Y-%m-%d"),
                    }
                )

        if to_download:
            self._download_and_save(to_download, end_date)
        return self._load_all_from_database(
            list(indicators.keys()), start_date, end_date
        )

    def _needs_update(self, series_id: str, info: Dict) -> bool:
        freq = self.indicator_frequency_map.get(series_id, "high_frequency")
        days = self.update_frequency.get(freq, 1)
        return (datetime.now().date() - info["max_date"].date()).days >= days

    def _download_and_save(self, series_list: List[dict], end_date: str):
        all_data = []
        end_date_dt = pd.to_datetime(end_date)

        for series_info in series_list:
            series_id = series_info["id"]
            start = series_info["start"]
            try:
                start_dt = pd.to_datetime(start)

                if start_dt > end_date_dt:
                    self.logger.info(
                        f"Skipping download for {series_id}: start date {start_dt.date()} is after end date {end_date_dt.date()}."
                    )
                    continue

                self.logger.info(f"Downloading {series_id} from {start} to {end_date}")
                series_data = self.fred.get_series(
                    series_id, observation_start=start, observation_end=end_date
                )

                if not series_data.empty:
                    df = series_data.dropna().reset_index()
                    df.columns = ["timestamp", "value"]
                    df["series_id"] = series_id
                    all_data.append(df)
            except Exception as e:
                self.logger.error(f"Download failed for {series_id}: {e}")

        if all_data:
            self._save_to_database(pd.concat(all_data, ignore_index=True))

    def _download_all_indicators(
        self, indicators: Dict[str, str], start: str, end: str
    ) -> pd.DataFrame:
        series = {
            sid: self.fred.get_series(sid, observation_start=start, observation_end=end)
            for sid in indicators
        }
        return pd.DataFrame(series).ffill()

    def validate_data(self, data: pd.DataFrame) -> bool:
        return super().validate_data(data)
