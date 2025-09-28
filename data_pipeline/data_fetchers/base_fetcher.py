# data_pipeline/data_fetchers/base_fetcher.py

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine


class BaseDataFetcher(ABC):
    def __init__(self, cache_dir: Path, db_config: Dict[str, Any]):
        self.cache_dir = cache_dir
        self.db_config = db_config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create a subdirectory within the main cache for this specific fetcher
        self.fetcher_cache_dir = cache_dir / self.data_type
        self.fetcher_cache_dir.mkdir(exist_ok=True, parents=True)

        # Default cache validity in hours
        self.cache_validity_hours = 24

        # MODIFIED: Conditionally initialize the database engine
        self.engine = None
        if self._uses_database():
            if self.db_config and all(self.db_config.values()):
                try:
                    self.engine = self._create_engine()
                    self.logger.info("Database engine created successfully.")
                except Exception as e:
                    self.logger.error(f"Failed to create database engine: {e}")
            else:
                self.logger.warning(
                    "Database configuration is incomplete. Database features will be disabled."
                )

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return the type of data this fetcher handles (e.g., 'prices', 'macro')."""
        pass

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """Main method to fetch data, to be implemented by subclasses."""
        pass

    # NEW METHOD: Subclasses will override this to indicate if they use the database
    def _uses_database(self) -> bool:
        """
        Indicates whether this fetcher uses a database connection.
        Subclasses that need a DB should override this and return True.
        """
        return False

    # --- Database Methods (now available to all subclasses) ---

    def _create_engine(self):
        """Create SQLAlchemy engine from db_config."""
        db_url = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        return create_engine(db_url, pool_pre_ping=True, pool_size=5)

    # --- Cache Methods ---

    def get_cache_path(self, identifier: str) -> Path:
        """Get the cache file path for a specific identifier (e.g., a symbol)."""
        return self.fetcher_cache_dir / f"{identifier}.parquet"

    def load_from_cache(self, identifier: str) -> Optional[pd.DataFrame]:
        """Load data from cache if it exists and is not stale."""
        cache_file = self.get_cache_path(identifier)
        if cache_file.exists():
            try:
                file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - file_mod_time > timedelta(
                    hours=self.cache_validity_hours
                ):
                    self.logger.info(f"Cache for {identifier} is stale. Will re-fetch.")
                    return None

                self.logger.debug(f"Loading {identifier} from cache.")
                return pd.read_parquet(cache_file)
            except Exception as e:
                self.logger.warning(
                    f"Cache read for {identifier} failed: {e}. Removing corrupted cache."
                )
                cache_file.unlink()
        return None

    def save_to_cache(self, data: pd.DataFrame, identifier: str):
        """Save data to a Parquet cache file."""
        if data.empty:
            self.logger.debug(f"Data for {identifier} is empty, skipping cache save.")
            return

        try:
            cache_file = self.get_cache_path(identifier)
            data.to_parquet(cache_file)
            self.logger.debug(f"Cached data for {identifier} to {cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save cache for {identifier}: {e}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Default validation: checks if DataFrame is not empty."""
        if data is None or data.empty:
            return False
        return True
