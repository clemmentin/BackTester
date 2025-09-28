import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import talib
from dotenv import load_dotenv

from data_pipeline.prepare_data import pandas_rsi

from .events import MarketEvent

load_dotenv()
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


class DataHandler(ABC):
    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError("Should implement get_latest_bars")

    @abstractmethod
    def update_bars(self):
        raise NotImplementedError("Should implement update_bars")


class HistoricCSVDataHandler(DataHandler):
    def __init__(self, events_queue, csv_dir, symbol_list, signal_csv_path):
        self.events = events_queue
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.signal_csv_path = signal_csv_path
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.signal_data = self.load_signal_data()
        self._open_and_load_csv_files()

    def load_signal_data(self):
        try:
            signals = pd.read_csv(
                self.signal_csv_path, header=0, index_col=0, parse_dates=True
            )
            signals.rename(
                columns={"annualized_volatility_pct": "vol_forecast"}, inplace=True
            )
            return signals
        except FileNotFoundError:
            print(
                f"WARNING: Signal file not found at {self.signal_csv_path}. Proceeding without signals."
            )
            return None

    def _open_and_load_csv_files(self):
        for s in self.symbol_list:
            file_path = f"{self.csv_dir}/{s}.csv"
            price_data = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
            price_data["sma200"] = price_data["Close"].rolling(window=200).mean()
            price_data["rsi"] = pandas_rsi(price_data["Close"], timeperiod=14)
            if self.signal_data is not None:
                combined_data = price_data.merge(
                    self.signal_data, left_index=True, right_index=True, how="left"
                )
                combined_data["vol_forecast"] = combined_data["vol_forecast"].ffill()
                combined_data.dropna(inplace=True)
            else:
                combined_data = price_data
                combined_data.dropna(inplace=True)
            self.symbol_data[s] = combined_data.iterrows()

    def update_bars(self):
        for s in self.symbol_list:
            try:
                timestamp, row = next(self.symbol_data[s])
            except StopIteration:
                self.continue_backtest = False
            else:
                bar = {
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"],
                    "vol_forecast": row.get("vol_forecast", 0),
                    "sma200": row.get("sma200", None),
                    "rsi": row.get("rsi", None),
                }
                self.latest_symbol_data[s] = bar
                market_event = MarketEvent(timestamp=timestamp, symbol=s, data=bar)
                self.events.put(market_event)

    def get_latest_bars(self, symbol, N=1):
        try:
            return [self.latest_symbol_data[symbol]]
        except KeyError:
            print(f"Data for symbol {symbol} is not available yet")
            return []


class HistoricDBDataHandler(DataHandler):
    def __init__(self, events_queue, symbol_list, all_processed_data, start_date=None):
        self.events = events_queue
        self.symbol_list = symbol_list
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.all_data = None
        self.timeline = None
        self.current_time_index = 0
        self._load_preprocessed_data(all_processed_data, start_date)
        self.enable_lookahead_check = False

    def _validate_vol_forecast(self, symbol, current_timestamp, row):
        if not self.enable_lookahead_check:
            return row.get("vol_forecast", 0)

        # Check if we have forecast date information
        if "vol_forecast_date" in row.index:
            forecast_date = row.get("vol_forecast_date")
            if pd.notna(forecast_date) and forecast_date >= current_timestamp:
                logging.warning(
                    f"LOOKAHEAD WARNING: {symbol} at {current_timestamp.date()} "
                    f"using forecast made on {forecast_date.date()}"
                )
                # Return a default volatility instead of biased forecast
                return 20.0  # Default 20% annual volatility

        return row.get("vol_forecast", 20.0)

    def _load_preprocessed_data(self, all_processed_data, start_date=None):
        print("DATA_HANDLER: Loading pre-processed data...")
        if all_processed_data is None or all_processed_data.empty:
            print("DATA_HANDLER ERROR: Received empty or None DataFrame.")
            self.continue_backtest = False
            return
        try:
            self.all_data = all_processed_data
            self.timeline = sorted(
                self.all_data.index.get_level_values("timestamp").unique()
            )
            if start_date:
                start_datetime = pd.to_datetime(start_date)
                start_index = np.searchsorted(
                    self.timeline, start_datetime, side="left"
                )
                if start_index < len(self.timeline):
                    self.current_time_index = start_index
                    effective_start_date = self.timeline[start_index].date()
                    print(
                        f"DATA_HANDLER: Timeline set to start on {effective_start_date}."
                    )
                else:
                    print(
                        f"DATA_HANDLER WARNING: Start date {start_date} is after the last data point."
                    )
                    self.continue_backtest = False
            print("DATA_HANDLER: Pre-processed data loaded successfully.")
        except Exception as e:
            print(f"DATA_HANDLER ERROR: Failed to load pre-processed data. {e}")
            self.continue_backtest = False

    def update_bars(self):
        if self.current_time_index >= len(self.timeline):
            self.continue_backtest = False
            return

        current_timestamp = self.timeline[self.current_time_index]

        try:
            todays_data = self.all_data.xs(current_timestamp, level="timestamp")
        except KeyError:
            self.current_time_index += 1
            return

        for symbol, row in todays_data.iterrows():
            validated_vol = self._validate_vol_forecast(symbol, current_timestamp, row)

            bar = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "sma200": row.get("sma200", 0),
                "sma50": row.get("sma50", 0),
                "vol_forecast": validated_vol,  # Use validated value
                "rsi": row.get("rsi", 0),
                "DGS10": row.get("DGS10", 0),
                "DGS10_MA60": row.get("DGS10_MA60", 0),
            }
            if self.current_time_index % 100 == 0 and symbol == self.symbol_list[0]:
                logging.debug(
                    f"Data point {self.current_time_index}: {current_timestamp.date()}, "
                    f"vol_forecast={validated_vol:.2f}%"
                )

            self.latest_symbol_data[symbol] = bar
            market_event = MarketEvent(
                timestamp=current_timestamp, symbol=symbol, data=bar
            )
            self.events.put(market_event)

        self.current_time_index += 1

    def get_latest_bars(self, symbol, N=1):
        try:
            return [self.latest_symbol_data[symbol]]
        except KeyError:
            return []
