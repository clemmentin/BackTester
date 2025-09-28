# data_pipeline/prepare_data.py

import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib
from arch import arch_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


def pandas_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_indicators_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all technical indicators for a single symbol's DataFrame.
    This function is designed to be used with pandas' groupby().apply().
    """
    # Ensure data is sorted by time
    group = group.sort_index()

    # Extract series for calculations
    close = group["close"].astype(float)
    high = group["high"].astype(float)
    low = group["low"].astype(float)
    volume = group["volume"].astype(float)
    group["sma_20"] = close.rolling(window=20, min_periods=10).mean()
    group["sma_50"] = close.rolling(window=50, min_periods=25).mean()
    group["sma_200"] = close.rolling(window=200, min_periods=100).mean()
    group["ema_12"] = close.ewm(span=12, adjust=False).mean()
    group["ema_26"] = close.ewm(span=26, adjust=False).mean()

    if len(close) >= 14:
        # group['rsi_14'] = talib.RSI(close.values, timeperiod=14)
        group["rsi_14"] = pandas_rsi(close, period=14)
    if len(close) >= 26:
        macd, macd_signal, macd_hist = talib.MACD(
            close.values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        group["macd"] = macd
        group["macd_signal"] = macd_signal
        group["macd_hist"] = macd_hist
    if len(high) >= 14:
        slowk, slowd = talib.STOCH(
            high.values,
            low.values,
            close.values,
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        group["stoch_k"] = slowk
        group["stoch_d"] = slowd

    # --- Volatility Indicators ---
    if len(close) >= 20:
        upper, middle, lower = talib.BBANDS(
            close.values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        group["bb_upper"] = upper
        group["bb_middle"] = middle
        group["bb_lower"] = lower
        group["bb_width"] = (upper - lower) / middle
    if len(high) >= 14:
        group["atr_14"] = talib.ATR(
            high.values, low.values, close.values, timeperiod=14
        )

    # --- Price & Volume Features ---
    group["returns_1d"] = close.pct_change(1)
    group["returns_5d"] = close.pct_change(5)
    group["returns_20d"] = close.pct_change(20)
    group["volume_sma_20"] = volume.rolling(window=20, min_periods=10).mean()
    group["volume_ratio"] = volume / group["volume_sma_20"]

    return group.fillna(method="ffill")


class TechnicalIndicatorCalculator:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def calculate_all_indicators(
        self, price_data: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        self.logger.info("Starting comprehensive feature engineering pipeline.")

        if not isinstance(price_data.index, pd.MultiIndex):
            self.logger.error(
                "Price data must have a MultiIndex ('timestamp', 'symbol'). Aborting."
            )
            return price_data

        try:
            # Step 1: Calculate price-based technical indicators
            data_with_ta = self._calculate_price_indicators(price_data)

            # Step 2: Calculate volatility measures (can also be grouped)
            data_with_vol = self._calculate_volatility_measures(data_with_ta)

            # Step 3: Merge macro features
            if macro_data is not None and not macro_data.empty:
                final_data = self._merge_macro_features(data_with_vol, macro_data)
            else:
                self.logger.info("No macro data provided, skipping merge.")
                final_data = data_with_vol

            # Final cleanup
            final_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.logger.info(
                f"Final feature set contains {len(final_data.columns)} columns."
            )

            return final_data

        except Exception as e:
            self.logger.critical(
                f"A critical error occurred during feature engineering: {e}",
                exc_info=True,
            )
            return price_data  # Return original data on failure

    def _calculate_price_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates standard price-based technical indicators using a groupby approach.
        MODIFIED: This now uses groupby().apply() for cleaner and potentially faster execution.
        """
        data = data.sort_index()

        all_symbols = sorted(data.index.get_level_values("symbol").unique())

        processed_groups = []

        for symbol in all_symbols:
            group_data = data.xs(symbol, level="symbol")

            processed_group = _calculate_indicators_for_group(group_data)

            processed_group["symbol"] = symbol
            processed_group.set_index("symbol", append=True, inplace=True)
            processed_group = processed_group.reorder_levels(
                ["timestamp", "symbol"]
            )

            processed_groups.append(processed_group)

        result_data = pd.concat(processed_groups)

        self.logger.info("Technical indicator calculation complete (sequential mode).")
        return result_data

    def _calculate_volatility_measures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical volatility for different windows."""
        self.logger.info("Calculating historical volatility measures.")

        if "returns_1d" not in data.columns:
            self.logger.warning(
                "'returns_1d' not found, volatility calculation will be skipped."
            )
            return data

        # Group by symbol and calculate rolling volatility
        vol_10d = data.groupby("symbol")["returns_1d"].rolling(
            window=10, min_periods=5
        ).std() * np.sqrt(252)
        vol_20d = data.groupby("symbol")["returns_1d"].rolling(
            window=20, min_periods=10
        ).std() * np.sqrt(252)
        vol_60d = data.groupby("symbol")["returns_1d"].rolling(
            window=60, min_periods=30
        ).std() * np.sqrt(252)

        # Reset index to align with the main dataframe
        data["volatility_10d"] = vol_10d.reset_index(level=0, drop=True)
        data["volatility_20d"] = vol_20d.reset_index(level=0, drop=True)
        data["volatility_60d"] = vol_60d.reset_index(level=0, drop=True)

        return data

    def _merge_macro_features(
        self, price_data: pd.DataFrame, macro_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges macro data into the price data.
        This approach uses pd.merge_asof for precise point-in-time correctness.
        """
        self.logger.info("Merging macro economic features.")

        # Apply realistic publication delays to macro data first
        macro_delayed = self._apply_macro_delays(macro_data)

        # Prepare price_data for merge (needs a single sorted timestamp index)
        price_data_reset = price_data.reset_index()
        price_data_sorted = price_data_reset.sort_values("timestamp")

        # Use merge_asof for a robust, point-in-time join
        # This is more accurate than reindex + ffill
        combined = pd.merge_asof(
            left=price_data_sorted,
            right=macro_delayed,
            on="timestamp",
            direction="backward",  # Use last known macro value
        )

        # Restore the original MultiIndex
        combined = combined.set_index(["timestamp", "symbol"]).sort_index()

        # Add a few derived macro features
        if "DGS10" in combined.columns and "DGS2" in combined.columns:
            combined["macro_10y2y_spread"] = combined["DGS10"] - combined["DGS2"]
        if "VIXCLS" in combined.columns:
            combined["macro_vix_ma20"] = (
                combined.groupby(level="symbol")["VIXCLS"]
                .rolling(window=20)
                .mean()
                .reset_index(level=0, drop=True)
            )

        self.logger.info("Macro feature merging complete.")
        return combined

    def _apply_macro_delays(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Applies realistic publication delays to macro data."""
        delays = {"CPIAUCSL": 15, "UNRATE": 5, "PAYEMS": 5, "GDP": 45}  # in days
        delayed = macro_data.copy()
        for col, delay in delays.items():
            if col in delayed.columns:
                delayed[col] = delayed[col].shift(delay)
        return delayed.ffill()


def prepare_features_for_backtest(
    price_data: pd.DataFrame,
    macro_data: Optional[pd.DataFrame] = None,
    config: Dict = None,
) -> pd.DataFrame:
    """
    Main entry point function to run the feature engineering pipeline.
    """
    logger.info("Preparing features for backtest...")
    calculator = TechnicalIndicatorCalculator(config)
    result = calculator.calculate_all_indicators(price_data, macro_data)
    logger.info("Feature preparation complete.")
    return result
