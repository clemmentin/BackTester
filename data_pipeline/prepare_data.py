import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


def _calculate_essential_features_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
      group: pd.DataFrame: 

    Returns:

    """
    group = group.sort_index()

    close = group["close"].astype(float)
    volume = group["volume"].astype(float)
    high = group["high"].astype(float)  # ATR needs high price
    low = group["low"].astype(float)  # ATR needs low price

    # === CRITICAL: Returns calculation (all factors need this) ===
    group["returns_1d"] = close.pct_change(1)
    group["returns_5d"] = close.pct_change(5)
    group["returns_20d"] = close.pct_change(20)
    group["returns_60d"] = close.pct_change(60)

    # === Volume features (for liquidity filtering) ===
    group["volume_sma_20"] = volume.rolling(window=20, min_periods=10).mean()
    group["volume_ratio"] = volume / group["volume_sma_20"]

    # === Optional: Moving averages (useful for trend analysis) ===
    group["sma_20"] = close.rolling(window=20, min_periods=10).mean()
    group["sma_50"] = close.rolling(window=50, min_periods=25).mean()
    group["sma_200"] = close.rolling(window=200, min_periods=100).mean()

    # === ATR (for slippage model) ===
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    # Calculate True Range (TR)
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR with a 14-period Wilder's smoothing
    group["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    return group.fillna(method="ffill")


class TechnicalIndicatorCalculator:
    """ """
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def calculate_all_indicators(
        self, price_data: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """

        Args:
          price_data: pd.DataFrame: 
          macro_data: Optional[pd.DataFrame]:  (Default value = None)

        Returns:

        """
        if not isinstance(price_data.index, pd.MultiIndex):
            self.logger.error("Price data must have MultiIndex ('timestamp', 'symbol')")
            return price_data

        try:
            # Step 1: Essential price features
            data_with_features = self._calculate_price_features(price_data)

            # Step 2: Volatility measures (REQUIRED for Price factor)
            data_with_vol = self._calculate_volatility_measures(data_with_features)

            # Step 3: Optional macro merge (currently unused but preserved)
            if macro_data is not None and not macro_data.empty:
                self.logger.info("Macro data provided, merging...")
                final_data = self._merge_macro_features(data_with_vol, macro_data)
            else:
                self.logger.info("No macro data (regime will use price only)")
                final_data = data_with_vol

            # Cleanup
            final_data.replace([np.inf, -np.inf], np.nan, inplace=True)

            self.logger.info(
                f"Feature engineering complete: {len(final_data.columns)} columns"
            )

            return final_data

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}", exc_info=True)
            return price_data

    def _calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
          data: pd.DataFrame: 

        Returns:

        """
        self.logger.info("Calculating essential price features...")

        data = data.sort_index()
        all_symbols = sorted(data.index.get_level_values("symbol").unique())

        processed_groups = []

        for symbol in all_symbols:
            group_data = data.xs(symbol, level="symbol")
            processed_group = _calculate_essential_features_for_group(group_data)

            processed_group["symbol"] = symbol
            processed_group.set_index("symbol", append=True, inplace=True)
            processed_group = processed_group.reorder_levels(["timestamp", "symbol"])

            processed_groups.append(processed_group)

        result_data = pd.concat(processed_groups)

        self.logger.info(
            f"Essential features calculated for {len(all_symbols)} symbols"
        )

        return result_data

    def _calculate_volatility_measures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility (REQUIRED for Price factor's low-vol sub-factor).

        Args:
          data: pd.DataFrame: 

        Returns:

        """
        self.logger.info("Calculating volatility measures (required for Price factor)")

        if "returns_1d" not in data.columns:
            self.logger.warning("'returns_1d' missing, volatility calculation skipped")
            return data

        # Annualized volatility at multiple windows
        vol_10d = data.groupby("symbol")["returns_1d"].rolling(
            window=10, min_periods=5
        ).std() * np.sqrt(252)

        vol_20d = data.groupby("symbol")["returns_1d"].rolling(
            window=20, min_periods=10
        ).std() * np.sqrt(252)

        vol_60d = data.groupby("symbol")["returns_1d"].rolling(
            window=60, min_periods=30
        ).std() * np.sqrt(252)

        data["volatility_10d"] = vol_10d.reset_index(level=0, drop=True)
        data["volatility_20d"] = vol_20d.reset_index(level=0, drop=True)
        data["volatility_60d"] = vol_60d.reset_index(level=0, drop=True)

        self.logger.info("Volatility calculation complete")

        return data

    def _merge_macro_features(
        self, price_data: pd.DataFrame, macro_data: pd.DataFrame
    ) -> pd.DataFrame:
        """

        Args:
          price_data: pd.DataFrame: 
          macro_data: pd.DataFrame: 

        Returns:

        """
        self.logger.info("Merging macro features (optional for regime enhancement)")

        # Apply publication delays
        macro_delayed = self._apply_macro_delays(macro_data)

        # Fix index name if needed
        if macro_delayed.index.name != "timestamp":
            self.logger.debug(
                f"Fixing macro index name from '{macro_delayed.index.name}' to 'timestamp'"
            )
            macro_delayed.index.name = "timestamp"

        # Prepare for merge_asof
        macro_delayed_reset = macro_delayed.reset_index()
        price_data_reset = price_data.reset_index()
        price_data_sorted = price_data_reset.sort_values("timestamp")

        # Point-in-time join
        combined = pd.merge_asof(
            left=price_data_sorted,
            right=macro_delayed_reset,
            on="timestamp",
            direction="backward",
        )

        # Restore MultiIndex
        combined = combined.set_index(["timestamp", "symbol"]).sort_index()

        # Derived macro features
        if "DGS10" in combined.columns and "DGS2" in combined.columns:
            combined["macro_10y2y_spread"] = combined["DGS10"] - combined["DGS2"]
            self.logger.debug("Created yield curve spread feature")

        if "VIXCLS" in combined.columns:
            combined["macro_vix_ma20"] = (
                combined.groupby(level="symbol")["VIXCLS"]
                .rolling(window=20)
                .mean()
                .reset_index(level=0, drop=True)
            )
            self.logger.debug("Created VIX moving average feature")

        self.logger.info(
            f"Macro merge complete: added {len([c for c in combined.columns if 'macro_' in c or c in macro_data.columns])} macro features"
        )

        return combined

    def _apply_macro_delays(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Apply realistic publication delays to macro indicators.

        Args:
          macro_data: pd.DataFrame: 

        Returns:

        """
        delays = {
            "CPIAUCSL": 15,  # CPI: ~2 weeks after month end
            "UNRATE": 5,  # Unemployment: first Friday of month
            "PAYEMS": 5,  # Payrolls: first Friday of month
            "GDP": 45,  # GDP: ~45 days after quarter end
        }

        delayed = macro_data.copy()
        applied_delays = []

        for col, delay in delays.items():
            if col in delayed.columns:
                delayed[col] = delayed[col].shift(delay)
                applied_delays.append(f"{col}({delay}d)")

        if applied_delays:
            self.logger.debug(
                f"Applied publication delays: {', '.join(applied_delays)}"
            )

        return delayed.ffill()


def prepare_features_for_backtest(
    price_data: pd.DataFrame,
    macro_data: Optional[pd.DataFrame] = None,
    config: Dict = None,
) -> pd.DataFrame:
    """

    Args:
      price_data: pd.DataFrame: 
      macro_data: Optional[pd.DataFrame]:  (Default value = None)
      config: Dict:  (Default value = None)

    Returns:

    """
    calculator = TechnicalIndicatorCalculator(config)
    result = calculator.calculate_all_indicators(price_data, macro_data)

    logger.info(
        f"Feature preparation complete: {result.shape[0]} rows, "
        f"{result.shape[1]} columns"
    )

    return result
