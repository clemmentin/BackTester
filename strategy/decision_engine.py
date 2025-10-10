import logging
import pandas as pd
from typing import Optional, Dict, List
from config.strategy_config import get_strategy_class, CURRENT_STRATEGY
from strategy.alpha.market_detector import MarketState
from strategy.alpha.alpha_engine import AlphaEngine
from strategy.hybrid_dual_alpha_strategy import HybridDualAlphaStrategy
from strategy.risk.risk_manager import RiskManager
from strategy.portfolio_constructor import PortfolioConstructor
from strategy.contracts import FinalTargetPortfolio


class DecisionEngine:
    """
    The master coordinator for the four-layer decision pipeline.
    Orchestrates data flow from raw market data to final target portfolio.
    """

    def __init__(
        self,
        initial_capital: float,
        symbol_list: List[str],
        all_processed_data: pd.DataFrame,
        data_manager: Optional = None,
        **kwargs,
    ):
        """Initialize all four layers of the decision pipeline"""
        self.logger = logging.getLogger(__name__)
        self.all_data = all_processed_data
        self.symbol_list = symbol_list
        self.data_manager = data_manager
        self.alpha_engine = AlphaEngine(**kwargs)
        self._fundamental_cache = None
        self._last_fundamental_update = None

        strategy_name = kwargs.get("strategy_name", CURRENT_STRATEGY)

        StrategyFilterClass = get_strategy_class(strategy_name)

        self.strategy_filter = StrategyFilterClass(symbol_list=symbol_list, **kwargs)

        # Layer 3: Risk Budgeting
        risk_config = kwargs.get("risk_config", {})
        self.risk_manager = RiskManager(
            initial_capital=initial_capital, config_dict=risk_config
        )

        # Layer 4: Portfolio Construction
        self.portfolio_constructor = PortfolioConstructor(**kwargs)

        self.logger.info("DecisionEngine initialized successfully.")

    def run_pipeline(
        self,
        timestamp: pd.Timestamp,
        portfolio_value: float,
        market_data_for_day: pd.DataFrame,
        current_positions: Dict[str, float] = None,
    ) -> FinalTargetPortfolio:
        """
        Executes the full four-layer decision pipeline.
        Returns:
            Final target portfolio (symbol -> weight mapping)
        """
        self.logger.info(f"--- Running Decision Pipeline for {timestamp.date()} ---")

        # Prepare market data
        market_data = self._prepare_market_data(timestamp)
        if market_data is None or market_data.empty:
            return {}

        historical_market_data = self._prepare_market_data(timestamp)
        if historical_market_data is None:
            return {}

        macro_data = self._get_macro_data(timestamp)

        market_state: MarketState = (
            self.alpha_engine.market_detector.detect_market_state(
                historical_market_data, timestamp, symbols=None, macro_data=macro_data
            )
        )

        # === LAYER 1: Alpha Generation ===
        raw_alpha_signals = self.alpha_engine.generate_alpha_signals(
            historical_market_data,
            self.symbol_list,
            timestamp,
            macro_data=macro_data,
            market_state=market_state,
        )

        # === LAYER 2: Strategic Filtering ===
        stratified_pool = self.strategy_filter.select_candidates(
            raw_alpha_signals, market_state, market_data, timestamp
        )

        # === LAYER 3: Risk Budgeting ===
        risk_budget = self.risk_manager.determine_risk_budget(
            portfolio_value, timestamp, market_state=market_state
        )

        # === LAYER 4: Portfolio Construction ===
        target_portfolio = self.portfolio_constructor.build_target_portfolio(
            raw_alpha_signals,
            stratified_pool,
            risk_budget,
            market_state=market_state,
            current_positions=current_positions,
            timestamp=timestamp,
        )

        return target_portfolio

    def _prepare_market_data(self, timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Prepare historical data slice for the pipeline"""
        if self.all_data is None or self.all_data.empty:
            self.logger.error("all_data is not available or empty.")
            return None

            # This lookback should be determined by the longest requirement (GARCH)
        lookback_days = 1200
        start_date = timestamp - pd.Timedelta(days=lookback_days)

        try:
            # Slicing from the full dataset
            historical_data = self.all_data.loc[start_date:timestamp]
            return historical_data
        except Exception as e:
            self.logger.error(
                f"Error preparing historical market data for detector: {e}"
            )
            return None

    def _get_macro_data(self, timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Get macro data up to timestamp.

        NEW METHOD: Extract macro data from data manager or all_data.
        """
        if self.data_manager is None:
            self.logger.debug("No data manager available for macro data")
            return None

        try:
            # Check if all_data has macro columns
            macro_columns = [
                col
                for col in self.all_data.columns
                if col in ["DGS10", "DGS2", "T10Y2Y", "VIXCLS", "UNRATE", "CPIAUCSL"]
            ]

            if not macro_columns:
                self.logger.debug("No macro columns found in all_data")
                return None

            macro_data = self.all_data[macro_columns].groupby("timestamp").first()
            macro_data = macro_data.loc[:timestamp]

            self.logger.debug(
                f"Loaded {len(macro_data)} rows of macro data up to {timestamp.date()}"
            )
            return macro_data

        except Exception as e:
            self.logger.warning(f"Error loading macro data: {e}")
