import logging
import numpy as np
from collections import deque
import config
from typing import Dict, Set, Optional
import pandas as pd

from strategy.contracts import RawAlphaSignalDict, StratifiedCandidatePool
from strategy.alpha import MarketState
from strategy.alpha.market_detector import MacroRegime
from strategy.signal_quality_filter import SignalQualityFilter


class ConservativeModule:
    """Conservative selector focused on defensive factors."""

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        module_params = kwargs.get("conservative_module", {})
        self.pos_mgmt_config = config.get_trading_param(
            "TRADING_PARAMS", "position_management"
        )
        self.min_confidence = module_params.get("conservative_min_confidence", 0.45)

    def select_candidates(
        self, market_state: MarketState, alpha_signals: RawAlphaSignalDict
    ) -> Set[str]:
        regime = market_state.regime.value
        total_max_pos = self.pos_mgmt_config.get("max_total_positions", 20)
        allocations = self.pos_mgmt_config.get("regime_allocation_pct", {}).get(
            regime, {}
        )
        conservative_pct = allocations.get("conservative", 0.2)
        max_positions = int(np.ceil(total_max_pos * conservative_pct))

        valid_signals = {
            symbol: signal.score
            for symbol, signal in alpha_signals.items()
            if signal.confidence >= self.min_confidence
        }

        if not valid_signals:
            return set()

        sorted_candidates = sorted(
            valid_signals.items(), key=lambda item: item[1], reverse=True
        )

        return {symbol for symbol, score in sorted_candidates[:max_positions]}


class AggressiveModule:
    """Aggressive selector focused on growth factors."""

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        module_params = kwargs.get("aggressive_module", {})
        self.pos_mgmt_config = config.get_trading_param(
            "TRADING_PARAMS", "position_management"
        )
        self.min_confidence = module_params.get("aggressive_min_confidence", 0.30)

    def select_candidates(
        self, market_state: MarketState, alpha_signals: RawAlphaSignalDict
    ) -> Set[str]:
        regime = market_state.regime.value

        total_max_pos = self.pos_mgmt_config.get("max_total_positions", 20)
        allocations = self.pos_mgmt_config.get("regime_allocation_pct", {}).get(
            regime, {}
        )
        aggressive_pct = allocations.get("aggressive", 0.8)
        max_positions = int(np.ceil(total_max_pos * aggressive_pct))

        valid_signals = {
            symbol: signal.score
            for symbol, signal in alpha_signals.items()
            if signal.confidence >= self.min_confidence
        }

        if not valid_signals:
            return set()

        sorted_candidates = sorted(
            valid_signals.items(), key=lambda item: item[1], reverse=True
        )

        return {symbol for symbol, score in sorted_candidates[:max_positions]}


class HybridDualAlphaStrategy:
    """
    Strategy filter (Layer 2) with macro-enhanced market detection.

    NEW: Passes macro_data to market detector for enhanced regime detection
    """

    def __init__(self, symbol_list, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Initialize modules
        self.conservative_module = ConservativeModule(**kwargs)
        self.aggressive_module = AggressiveModule(**kwargs)

        # Initialize signal quality filter
        self.signal_quality_filter = SignalQualityFilter(**kwargs)
        self.logger.info("Signal quality filter initialized in Layer 2")

        # Macro allocation map (keyed by MacroRegime)
        self.macro_allocation_map = {
            MacroRegime.OFFENSE: {"conservative": 0.20, "aggressive": 0.80},
            MacroRegime.DEFENSE: {"conservative": 0.80, "aggressive": 0.20},
            MacroRegime.SIDEWAYS: {"conservative": 0.60, "aggressive": 0.40},
        }

        self.logger.info(
            "HybridDualAlphaStrategy initialized with Macro Regime allocation"
        )

    def select_candidates(
        self,
        raw_alpha_signals: RawAlphaSignalDict,
        market_state: MarketState,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> StratifiedCandidatePool:
        """
        Stratify candidates into conservative and aggressive pools.

        NEW: macro_data parameter is now available (passed from backtester)
        Note: market_state already contains macro-enhanced regime detection
        """
        # Step 0: Apply signal quality filter
        filtered_signals, filter_stats = self.signal_quality_filter.filter_signals(
            raw_alpha_signals, regime=market_state.regime.value
        )

        self.logger.info(
            f"Quality filter: {len(filtered_signals)}/{len(raw_alpha_signals)} signals passed"
        )

        # Step 1: Module selection with filtered signals
        conservative_candidates = self.conservative_module.select_candidates(
            market_state, filtered_signals
        )
        aggressive_candidates = self.aggressive_module.select_candidates(
            market_state, filtered_signals
        )

        # Step 2: Get allocation ratios from macro regime
        macro_regime = market_state.macro_regime
        allocation_ratios = self.macro_allocation_map.get(
            macro_regime, self.macro_allocation_map[MacroRegime.SIDEWAYS]
        )

        # Step 3: Create stratified pool
        pool = StratifiedCandidatePool(
            conservative_candidates=conservative_candidates,
            aggressive_candidates=aggressive_candidates,
            allocation_ratio=allocation_ratios,
        )

        # NEW: Log macro signals if available
        if market_state.macro_signals:
            macro_warnings = []
            if market_state.macro_signals.get("recession_risk", 0) > 0.5:
                macro_warnings.append(
                    f"RecessionRisk={market_state.macro_signals['recession_risk']:.2f}"
                )
            if market_state.macro_signals.get("crisis_warning", 0) > 0.5:
                macro_warnings.append(
                    f"CrisisWarning={market_state.macro_signals['crisis_warning']:.2f}"
                )

            if macro_warnings:
                self.logger.warning(
                    f"Macro Warnings Active: {', '.join(macro_warnings)}"
                )

        self.logger.info(
            f"Macro Regime: {macro_regime.value.upper()}. "
            f"Allocation: C={allocation_ratios['conservative']:.0%}/A={allocation_ratios['aggressive']:.0%}. "
            f"Candidates: Cons={len(conservative_candidates)}, Agg={len(aggressive_candidates)}"
        )

        return pool
