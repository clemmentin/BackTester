import logging
from typing import Dict, Tuple
import pandas as pd

from strategy.risk.risk_manager import RiskManager
from strategy.risk.stop_loss import StopManager
from strategy.contracts import RawAlphaSignal
import config
from strategy.alpha.alpha_engine import AlphaEngine
from strategy.alpha.market_detector import MarketDetector, MarketRegime
import numpy as np


class DecisionEngine:
    """ """
    def __init__(
        self,
        initial_capital: float,
        symbol_list: list,
        all_processed_data: pd.DataFrame,
        data_manager=None,
        **strategy_kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.symbol_list = symbol_list
        self.all_data = all_processed_data

        # --- Initialize Core Components ---
        self.alpha_engine = AlphaEngine(**strategy_kwargs)
        self.market_detector = (
            self.alpha_engine.market_detector
        )  # Reuse the instance from AlphaEngine

        # --- [NEW] Load Offensive Mode Parameters from WFO ---
        self.min_ev_percentile_filter = strategy_kwargs.get(
            "min_ev_percentile_filter", 0.25
        )
        self.sizing_ev_to_confidence_ratio = strategy_kwargs.get(
            "sizing_ev_to_confidence_ratio", 0.6
        )
        self.bull_market_leverage = strategy_kwargs.get("bull_market_leverage", 1.0)

        # --- Load Base Configuration for Sizing ---
        sizing_config = config.get_trading_param(
            "TRADING_PARAMS", "position_sizing", default={}
        )
        self.base_target_utilization = sizing_config.get("target_utilization", 0.95)
        self.hard_cap_utilization = sizing_config.get(
            "hard_cap_utilization", 0.98
        )  # Max possible utilization

        self.logger.info(
            f"DecisionEngine initialized with offensive params: "
            f"EV_Filter={self.min_ev_percentile_filter:.2f}, "
            f"EV_Sizing_Ratio={self.sizing_ev_to_confidence_ratio:.2f}, "
            f"Bull_Leverage={self.bull_market_leverage:.2f}"
        )

    def run_pipeline(
        self,
        timestamp: pd.Timestamp,
        portfolio_value: float,
        market_data_for_day: pd.DataFrame,
        current_positions: Dict,
    ) -> Tuple[Dict, Dict]:
        """

        Args:
          timestamp: pd.Timestamp: 
          portfolio_value: float: 
          market_data_for_day: pd.DataFrame: 
          current_positions: Dict: 

        Returns:

        """
        diagnostics = {}

        # --- 1. Detect Market State ---
        market_state = self.market_detector.detect_market_state(
            self.all_data, timestamp
        )
        diagnostics["market_state"] = market_state.__dict__

        # --- 2. Generate Raw Alpha Signals ---
        raw_signals_by_source = self.alpha_engine.generate_alpha_signals(
            market_data=self.all_data,
            symbols=self.symbol_list,
            timestamp=timestamp,
            market_state=market_state,
        )

        # For simplicity, we combine all signals into one dictionary for filtering and sizing.
        # In a more complex system, you might handle sources differently.
        all_raw_signals = {
            symbol: signal
            for source_signals in raw_signals_by_source.values()
            for symbol, signal in source_signals.items()
        }
        diagnostics["initial_signal_count"] = len(all_raw_signals)

        # --- 3. [NEW] Filter Signals Based on EV Percentile ---
        if not all_raw_signals:
            return {}, diagnostics

        # Calculate the EV threshold for filtering
        all_evs = [s.expected_value for s in all_raw_signals.values()]
        ev_threshold = np.percentile(all_evs, self.min_ev_percentile_filter * 100)

        # Apply the filter
        filtered_signals = {
            symbol: signal
            for symbol, signal in all_raw_signals.items()
            if signal.expected_value >= ev_threshold
        }
        diagnostics["signals_after_ev_filter"] = len(filtered_signals)

        if not filtered_signals:
            return {}, diagnostics

        # --- 4. [NEW] Determine Dynamic Total Risk Budget ---
        current_target_utilization = self.base_target_utilization

        # Apply bull market leverage
        if market_state.regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
            current_target_utilization *= self.bull_market_leverage

        # Apply macro risk adjustment from MarketDetector
        current_target_utilization *= market_state.transition_adjustment

        # Ensure the final utilization does not exceed the hard cap
        final_target_utilization = min(
            current_target_utilization, self.hard_cap_utilization
        )
        total_risk_budget = portfolio_value * final_target_utilization
        diagnostics["final_target_utilization"] = final_target_utilization

        # --- 5. [NEW] Calculate Final Target Positions ---
        target_portfolio_values = self._calculate_target_positions(
            filtered_signals, total_risk_budget
        )
        diagnostics["final_signals"] = filtered_signals

        return target_portfolio_values, diagnostics

    def _calculate_target_positions(
        self, signals: Dict, total_risk_budget: float
    ) -> Dict:
        """Calculates the target dollar value for each position based on signal strength.

        Args:
          signals: Dict: 
          total_risk_budget: float: 

        Returns:

        """
        if not signals:
            return {}

        # Normalize EV scores for stable sizing calculation
        all_evs = [s.expected_value for s in signals.values()]
        # Simple normalization: divide by sum of absolute values
        sum_abs_ev = sum(abs(ev) for ev in all_evs)
        if sum_abs_ev == 0:
            sum_abs_ev = 1.0

        # Calculate final signal strength using the EV-to-Confidence ratio
        signal_strengths = {}
        for symbol, signal in signals.items():
            normalized_ev = signal.expected_value / sum_abs_ev

            strength = (self.sizing_ev_to_confidence_ratio * normalized_ev) + (
                (1 - self.sizing_ev_to_confidence_ratio) * signal.confidence
            )
            signal_strengths[symbol] = max(
                0, strength
            )  # Ensure strength is not negative

        total_strength = sum(signal_strengths.values())
        if total_strength <= 0:
            return {}

        # Allocate the dynamic risk budget based on relative signal strengths
        target_portfolio_values = {
            symbol: (strength / total_strength) * total_risk_budget
            for symbol, strength in signal_strengths.items()
        }

        return target_portfolio_values

    def _combine_and_filter_signals(
        self, signals_by_source: Dict
    ) -> list[RawAlphaSignal]:
        """A placeholder for logic that combines signals from various alpha sources.
        For this architecture, we assume a simple union of all generated signals.
        In a more complex system, this could involve weighting, conflict resolution, etc.

        Args:
          signals_by_source: Dict: 

        Returns:

        """
        all_signals = []
        for source, signals in signals_by_source.items():
            all_signals.extend(list(signals.values()))

        # Sort by expected value to prioritize the best opportunities
        all_signals.sort(key=lambda s: s.expected_value, reverse=True)

        return all_signals

    def _construct_portfolio(
        self,
        signals: list[RawAlphaSignal],
        risk_budget: "RiskBudget",
        market_state: "MarketState",
    ) -> Dict[str, float]:
        """Constructs a target portfolio from signals and a risk budget.
        This simplified version uses a score-weighted allocation.

        Args:
          signals: list[RawAlphaSignal]: 
          risk_budget: "RiskBudget": 
          market_state: "MarketState": 

        Returns:

        """
        if not signals:
            return {}

        # Use expected_value for weighting if available, otherwise fall back to score
        total_score = sum(
            s.expected_value if s.expected_value > 0 else 0 for s in signals
        )
        if total_score <= 0:
            return {}

        target_portfolio = {}
        # Limit positions by the number specified in the risk budget
        num_positions = min(len(signals), risk_budget.max_position_count)

        # Allocate total target exposure across the top N signals
        for i in range(num_positions):
            signal = signals[i]
            weight = (
                signal.expected_value if signal.expected_value > 0 else 0
            ) / total_score
            target_value = risk_budget.target_portfolio_exposure * weight
            target_portfolio[signal.symbol] = target_value

        return target_portfolio

    def _apply_cash_constraint(
        self,
        ideal_target_portfolio: Dict[str, float],
        current_position_values: Dict[str, float],
        available_cash: float,
    ) -> Dict[str, float]:
        """Adjusts the ideal target portfolio to ensure new buys do not exceed available cash.

        Args:
          ideal_target_portfolio: Dict[str: 
          float]: 
          current_position_values: Dict[str: 
          available_cash: float: 

        Returns:

        """
        cash_needed = 0
        # Calculate the total cash required for all new buys and position increases
        for symbol, target_value in ideal_target_portfolio.items():
            current_value = current_position_values.get(symbol, 0)
            if target_value > current_value:
                cash_needed += target_value - current_value

        # If we need more cash than we have, calculate a scaling factor
        if cash_needed > available_cash and cash_needed > 0:
            # Use a small buffer to account for commissions and slippage
            cash_buffer = 0.99
            # If available_cash is negative, scale_factor will be 0 or negative
            scale_factor = max(0, (available_cash * cash_buffer) / cash_needed)

            self.logger.warning(
                f"Cash constraint hit. Needed ${cash_needed:,.0f} but have ${available_cash:,.0f}. "
                f"Scaling new buys by {scale_factor:.2%}."
            )

            final_target_portfolio = {}
            for symbol, target_value in ideal_target_portfolio.items():
                current_value = current_position_values.get(symbol, 0)

                if target_value > current_value:
                    # This is a new buy or an increase, so apply the scaling
                    investment_increase = (target_value - current_value) * scale_factor
                    final_target_portfolio[symbol] = current_value + investment_increase
                else:
                    # This is a hold or a sell; no new cash is needed, so keep the original target.
                    final_target_portfolio[symbol] = target_value

            return final_target_portfolio
        else:
            # Cash is sufficient, no adjustment needed
            return ideal_target_portfolio
