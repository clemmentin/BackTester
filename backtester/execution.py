from .events import FillEvent
import logging
import numpy as np


class SimulatedExecutionHandler:
    """ """
    def __init__(self, events_queue, data_handler, transaction_costs_config=None):
        self.events = events_queue
        self.data_handler = data_handler
        self.logger = logging.getLogger(__name__)

        # Load transaction cost parameters from config
        if transaction_costs_config is None:
            transaction_costs_config = {}
        self.enable_slippage_model = transaction_costs_config.get(
            "enable_volume_slippage_model", False
        )
        self.slippage_k = transaction_costs_config.get("slippage_model_k", 0.1)
        self.fallback_slippage_rate = transaction_costs_config.get(
            "slippage_rate", 0.0005
        )
        self.commission_rate = transaction_costs_config.get("commission_rate", 0.001)
        self.min_commission = transaction_costs_config.get("min_commission", 1.0)

    def execute_order(self, event):
        """

        Args:
          event: 

        Returns:

        """
        if event.type != "Order":
            return

        latest_bar = self.data_handler.get_latest_bars(event.symbol)
        if not latest_bar or not latest_bar[0] or latest_bar[0]["close"] <= 0:
            self.logger.error(
                f"Could not get valid bar for {event.symbol} on {event.timestamp.date()}. Order cancelled."
            )
            return

        bar_data = latest_bar[0]
        close_price = bar_data["close"]
        slippage_per_share = 0.0

        if self.enable_slippage_model:
            atr = bar_data.get("atr_14", 0)
            adtv = bar_data.get(
                "volume_sma_20", 0
            )  # Use 20-day SMA of volume as ADTV proxy

            # Calculate slippage only if ATR and ADTV data are valid
            if atr > 0 and adtv > 0 and close_price > 0:
                # Slippage model: K * ATR * (Order Size / ADTV) ^ 0.5
                # Using sqrt for a sub-linear market impact
                market_impact_ratio = event.quantity / adtv
                slippage_per_share = (
                    self.slippage_k * atr * np.sqrt(market_impact_ratio)
                )
            else:
                # Fallback to a fixed slippage rate if data is missing
                slippage_per_share = close_price * self.fallback_slippage_rate
        else:
            # Use fixed slippage rate if the dynamic model is disabled
            slippage_per_share = close_price * self.fallback_slippage_rate

        # Apply slippage to the fill price based on trade direction
        if event.direction == "Buy":
            fill_price = close_price + slippage_per_share
        elif event.direction == "Sell":
            fill_price = close_price - slippage_per_share
        else:
            fill_price = close_price  # Should not happen in this system

        fill_cost = fill_price * event.quantity
        commission = self.calculate_commission(fill_cost)

        # Restore and enhance previous print format
        print(f"[Order] {event.symbol}:")
        print(f"  Direction: {event.direction}")
        print(f"  Amount: {event.quantity}")
        print(f"  Close Price: ${close_price:.2f}")
        print(f"  Slippage/Share: ${slippage_per_share:.4f}")
        print(f"  Fill Price (w/ slippage): ${fill_price:.2f}")
        print(f"  Cost (w/ slippage): ${fill_cost:,.2f}")
        print(f"  Commission: ${commission:,.2f}")

        fill_event = FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            quantity=event.quantity,
            direction=event.direction,
            fill_cost=fill_cost,
            commission=commission,
        )

        print(
            f"EXECUTION: Order for {event.quantity} {event.symbol} {event.direction} filled at ${fill_price:.2f}."
        )
        self.events.put(fill_event)

    def calculate_commission(self, fill_cost):
        """

        Args:
          fill_cost: 

        Returns:

        """
        return max(self.min_commission, fill_cost * self.commission_rate)
