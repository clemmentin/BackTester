import logging
from abc import ABC, abstractmethod


class BaseStrategy(ABC):

    def __init__(self, events_queue, symbol_list, **kwargs):
        self.events = events_queue
        self.symbol_list = symbol_list
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def calculate_signal(self, event):
        """
        Generate trading signals based on market events
        Must be implemented by all strategy subclasses
        """
        raise NotImplementedError("Should implement calculate_signal()")

    def set_portfolio_reference(self, portfolio):
        """
        Set reference to portfolio for advanced strategies
        """
        self.portfolio_ref = portfolio
