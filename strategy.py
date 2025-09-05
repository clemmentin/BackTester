from abc import ABC, abstractmethod
from events import SignalEvent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='backtest_log.log',
    filemode='w'
)
class Strategy(ABC):
    @abstractmethod
    def calculate_signal(self,event):
        raise NotImplementedError('Should implement calculate_signals()')

class DummyStrategy(Strategy):
    def __init__(self, events_queue, symbol_list):
        self.events = events_queue
        self.symbol_list = symbol_list
        self.ticks = {s:0 for s in self.symbol_list}
        self.positions = {s: {'bought': False, 'entry_Price': 0.0} for s in self.symbol_list}
        self.profit_target = 0.05
        self.stop_loss = 0.02
    def calculate_signal(self,event):
        if event.type == 'Market':
            symbol = event.symbol
            current_price = event.data['close']
            if not self.positions[symbol]['bought']:
                self.ticks[symbol] += 1
                if self.ticks[symbol] == 10:
                    signal = SignalEvent(timestamp=event.timestamp,
                                         symbol = symbol,
                                         signal_type = 'Long')
                    self.events.put(signal)
                    self.positions[symbol]['bought'] = True
                    self.positions[symbol]['entry_price'] = current_price
                    print(f"STRATEGY: Generate LONG signal for {symbol} at reference price {current_price:.2f}")
                    self.ticks[symbol] = 0
            elif self.positions[symbol]['bought']:
                entry_price = self.positions[symbol]['entry_price']
                profit_target_price = entry_price * (1 + self.profit_target)
                stop_loss_price = entry_price * (1 - self.stop_loss)
                if current_price >= profit_target_price:
                    signal = SignalEvent(timestamp=event.timestamp,
                                         symbol=symbol,
                                         signal_type='Exit')
                    self.events.put(signal)
                    self.positions[symbol]['bought'] = False
                    self.positions[symbol]['entry_price'] = 0.0
                    print(f"STRATEGY: Take Profit! Sell {symbol} at {current_price:.2f} (Entry: {entry_price:.2f})")
                elif current_price <= stop_loss_price:
                    signal = SignalEvent(timestamp=event.timestamp,
                                         symbol=symbol,
                                         signal_type='Exit')
                    self.events.put(signal)
                    self.positions[symbol]['bought'] = False
                    self.positions[symbol]['entry_price'] = 0.0
                    print(f"STRATEGY: Stop Loss! Sell {symbol} at {current_price:.2f} (Entry: {entry_price:.2f})")

class VolatilityTargetingStrategy(Strategy):
    def __init__(self, events_queue, symbol_list, buy_threshold, sell_threshold):
        self.events = events_queue
        self.symbol_list = symbol_list
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position = {s: {'bought': False, 'entryPrice': 0.0} for s in self.symbol_list}
        self.stop_loss = 0.05
    def calculate_signal(self, event):
        if event.type == 'Market':
            symbol = event.symbol
            vol_forecast = event.data.get('vol_forecast', None)
            current_price = event.data.get('close')
            sma200 = event.data.get('sma200', None)
            if vol_forecast is None or sma200 is None or current_price is None:
                return
            is_bought = self.position[symbol]['bought']
            entryPrice = self.position[symbol]['entryPrice']
            up = current_price > sma200
            if vol_forecast < self.buy_threshold and up and not is_bought:
                signal = SignalEvent(timestamp=event.timestamp,
                                     symbol=symbol,
                                     signal_type='Long')
                self.events.put(signal)
                self.position[symbol]['bought'] = True
                self.position[symbol]['entryPrice'] = current_price
                logging.info(
                    f"STRATEGY: Vol forecast ({vol_forecast:.2f}%) is BELOW buy threshold ({self.buy_threshold}%) -> Generate LONG signal."
                )
            elif is_bought:
                stop_loss_price = entryPrice * (1 - self.stop_loss)
                exitReason = None
                if current_price <= stop_loss_price:
                    exitReason = f"Stop Loss triggered at {current_price:.2f}"
                elif vol_forecast > self.sell_threshold:
                    exitReason = f"Vol forecast ({vol_forecast:.2f}%) is ABOVE sell threshold ({self.sell_threshold}%)"
                elif not up:
                    exitReason = f"Price ({current_price:.2f}) crossed BELOW SMA200 ({sma200:.2f})"
                if exitReason:
                    signal = SignalEvent(timestamp=event.timestamp,
                                         symbol=symbol,
                                         signal_type='Exit')
                    self.events.put(signal)
                    self.position[symbol]['bought'] = False
                    self.position[symbol]['entryPrice'] = 0.0
                    logging.info(f"[{event.timestamp}]STRATEGY: {exitReason} -> Generate EXIT signal.")