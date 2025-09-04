from abc import ABC, abstractmethod
from events import SignalEvent
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
        self.bought = {s: False for s in self.symbol_list}
    def calculate_signal(self, event):
        if event.type == 'Market':
            symbol = event.symbol
            vol_forecast = event.data.get('vol_forecast', None)
            current_price = event.data.get('close')
            sma200 = event.data.get('sma200', None)
            if vol_forecast is None or sma200 is None or current_price is None:
                return
            up = current_price > sma200
            if vol_forecast < self.buy_threshold and up and not self.bought[symbol]:
                signal = SignalEvent(timestamp=event.timestamp,
                                     symbol=symbol,
                                     signal_type='Long')
                self.events.put(signal)
                self.bought[symbol] = True
                print(
                    f"STRATEGY: Vol forecast ({vol_forecast:.2f}%) is BELOW buy threshold ({self.buy_threshold}%) -> Generate LONG signal."
                )
            elif (vol_forecast > self.sell_threshold or not up)and self.bought[symbol]:
                signal = SignalEvent(timestamp=event.timestamp,
                                     symbol=symbol,
                                     signal_type='Exit')
                self.events.put(signal)
                self.bought[symbol] = False
                print(
                    f"STRATEGY: Vol forecast ({vol_forecast:.2f}%) is ABOVE sell threshold ({self.sell_threshold}%) -> Generate EXIT signal."
                )