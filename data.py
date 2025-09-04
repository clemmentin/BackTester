import pandas as pd
from abc import ABC, abstractmethod
from events import  MarketEvent

class DataHandler(ABC):
    @abstractmethod
    def get_latest_bars(self, symbol, N = 1):
        raise NotImplementedError('Should implement get_latest_bars')
    @abstractmethod
    def update_bars(self):
        raise NotImplementedError('Should implement update_bars')
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
                self.signal_csv_path,
                header=0,
                index_col=0,
                parse_dates=True
            )
            signals.rename(columns={'annualized_volatility_pct': 'vol_forecast'}, inplace=True)
            return signals
        except FileNotFoundError:
            print(f"WARNING: Signal file not found at {self.signal_csv_path}. Proceeding without signals.")
            return None
    def _open_and_load_csv_files(self):
        for s in self.symbol_list:
            file_path = f"{self.csv_dir}/{s}.csv"
            price_data = pd.read_csv(
                file_path, header=0, index_col=0, parse_dates=True
            )
            price_data['sma200'] = price_data['Close'].rolling(window=200).mean()
            if self.signal_data is not None:
                combined_data = price_data.merge(self.signal_data, left_index=True, right_index=True, how='left')
                combined_data['vol_forecast'] = combined_data['vol_forecast'].ffill()
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
                    'open': row['Open'], 'high': row['High'],
                    'low': row['Low'], 'close': row['Close'],
                    'volume': row['Volume'],
                    'vol_forecast': row.get('vol_forecast', 0),
                    'sma200': row.get('sma200', None)
                }
                self.latest_symbol_data[s] = bar
                market_event = MarketEvent(timestamp = timestamp, symbol = s, data = bar)
                self.events.put(market_event)
    def get_latest_bars(self, symbol, N = 1):
        try:
            return [self.latest_symbol_data[symbol]]
        except KeyError:
            print(f'Data for symbol {symbol} is not available yet')
            return []