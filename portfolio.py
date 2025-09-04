from events import OrderEvent

class Portfolio:
    def __init__(self, events_queue, symbol_list, data_handler, initial_capital = 100000.0):
        self.events = events_queue
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.data_handler = data_handler
        self.all_positions = []
        self.current_positions = {s:0.0 for s in self.symbol_list}
        self.all_holdings = []
        self.current_holdings = {'cash': initial_capital, 'total': initial_capital}
        self.record_initial_state()
    def record_initial_state(self):
        initial_state = {s:0.0 for s in self.symbol_list}
        initial_state['cash'] = self.initial_capital
        initial_state['total'] = self.initial_capital
        self.all_holdings.append(initial_state)
    def update_timeindex(self,event):
        if event.type == 'Market':
            for s in self.symbol_list:
                self.current_holdings[s] = 0.0
            current_total_value = self.current_holdings['cash']
            for symbol in self.symbol_list:
                position_qty = self.current_positions[symbol]
                if position_qty != 0:
                    latest_bar = self.data_handler.get_latest_bars(symbol)
                    if latest_bar:
                        market_val = latest_bar[0]['close'] * position_qty
                        self.current_holdings[symbol] = market_val
                        current_total_value += market_val
            self.current_holdings['total'] = current_total_value
            new_record = self.current_holdings.copy()
            new_record['timestamp'] = event.timestamp
            self.all_holdings.append(new_record)
            print(f"PORTFOLIO PnL: Date: {event.timestamp.strftime('%Y-%m-%d')}, Total Value: ${self.current_holdings['total']:.2f}")
    def on_signal(self,event):
        if event.type == 'Signal':
            if event.signal_type == 'Long':
                latest_bar = self.data_handler.get_latest_bars(event.symbol)
                if latest_bar:
                    close_price = latest_bar[0]['close']
                    target_investment = self.current_holdings['total'] * 0.95
                    current_position_value = self.current_positions.get(event.symbol, 0) * close_price
                    if current_position_value < target_investment:
                        investment_amount = target_investment - current_position_value
                        quantity = int(investment_amount / close_price)

                        if quantity > 0 and self.current_holdings['cash'] > (investment_amount):
                            order = OrderEvent(timestamp=event.timestamp,
                                               symbol=event.symbol,
                                               order_type='Mkt',
                                               quantity=quantity,
                                               direction='Buy')
                            print(f"PORTFOLIO: Received a LONG signal. Target is {target_investment:.2f}, current is {current_position_value:.2f}. Generating a BUY order for {quantity} shares.")
                            self.events.put(order)
            if event.signal_type == 'Exit':
                current_quantity = self.current_positions.get(event.symbol, 0)
                if current_quantity > 0:
                    order = order = OrderEvent(timestamp= event.timestamp,
                                   symbol=event.symbol,
                                   order_type='Mkt',
                                   quantity=current_quantity,
                                   direction='Sell')
                    print(f"PORTFOLIO: Received an EXIT signal. Generating a SELL order for {current_quantity} shares of {event.symbol}.")
                    self.events.put(order)
                else:
                    print(f"PORTFOLIO: Received an EXIT signal, but no position to sell for {event.symbol}.")
    def on_fill(self, event):
        if event.type == 'Fill':
            if event.direction == 'Buy':
                self.current_positions[event.symbol] += event.quantity
            elif event.direction == 'Sell':
                self.current_positions[event.symbol] -= event.quantity
            trade_cost = event.fill_cost + event.commission
            if event.direction == 'Buy':
                self.current_holdings['cash'] -= trade_cost
            elif event.direction == 'Sell':
                self.current_holdings['cash'] += (event.fill_cost - event.commission)
            print(
                f"PORTFOLIO: Updated on fill. Cash: {self.current_holdings['cash']:.2f}, Positions: {self.current_positions}")