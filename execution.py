from events import FillEvent
class SimulatedExecutionHandler:
    def __init__(self, events_queue, data_handler):
        self.events = events_queue
        self.data_handler = data_handler
    def execute_order(self, event):
        if event.type == 'Order':
            latest_bar = self.data_handler.get_latest_bars(event.symbol)
            if not latest_bar:
                print('Error')
                return
            fill_price = latest_bar[0]['close']
            fill_cost = fill_price * event.quantity
            commission = self.calculate_commission(fill_cost)
            fill_event = FillEvent(timestamp=event.timestamp,
                                   symbol=event.symbol,
                                   quantity=event.quantity,
                                   direction=event.direction,
                                   fill_cost=fill_cost,
                                   commission=commission)
            print(f"EXECUTION: Order for {event.quantity} {event.symbol} {event.direction} filled at ${fill_price:.2f}.")
            self.events.put(fill_event)
    def calculate_commission(self, fill_cost):
        return max(1.0, fill_cost * 0.001)
