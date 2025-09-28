class Event:
    pass


class MarketEvent(Event):
    def __init__(self, timestamp, symbol, data):
        self.type = "Market"
        self.timestamp = timestamp
        self.symbol = symbol
        self.data = data


class SignalEvent(Event):
    def __init__(self, timestamp, symbol, signal_type, score=1.0, strength=1.0):
        self.type = "Signal"
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type
        self.score = score
        self.strength = strength


class OrderEvent(Event):
    def __init__(self, timestamp, symbol, order_type, quantity, direction):
        self.type = "Order"
        self.timestamp = timestamp
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction


class FillEvent(Event):
    def __init__(self, timestamp, symbol, quantity, direction, fill_cost, commission):
        self.type = "Fill"
        self.timestamp = timestamp
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission
