

class BaseServer:
    def __init__(self, args, clients):
        self.args = args
        self.clients = clients
        self.init_aggregator = self.init_aggregator()

    def init_aggregator(self):
        pass
    
    def sample_clients(self):
        pass

    def aggregate(self):
        pass