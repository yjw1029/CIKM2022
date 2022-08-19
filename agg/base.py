
class BaseAgg:
    def __init__(self, args, server_model):
        self.args = args
        self.server_model = server_model

    def aggregate(self, clients_rslts):
        pass