

class BaseMetric:
    def update(self, preds, labels):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError