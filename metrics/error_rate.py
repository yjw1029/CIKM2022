import torch

from .base import BaseMetric

class ErrorRate(BaseMetric):
    def __init__(self):
        self.all_error = 0
        self.all_cnt = 0

    def update(self, preds, labels):
        with torch.no_grad():
            self.all_cnt += preds.shape[0]
            self.all_error += (preds.argmax(dim=1) != labels).sum()

    def compute(self):
        with torch.no_grad():
            return self.all_error / self.all_cnt