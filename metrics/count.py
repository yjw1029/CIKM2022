import torch

from .base import BaseMetric


class Count(BaseMetric):
    def __init__(self):
        self.all_count = 0
        self.all_cnt = 0

    def update(self, loss):
        with torch.no_grad():
            self.all_cnt += 1
            self.all_count += loss

    def compute(self):
        with torch.no_grad():
            return self.all_count / self.all_cnt

    def clear(self):
        self.all_count = 0
        self.all_cnt = 0