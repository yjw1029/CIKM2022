import torch

from .base import BaseMetric

class ErrorRate(BaseMetric):
    def __init__(self):
        self.all_error = 0
        self.all_cnt = 0

    def update(self, preds, labels):
        '''
        Args:
            preds: model predictions
            labels: the ground truths
        
        update the number of error samples and all samples.
        '''
        with torch.no_grad():
            self.all_cnt += preds.shape[0]
            self.all_error += (preds.argmax(dim=1) != labels).sum()

    def compute(self):
        '''
        compute the error rate
        '''
        with torch.no_grad():
            return self.all_error / self.all_cnt

    def clear(self):
        '''
        clear the number of error samples and all samples.
        '''
        self.all_error = 0
        self.all_cnt = 0