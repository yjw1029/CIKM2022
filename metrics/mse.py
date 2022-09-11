import torch
import torch.nn.functional as F

from .base import BaseMetric

class MSE(BaseMetric):
    def __init__(self):
        self.all_mse = 0
        self.all_cnt = 0

    def update(self, preds, labels):
        '''
        Args:
            preds: model predictions
            labels: the ground truths
        
        update the number of all samples and mse loss.
        '''
        with torch.no_grad():
            self.all_cnt += preds.shape[0]
            self.all_mse += F.mse_loss(preds, labels, reduction="mean") * preds.shape[0]

    def compute(self):
        '''
        compute mse loss
        '''
        with torch.no_grad():
            return self.all_mse / self.all_cnt

    def clear(self):
        '''
        clear the number of samples and mse loss
        '''
        self.all_mse = 0
        self.all_cnt = 0