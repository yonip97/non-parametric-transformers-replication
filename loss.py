import math
import time

import torch
import numpy as np
#from torch.optim.lr_scheduler import CosineAnnealingLr

class Loss():
    def __init__(self, data, max_steps,tradeoff = 1):
        '''
        :param data: data wrapper.
        :param max_steps: max steps for the cosine annealing
        :param tradeoff: the tradeoff parameter between the features loss and the target loss
        for continuous attributes the loss is calculated as mse loss
        for categorical attributes the loss is calculated as cross entropy loss
        '''
        self.categorical = data.categorical
        self.continuous = data.continuous
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mse_loss = torch.nn.MSELoss(reduction = 'none')
        self.init_tradeoff = tradeoff
        self.curr_tradeoff = tradeoff
        self.label_col = data.target_col
        self.max_steps = max_steps
        self.steps_taken = 0



    def compute(self, pred, orig_data, M):
        losses = {}
        for col in self.categorical.keys():
            losses[col] =(M[:,col] *self.cross_entropy_loss.forward(pred[col],orig_data[:,col].long())).sum()
        for col in self.continuous:
            losses[col] =(M[:,col]* self.mse_loss.forward(pred[col], orig_data[:, col])).sum()
        label_loss = losses.pop(self.label_col, None)
        features_loss = torch.sum(torch.stack(list(losses.values())))
        return (1 - self.curr_tradeoff) * label_loss + self.curr_tradeoff * features_loss

    def Scheduler_cosine_step(self):
        if self.steps_taken <= self.max_steps:
            self.curr_tradeoff = self.init_tradeoff * (1 / 2) * (
                np.cos(np.pi * (self.steps_taken / self.max_steps)) + 1)
        else:
            self.curr_tradeoff = 0
        self.steps_taken += 1



