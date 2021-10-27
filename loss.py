import math
import time
from collections import defaultdict
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

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mse_loss = torch.nn.MSELoss(reduction = 'none')
        self.init_tradeoff = tradeoff
        self.curr_tradeoff = tradeoff
        if data.target_type == 'categorical':
            temp = data.categorical.copy()
            temp.pop(data.target_col)
            self.categorical = temp
            self.continuous = data.continuous
        else:
            temp = data.continuous.copy()
            temp.remove(data.target_col)
            self.continuous = temp
            self.categorical = data.categorical
        self.target = data.target_col
        self.target_type = data.target_type
        self.max_steps = max_steps
        self.steps_taken = 0


    def compute(self, pred, orig_data, M):
        losses = {'losses': {"features": dict()}, "predictions": dict()}
        for col in self.categorical.keys():
            losses["predictions"][col] = M[:,col].sum()
            losses["losses"]["features"][col] =(M[:,col] *self.cross_entropy_loss.forward(pred[col],orig_data[:,col].long())).sum()
        for col in self.continuous:
            losses["predictions"][col] = M[:, col].sum()
            losses["losses"]["features"][col] =(M[:,col]* self.mse_loss.forward(pred[col], orig_data[:, col])).sum()
        if self.target_type =='categorical':
            sum_label_loss = (M[:,self.target] *self.cross_entropy_loss.forward(pred[self.target],orig_data[:,self.target].long())).sum()
        else:
            sum_label_loss = (M[:,self.target]* self.mse_loss.forward(pred[self.target], orig_data[:, self.target])).sum()
        losses["predictions"][self.target] = M[:,self.target].sum()
        sum_feature_loss = torch.sum(torch.stack(list(losses["losses"]["features"].values()),0))
        #loss = (sum_feature_loss+sum_label_loss) /torch.sum(torch.stack(list(losses['predictions'].values()),0))
        label_predictions = losses['predictions'].pop(self.target)
        features_predictions = torch.sum(torch.stack(list(losses['predictions'].values()),0))
        if label_predictions != 0:
            label_loss = sum_label_loss / label_predictions
            if features_predictions != 0:
                feature_loss = sum_feature_loss / features_predictions
                loss = self.curr_tradeoff * feature_loss + (1 - self.curr_tradeoff) * label_loss
            else:
                loss = (1 - self.curr_tradeoff) * label_loss
        else:
            if features_predictions != 0:
                feature_loss = sum_feature_loss / features_predictions
                loss = self.curr_tradeoff * feature_loss
            else:
                loss = torch.tensor(0)
        return loss
        #
        # feature_loss = sum_feature_loss / torch.sum(torch.stack(list(losses['predictions'].values()),0))
        # loss = self.curr_tradeoff * feature_loss + (1 - self.curr_tradeoff) * label_loss
        # label_loss = self.losses.pop(self.label_col, None)
        # features_loss = torch.sum(torch.stack(list(losses.values())))
        # return (1 - self.curr_tradeoff) * label_loss + self.curr_tradeoff * features_loss


    def Scheduler_cosine_step(self):
        if self.steps_taken <= self.max_steps:
            self.curr_tradeoff = self.init_tradeoff * (1 / 2) * (
                np.cos(np.pi * (self.steps_taken / self.max_steps)) + 1)
        else:
            self.curr_tradeoff = 0
        self.steps_taken += 1

    def finalize_loss(self):
        defaultdict(dict)


