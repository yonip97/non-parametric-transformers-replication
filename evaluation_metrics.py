import numpy as np
import torch
from sklearn.metrics import *
from scipy.special import softmax
import torch
from torch.nn import NLLLoss,MSELoss,LogSoftmax

class acc():
    @staticmethod
    def compute(pred_labels, true_labels,mask):
        # temporary
        pred_labels = torch.argmax(pred_labels,dim=1)
        true_labels = torch.tensor(true_labels).long()
        return torch.sum((pred_labels == true_labels).float() * mask)/torch.sum(mask)


class nll():
    def __init__(self):
        self.loss = NLLLoss(reduction='none')
        self.logsoftmax =LogSoftmax(dim = 1)

    def compute(self,pred_labels, true_labels,mask):
        #true_labels = torch.tensor(true_labels).long()
        return (torch.sum(self.loss.forward(self.logsoftmax(pred_labels),true_labels.long())*mask)/torch.sum(mask)).item()


class auc_roc():
    #TODO: finish with mask

    def compute(self,pred_labels,true_labels,mask):
        return roc_auc_score(true_labels.numpy(),pred_labels)

class mse():
    def __init__(self):
        self.loss = MSELoss(reduction='none')

    def compute(self,pred_labels, true_labels,mask):
        true_labels = torch.tensor(true_labels).float()
        return torch.sum(self.loss.forward(pred_labels,true_labels)*mask)/torch.sum(mask)

class rmse():
    def __init__(self):
        self.mse = mse()

    def compute(self, pred_labels, true_labels,mask):
        return self.mse.compute(pred_labels,true_labels,mask)**0.5
