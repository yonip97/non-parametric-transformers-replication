from sklearn.metrics import *
import torch
from torch.nn import NLLLoss,MSELoss,LogSoftmax

class acc():
    @staticmethod
    def compute(pred_labels, true_labels,mask):
        pred_labels = torch.argmax(pred_labels,dim=1)
        true_labels = torch.tensor(true_labels).long()
        return torch.sum((pred_labels == true_labels).float() * mask)


class nll():
    def __init__(self):
        self.loss = NLLLoss(reduction='none')
        self.logsoftmax =LogSoftmax(dim = 1)

    def compute(self,pred_labels, true_labels,mask):
        return (torch.sum(self.loss.forward(self.logsoftmax(pred_labels),true_labels.long())*mask)).item()


class auc_roc():

    def compute(self,pred_labels,true_labels,loss_indices):
        loss_indices= loss_indices == 1
        pred_labels = torch.argmax(pred_labels[loss_indices],dim=1).detach().cpu().numpy()
        true_labels = true_labels[loss_indices].detach().cpu().numpy()
        return roc_auc_score(true_labels,pred_labels)


class mse():
    def __init__(self):
        self.loss = MSELoss(reduction='none')

    def compute(self,pred_labels, true_labels,mask):
        true_labels = torch.tensor(true_labels).float()
        return torch.sum(self.loss.forward(pred_labels,true_labels)*mask)

class rmse():
    def __init__(self):
        self.mse = mse()

    def compute(self, pred_labels, true_labels,mask):
        return self.mse.compute(pred_labels,true_labels,mask)**0.5
