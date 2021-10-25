from npt import NPT
from loss import Loss
import torch
from lamb import Lamb
import time
import torch.utils.data as utils_data
import numpy as np
#from evaluation_metrics import *
from util import permute
import math
from npt import NPT
from loss import Loss
from lookahead import Lookahead
from lamb import Lamb
from preprocessing import Preprocessing

class Trainer():
    def __init__(self, params_dict, eval_metric, eval_every_n_th_epoch,data, device, clip=None,
                 lr_scheduler='flat_then_anneal', tradeoff_scheduler='cosine'):
        self.params = params_dict
        self.eval_metric = eval_metric
        self.eval_steps = eval_every_n_th_epoch
        if clip != None:
            self.clip = clip
        else:
            self.clip = None
        self.lr_scheduler = lr_scheduler
        self.tradeoff_scheduler = tradeoff_scheduler
        self.device = device

    def build_npt(self,encoded_data):
        self.model = NPT(encoded_data,self.params['model_layers'], self.params['heads'], self.params['rff'], self.device,
                         self.params['drop'])
        if self.clip is not None:
            for p in self.model.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -self.clip, self.clip))

    def build_loss_function(self, data):
        self.loss_function = Loss(data, self.params['max_steps'], self.params['init_tradeoff'])

    def build_optimizer(self):
        lamb_optimizer = Lamb(self.model.parameters(), self.params['lr'], self.params['betas'], self.params['eps'])
        lookahead_optimizer = Lookahead(lamb_optimizer, self.params['max_steps'], self.params['flat'], self.params['k'],
                                        self.params['alpha'])
        self.optimizer = lookahead_optimizer

    def get_epochs(self, data, max_steps, batch_size):
        if batch_size == -1:
            return max_steps, len(data.train_data)
        else:
            steps_per_epoch = math.ceil(len(data.train_data) / batch_size)
            epochs = math.ceil(max_steps / steps_per_epoch)
            return epochs, batch_size

    def run(self,data,p,batch_size,cv = None,test = None):
        if test == None:
            self.run_training(data,p,batch_size,cv)
        elif test == 'duplicate':
            self.run_duplicate(data,p,batch_size, cv)
        elif test == 'corruption':
            self.run_corroption(data,p,batch_size,cv)
        elif test == 'deletion':
            self.run_deletion(data,p,batch_size,cv)

    def run_training(self, data,p, batch_size, cv=None):
        if cv == None:
            epochs, batch_size = self.get_epochs(data, self.params['max_steps'], batch_size)
            encoded_data = Preprocessing(data, p)
            self.build_npt(encoded_data)
            self.build_optimizer()
            self.build_loss_function(encoded_data)
            self.train(encoded_data,epochs,batch_size)
            test_eval = self.test(encoded_data)
            print(f"The evaluation metric loss on the test set is {test_eval}")
            return test_eval
        else:
            test_evals = []
            for i in range(cv):
                data.next()
                epochs, batch_size = self.get_epochs(data, self.params['max_steps'], batch_size)
                encoded_data = Preprocessing(data, p)
                self.build_npt(encoded_data)
                self.build_optimizer()
                self.build_loss_function(encoded_data)
                self.train(encoded_data,epochs,batch_size)
                test_eval = self.test(encoded_data)
                print(f"The evaluation metric loss on the test set is {test_eval}")
                test_evals.append(test_eval)
            return np.array(test_evals)

    def train(self, encoded_data,epochs, batch_size):
        X_train,M_train,orig_train_data = encoded_data.get('train')
        X_val,M_val,orig_val_data = encoded_data.get('val')
        train = utils_data.TensorDataset(X_train.to(self.device), M_train.to(self.device),orig_train_data.to(self.device))
        train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=True)
        start = time.time()
        for epoch in range(1, epochs + 1):
            for batch_data in train_loader:
                batch_X, batch_M, batch_real_data = batch_data
                batch_loss = self.pass_through(batch_X, batch_M, batch_real_data)
            if epoch % self.eval_steps == 0:
                #true_labels = np.concatenate((data.train_data[:, -1], data.val_data[:, -1]))
                #eval_X, eval_M = data.mask_targets('val')
                eval_loss = self.evaluate(encoded_data, X_val, M_val, orig_val_data[:,-1])
                print(f"time for {self.eval_steps} epochs is {time.time()-start} seconds")
                print(f"current lr is {self.optimizer.param_groups[0]['lr']}")
                print(f"current tradeoff is {self.loss_function.curr_tradeoff}")
                start = time.time()
                print(f"The evaluation metric loss on the validation set is {eval_loss} after {epoch} epochs")
                print(f"model loss was {batch_loss} after {epoch} epochs")

    def pass_through(self, batch_X, batch_M, batch_real_data):
        z = self.model.forward(batch_X, batch_M)
        batch_loss = self.loss_function.compute(z, batch_real_data, batch_M)
        batch_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler == 'flat_then_anneal':
            self.optimizer.flat_then_anneal()
        if self.tradeoff_scheduler == 'cosine':
            self.loss_function.Scheduler_cosine_step()
        self.optimizer.zero_grad()
        return batch_loss.item()

    def evaluate(self, data, eval_X, eval_M, true_labels):
        self.model.eval()
        with torch.no_grad():
            z = self.model.forward(eval_X.to(self.device), eval_M.to(self.device))
            pred = z[data.target_col].detach().cpu()
            # eval_acc = acc.compute(pred,true_labels,eval_M[:,-1])
            if data.target_type =='continuous':
                pred *= data.stats[data.target_col]['std']
                pred += data.stats[data.target_col]['mean']
                true_labels *= data.stats[data.target_col]['std']
                true_labels += data.stats[data.target_col]['mean']
            eval_loss = self.eval_metric.compute(pred, true_labels, eval_M[:, -1])
        self.model.train()
        return eval_loss

    def test(self,data):
        X_test, M_test,orig_test_tensors = data.get('test')
        test_eval = self.evaluate(data, X_test, M_test, orig_test_tensors[:,-1])
        return test_eval


    def run_duplicate(self,data,batch_size,cv = None):
        if cv == None:
            self.build_npt(data)
            self.build_optimizer()
            self.build_loss_function(data)
            self.data_duplicate(data, batch_size)
            X_test, M_test = data.mask_targets('test')
            true_labels = np.concatenate((data.train_data[:, -1], data.val_data[:, -1], data.test_data[:, -1]))
            true_labels_modified = np.concatenate((true_labels, true_labels), 0)
            X_test_modified = torch.cat(
                (X_test, torch.Tensor(data.train_data), torch.tensor(data.val_data), torch.tensor(data.test_data)), 0)
            M_test_modified = torch.cat((M_test, torch.zeros(M_test.shape)), 0)
            test_eval = self.evaluate(data, X_test_modified, M_test_modified, true_labels_modified)
            print(f"The evaluation metric loss on the test set is {test_eval}")
            return test_eval
        else:
            test_evals = []
            for i in range(cv):
                data.next()
                self.build_npt(data)
                self.build_optimizer()
                self.build_loss_function(data)
                self.data_duplicate(data, batch_size)
                X_test, M_test = data.mask_targets('test')
                true_labels = np.concatenate((data.train_data[:, -1], data.val_data[:, -1], data.test_data[:, -1]))
                true_labels_modified = np.concatenate((true_labels, true_labels), 0)
                X_test_modified = torch.cat((X_test, torch.Tensor(data.train_data).float(), torch.tensor(data.val_data).float(), torch.tensor(data.test_data).float()),0)
                M_test_modified = torch.cat((M_test, torch.zeros(M_test.shape)), 0)
                test_eval = self.evaluate(data, X_test_modified, M_test_modified, true_labels_modified)
                print(f"The evaluation metric loss on the test set is {test_eval}")
                test_evals.append(test_eval)
            return np.array(test_evals)

    def data_duplicate(self, data, batch_size):
        epochs, batch_size = self.get_epochs(data, self.params['max_steps'], batch_size)
        for epoch in range(1, epochs + 1):
            train_data, val_data, test_data = data.normalize()
            X, M = data.create_training_mask(train_data)
            train = utils_data.TensorDataset(X.to(self.device), M.to(self.device),
                                             torch.tensor(train_data, requires_grad=False, dtype=torch.float).to(
                                                 self.device))
            train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=True)
            for batch_data in train_loader:
                batch_X, batch_M, batch_real_data = batch_data
                batch_X_modified = torch.cat((batch_X, batch_real_data), 0)
                batch_M_modified = torch.cat((batch_M, torch.zeros(batch_M.shape).to(self.device).float()), 0)
                batch_real_data_modified = torch.cat((batch_real_data, batch_real_data), 0)
                batch_loss = self.pass_through(batch_X_modified, batch_M_modified, batch_real_data_modified)
            if epoch % self.eval_steps == 0:
                true_labels = np.concatenate((data.train_data[:, -1], data.val_data[:, -1]))
                eval_X, eval_M = data.mask_targets('val')
                eval_X_modified = torch.cat((eval_X, torch.tensor(data.train_data).float(), torch.tensor(data.val_data).float()), 0)
                eval_M_modified = torch.cat((eval_M, torch.zeros(eval_M.shape)), 0)
                true_labels_modified = np.concatenate((true_labels, true_labels), 0)
                eval_loss = self.evaluate(data, eval_X_modified, eval_M_modified, true_labels_modified)
                print(f"model loss was {batch_loss} after {epoch} epochs")
                print(f"The evaluation metric loss on the validation set is {eval_loss} after {epoch} epochs")

    def data_corruption(self,data,batch_size,cv = None):
        self.run_training(data,batch_size,cv)
        full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
        full_data = utils_data.TensorDataset(torch.tensor(full_set, requires_grad=False, dtype=torch.float))
        data_loader = utils_data.DataLoader(full_data, batch_size=batch_size, shuffle=True)
        for batch_data in data_loader:
            batch_data = batch_data[0]
            for i in range(len(batch_data)):
                permuted_X, M, real_label = permute(batch_data, i)
                score = self.evaluate(data,permuted_X,M,real_label)
                print(f"the evaluation metric loss of the sample is {score}")

    def data_deletion(self,data,batch_size,cv = None):
        self.run_training(data,batch_size,cv)
        self.model.eval()
        full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
        full_data = torch.tensor(full_set, dtype=torch.float)
        max_change = 0.1
        max_change_per_delete = 0.01
        N_max_retry = 50
        eps = 0.02
        N_retry = 0
        result_dict = {}
        for i in range(len(full_set)):
            y = full_data[i]
            true_label = y[-1].item()
            y_masked = y.reshape(1, -1)
            y_masked[:, -1] = 0
            R = np.concatenate((np.arange(0, i), np.arange(i + 1, len(full_data))))
            while True:
                deleted_index = np.random.randint(low=0, high=len(R))
                new_R = np.delete(R, deleted_index)
                selected_rows = full_data[torch.tensor(new_R).long()]
                X = torch.cat((selected_rows, y_masked), dim=0)
                M = torch.zeros(X.size())
                M[-1, -1] = 1
                y_proposed = self.model.forward(X.to(self.device), M.to(self.device))[-1, -1].item()
                delta_proposal = abs((y_proposed - true_label) / true_label)
                if delta_proposal < max_change_per_delete:
                    if delta_proposal < max_change:
                        R = new_R
                        N_retry = 0
                    else:
                        break
                else:
                    N_retry += 1
                if N_retry >= N_max_retry:
                    max_change_per_delete *= 1.1
                    N_retry = 0
                if len(R) < eps * len(full_data):
                    break
            result_dict[i] = R
        return result_dict




















# def data_corruption(data, model, epochs, optimizer, batch_size, loss_function, eval_metric, eval_each_n_epochs,
#                     switch_to_pred=False, clip=None, use_lr_scheduler=False, use_tradeoff_scheduler=False):
#     train_model(data, model, epochs, optimizer, batch_size, loss_function, eval_metric, eval_each_n_epochs,
#                 switch_to_pred, clip, use_lr_scheduler, use_tradeoff_scheduler)
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'
#     full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
#     full_data = utils_data.TensorDataset(torch.tensor(full_set, requires_grad=False, dtype=torch.float))
#     data_loader = utils_data.DataLoader(full_data, batch_size=batch_size, shuffle=True)
#     for batch_data in data_loader:
#         batch_data = batch_data[0]
#         for i in range(len(batch_data)):
#             permuted_X, M, real_label = permute(batch_data, i)
#             if switch_to_pred:
#                 model.eval()
#                 z = model.forward(permuted_X.to(device), M.to(device))
#                 pred = z[-1, data.target_col].detach().cpu().view(1, -1)
#                 score = eval_metric.compute(pred, real_label.view(1), M[-1, data.target_col].cpu())
#                 model.train()
#             else:
#                 with torch.no_grad():
#                     z = model(permuted_X.to(device), M.to(device))
#                     pred = z[data.target_col].detach().cpu()
#                     score = eval_metric.compute(pred[-1].view(1, -1), real_label.view(1), M[-1, data.target_col])
#             print(f"the nll loss of the sample is {score.item()}")


# def data_deletion(data, model, epochs, optimizer, batch_size, loss_function, eval_metric, eval_each_n_epochs,
#                   switch_to_pred=False, clip=None, use_lr_scheduler=False, use_tradeoff_scheduler=False):
#     train_model(data, model, epochs, optimizer, batch_size, loss_function, eval_metric, eval_each_n_epochs,
#                 switch_to_pred, clip, use_lr_scheduler, use_tradeoff_scheduler)
#     model.eval()
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'
#     full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
#     data = torch.tensor(full_set, dtype=torch.float)
#     max_change = 0.1
#     max_change_per_delete = 0.01
#     N_max_retry = 50
#     eps = 0.02
#     N_retry = 0
#     for i in range(len(full_set)):
#         y = data[i]
#         true_label = y[-1].item()
#         y_masked = y.reshape(1, -1)
#         y_masked[:, -1] = 0
#         R = np.concatenate((np.arange(0, i), np.arange(i + 1, len(data))))
#         while True:
#             deleted_index = np.random.randint(low=0, high=len(R))
#             new_R = np.delete(R, deleted_index)
#             selected_rows = data[torch.tensor(new_R).long()]
#             X = torch.cat((selected_rows, y_masked), dim=0)
#             M = torch.zeros(X.size())
#             M[-1, -1] = 1
#             y_proposed = model.forward(X.to(device), M.to(device))[-1, -1].item()
#             delta_proposal = abs((y_proposed - true_label) / true_label)
#             if delta_proposal < max_change_per_delete:
#                 if delta_proposal < max_change:
#                     R = new_R
#                     N_retry = 0
#                 else:
#                     break
#             else:
#                 N_retry += 1
#             if N_retry >= N_max_retry:
#                 max_change_per_delete *= 1.1
#                 N_retry = 0
#             if len(R) < eps * len(data):
#                 break
#         return R

# def train_model(data, model, epochs, optimizer, batch_size, loss_function, eval_metric, eval_each_n_epochs, clip = None, use_lr_scheduler = False, use_tradeoff_scheduler = False):
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'
#     start = time.time()
#     for epoch in range(epochs):
#         X, M = data.create_mask()
#         train = utils_data.TensorDataset(X.to(device), M.to(device), torch.tensor(data.train_data,requires_grad=False,dtype=torch.float).to(device))
#         train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=True)
#         for batch_data in train_loader:
#             batch_X, batch_M, batch_real_data = batch_data
#             z = model.forward(batch_X, batch_M)
#             batch_loss = loss_function.compute(z, batch_real_data, batch_M)
#             batch_loss.backward()
#             #print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
#             #print(f"batch time {time.time()-start} seconds")
#             if clip != None:
#                 optimizer.clip_gradient(model, clip)
#             if use_lr_scheduler:
#                 optimizer.flat_then_anneal()
#             if use_tradeoff_scheduler:
#                 loss_function.Scheduler_step()
#             optimizer.step()
#             optimizer.zero_grad()
#
#         if epoch % eval_each_n_epochs == 0:
#             model.eval()
#             true_labels = np.concatenate((data.train_data[:, -1], data.val_data[:, -1]))
#             eval_X, eval_M = data.mask_targets('val')
#             z = model.forward(eval_X.to(device), eval_M.to(device))
#             if model.finalize:
#                 pred = z[:,-1].detach().cpu()
#                 #eval_acc = acc.compute(pred, true_labels, eval_M[:, -1])
#             else:
#                 pred = z[data.target_col].detach().cpu()
#                 #eval_acc = acc.compute(pred,true_labels,eval_M[:,-1])
#             eval_loss = eval_metric.compute(pred, true_labels, eval_M[:, -1])
#             model.train()
#             print(f"Epoch {epoch} the rmse loss is {eval_loss}")
#             print(f"100 epochs time {time.time() - start}")
#             print(f"epoch number {epoch} and the loss is {batch_loss}")
#             start = time.time()
#             #print(f"Epoch {epoch} the accuracy is {eval_acc}")
#     model.eval()
#     X_test, M_test = data.mask_targets('test')
#     true_labels = np.concatenate((data.train_data[:,-1],data.val_data[:,-1],data.test_data[:,-1]))
#     z = model.forward(X_test.to(device),M_test.to(device))
#     if model.finalize:
#         pred = z[:,-1].detach().cpu()
#         #acc_score = acc.compute(pred[:,-1].detach().cpu(),true_labels,M_test[:,-1])
#     else:
#         pred = z[data.target_col].detach().cpu()
#     nll_score = eval_metric.compute(pred, true_labels,M_test[:,-1])
#             #acc_score = acc.compute(pred[data.target_col].detach().cpu(), true_labels, M_test[:, -1])
#     print(f"rmse score {nll_score}")
#     return nll_score

# def duplicate_test(data, model, epochs, optimizer, batch_size, loss_function, eval_metric, eval_each_n_epochs,
#                    switch_to_pred=False, clip=None, cv_runs=1):
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'
#     for epoch in range(epochs):
#         for run in range(cv_runs):
#             X, M = data.create_mask('train')
#             train = utils_data.TensorDataset(X.to(device).float(), M.to(device).float(),
#                                              torch.tensor(data.train_data, requires_grad=False).to(device).float())
#             train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=True)
#             for batch_data in train_loader:
#                 start = time.time()
#                 batch_X, batch_M, batch_real_data = batch_data
#                 batch_X_modified = torch.cat((batch_X, batch_real_data), 0)
#                 batch_M_modified = torch.cat((batch_M, torch.zeros(batch_M.shape).to(device).float()), 0)
#                 batch_real_data_modified = torch.cat((batch_real_data, batch_real_data), 0)
#                 z_modified = model.forward(batch_X_modified, batch_M_modified)
#                 loss = loss_function.forward(z_modified, batch_real_data_modified, batch_M_modified)
#                 if loss != -1:
#                     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
#                     loss.backward()
#                     if clip != None:
#                         optimizer.clip_gradient(model, clip)
#                     optimizer.step()
#                     optimizer.zero_grad()
#                     print(time.time() - start)
#                     print(f"epoch number {epoch} and the loss is {loss}")
#         if cv_runs != 1:
#             data.restart_splitter()
#     if switch_to_pred:
#         model.eval()
#     X_test, M_test = data.mask_targets()
#     X_test_modified = torch.cat((X_test, torch.tensor(data.test_data)), 0)
#     M_test_modified = torch.cat((M_test, torch.zeros(M_test.shape)), 0)
#     with torch.no_grad():
#         pred = model.forward(X_test_modified.to(device), M_test_modified.to(device))
#         score = eval_metric.compute(pred[:, -1].detch().cpu().numpy(), data.test_data[:, -1])
#         print(score)
#