from npt import NPT
from loss import Loss
import torch
from lamb import Lamb
from torch import autograd
import time
import torch.utils.data as utils_data
from util import permute
import math
from npt import NPT
from loss import Loss
from lookahead import Lookahead
from lamb import Lamb
from preprocessing import Preprocessing
from run_documentation import run_logger
from util import probs
from evaluation_metrics import *

MAX_BATCH_SIZE = 1100
evaluation_metrics_dict = {'acc': acc, 'nll': nll, 'rmse': rmse, 'mse': mse}


class Trainer():
    def __init__(self, params_dict, data, device,experiment = None,
                 lr_scheduler='flat_then_anneal', tradeoff_scheduler='cosine'):
        self.params = params_dict
        self.eval_metrics = {eval_metric:evaluation_metrics_dict[eval_metric]() for eval_metric in params_dict['evaluation_metrics']}
        self.eval_steps = params_dict['eval_every_n_th_epoch']
        self.print_steps = params_dict['print_every_n_th_epoch']
        self.clip = params_dict['clip']
        self.cv = params_dict['cv']
        self.p = probs(params_dict['features_mask_prob'], params_dict['label_mask_prob'])
        self.lr_scheduler = lr_scheduler
        self.tradeoff_scheduler = tradeoff_scheduler
        self.device = device
        self.run_logger = run_logger(data,experiment)
        self.cacher_improvements_necessary = params_dict['improvements_necessary']
        self.run_logger.record_hyperparameters(params_dict)
        self.step_each_n_batches = 1
        self.batches_per_epoch = 0

    def build_npt(self, encoded_data):
        self.model = NPT(encoded_data, self.params['model_layers'], self.params['heads'], self.params['rff'],
                         self.device,
                         self.params['drop'])
        if self.clip is not None:
            for p in self.model.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -self.clip, self.clip))

    def build_loss_function(self, data):
        self.loss_function = Loss(data, self.params['max_steps'], self.params['init_tradeoff'])

    def build_optimizer(self):
        lamb_optimizer = Lamb(self.model.parameters(), self.params['lr'], self.params['betas'], self.params['eps'])
        lookahead_optimizer = Lookahead(lamb_optimizer, self.params['max_steps'],
                                        self.params['flat'], self.params['k'],self.params['alpha'])
        self.optimizer = lookahead_optimizer

    def get_epochs_and_run_batch_size(self, data, max_steps, batch_size):
        if batch_size == -1:
            if data.orig_tensors.size(0) <= MAX_BATCH_SIZE:
                return max_steps,data.orig_tensors.size(0)
            else:
                self.step_each_n_batches = math.ceil(data.orig_tensors.size(0) / MAX_BATCH_SIZE)
                return max_steps, MAX_BATCH_SIZE
        else:
            if batch_size <= MAX_BATCH_SIZE:
                steps_per_epoch = math.ceil(data.orig_tensors.size(0) / batch_size)
                epochs = math.ceil(max_steps / steps_per_epoch)
                return epochs, batch_size
            else:
                steps_per_epoch = math.ceil(data.orig_tensors.size(0) / MAX_BATCH_SIZE)
                epochs = math.ceil(max_steps / steps_per_epoch)
                self.step_each_n_batches = math.ceil(batch_size / MAX_BATCH_SIZE)
                return epochs, MAX_BATCH_SIZE

    def run(self, data, batch_size, cv=None, experiment=None):
        if experiment is None:
            self.run_training(data, batch_size, cv)
        if experiment == 'duplicate':
            self.run_training(data,batch_size,cv,True)
        elif experiment == 'corruption':
            self.corruption_experiment(data, batch_size, cv)
        elif experiment == 'deletion':
            self.run_deletion(data, batch_size, cv)

    def run_training(self, data, batch_size, cv=None,duplicate = False):
        results = {}
        if cv == None:
            self.run_logger.define_cacher(improvements_necessary=self.cacher_improvements_necessary)
            encoded_data = Preprocessing(data, self.p)
            epochs, run_batch_size = self.get_epochs_and_run_batch_size(encoded_data, self.params['max_steps'], batch_size)
            self.build_npt(encoded_data)
            self.build_optimizer()
            self.build_loss_function(encoded_data)
            if duplicate:
                self.duplicate_experiment(encoded_data, epochs, run_batch_size)
                run_results = self.test_duplicate(encoded_data)
            else:
                self.train(encoded_data, epochs, run_batch_size)
                run_results = self.test(encoded_data)
            for metric,score in run_results.items():
                print(f"The {metric} loss on the test set is {score:.4f}")
                results[metric] = np.array(score)
        else:
            for i in range(1, cv + 1):
                print(f"Cross validation at split {i}/{cv}")
                data.next()
                self.run_logger.define_cacher(cv=i, improvements_necessary=self.cacher_improvements_necessary)
                encoded_data = Preprocessing(data, self.p)
                epochs, run_batch_size = self.get_epochs_and_run_batch_size(encoded_data, self.params['max_steps'], batch_size)
                self.build_npt(encoded_data)
                self.build_optimizer()
                self.build_loss_function(encoded_data)
                if duplicate:
                    self.duplicate_experiment(encoded_data, epochs, run_batch_size)
                    run_result = self.test(encoded_data)
                else:
                    self.train(encoded_data, epochs, run_batch_size)
                    run_result = self.test(encoded_data)
                for metric,score in run_result.items():
                    print(f"The {metric} loss on the test set is {score:.4f}")
                    if metric not in results.keys():
                        results[metric] = np.array(score)
                    else:
                        results[metric] = np.append(results[metric],score)
        self.calculate_and_save_results(results)

    def calculate_and_save_results(self, results):
        self.run_logger.create_results_file()
        for metric,scores in results.items():
            printable_results = f'''
            Run results for {metric} loss are:
            The mean of the results is {scores.mean():.4f}
            The std of the results is {scores.std():.4f}
            '''
            print(printable_results)
            self.run_logger.save_run_metric_results(printable_results)

    def train(self, encoded_data, epochs, batch_size):
        start = time.time()
        for epoch in range(1, epochs + 1):
            X_train,M_train,train_loss_indices,orig_data = encoded_data.masking('train')
            train = utils_data.TensorDataset(X_train.to(self.device), M_train.to(self.device),train_loss_indices.to(self.device),
                                             orig_data.to(self.device))
            train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=False)
            for batch_data in train_loader:
                self.batches_per_epoch += 1
                batch_X, batch_M,batch_loss_indices, batch_real_data = batch_data
                batch_loss = self.pass_through(batch_X, batch_M,batch_loss_indices, batch_real_data)
            if epoch % self.eval_steps == 0:
                self.model.eval()
                #TODO: replace this masking with randomized shuffle because the mask of the validation set is constant
                X_val, M_val, val_loss_indices,orig_data = encoded_data.masking('val')
                eval_loss = self.pass_through(X_val.to(self.device), M_val.to(self.device),val_loss_indices.to(self.device),
                                              orig_data.to(self.device))
                self.run_logger.check_improvement(self.model, eval_loss, epoch)
                if epoch % self.print_steps == 0:
                    print(f"Time for {self.eval_steps} epochs is {time.time() - start:.3f} seconds")
                    print(f"Current lr is {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"The validation loss is {eval_loss:.3f} after {epoch} epochs")
                    print(f"Model loss was {batch_loss:.3f} after {epoch} epochs")
                    start = time.time()
                self.model.train()
            self.optimizer.zero_grad()
            self.batches_per_epoch = 0

    def pass_through(self, batch_X, batch_M, loss_indices, batch_real_data):
        if self.model.training:
            z = self.model.forward(batch_X, batch_M)
            batch_loss = self.loss_function.compute(z, batch_real_data, loss_indices)
            batch_loss.backward()
            if self.batches_per_epoch % self.step_each_n_batches == 0:
                self.step()
        else:
            with torch.no_grad():
                z = self.model.forward(batch_X, batch_M)
                batch_loss = self.loss_function.val_loss(z, batch_real_data, loss_indices)
        return batch_loss.item()

    def step(self):
        self.optimizer.step()
        self.lr_scheduler_step()
        self.tradeoff_scheduler_step()
        self.optimizer.zero_grad()

    def lr_scheduler_step(self):
        if self.lr_scheduler == 'constant':
            return
        elif self.lr_scheduler == 'flat_then_anneal':
            self.optimizer.flat_then_anneal()

    def tradeoff_scheduler_step(self):
        if self.tradeoff_scheduler == 'constant':
            return
        elif self.tradeoff_scheduler == 'cosine':
            self.loss_function.Scheduler_cosine_step()

    def test(self, data):
        evaluation_metrics_results = {}
        self.run_logger.load_model(self.model)
        X_test, M_test, test_loss_indices,orig_data = data.masking('test')
        true_labels = orig_data[:,-1]
        self.model.eval()
        with torch.no_grad():
            z = self.model.forward(X_test.to(self.device), M_test.to(self.device))
            pred = z[data.target_col].detach().cpu()
            if data.target_type == 'continuous':
                pred *= data.stats[data.target_col]['std']
                pred += data.stats[data.target_col]['mean']
                true_labels *= data.stats[data.target_col]['std']
                true_labels += data.stats[data.target_col]['mean']
            for eval_metric,eval_metric_instance in self.eval_metrics.items():
                eval_loss = eval_metric_instance.compute(pred, true_labels,test_loss_indices[:,-1])
                evaluation_metrics_results[eval_metric] = eval_loss
        self.model.train()
        return evaluation_metrics_results

    def duplicate_experiment(self, encoded_data, epochs, batch_size):
        start = time.time()
        for epoch in range(1, epochs + 1):
            X_train,M_train,train_loss_indices,orig_data = encoded_data.masking('train')
            train = utils_data.TensorDataset(X_train.to(self.device), M_train.to(self.device),train_loss_indices.to(self.device),
                                             orig_data.to(self.device))
            train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=False)
            for batch_data in train_loader:
                self.batches_per_epoch += 1
                batch_X_modified,batch_M_modified,train_loss_indices_modified,batch_real_data_modified = self.duplicate(batch_data)
                batch_loss = self.pass_through(batch_X_modified, batch_M_modified,train_loss_indices_modified, batch_real_data_modified)
            if epoch % self.eval_steps == 0:
                self.model.eval()
                data_tuple = [item.to(self.device) for item in encoded_data.masking('val')]
                X_val_modified,M_val_modified,val_loss_indices_modified,orig_data_modified = self.duplicate(data_tuple)
                eval_loss = self.pass_through(X_val_modified.to(self.device), M_val_modified.to(self.device),
                                              val_loss_indices_modified.to(self.device),
                                              orig_data_modified.to(self.device))
                self.run_logger.check_improvement(self.model, eval_loss, epoch)
                if epoch % self.print_steps == 0:
                    print(f"Time for {self.eval_steps} epochs is {time.time() - start:.3f} seconds")
                    print(f"Current lr is {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"The validation loss is {eval_loss:.3f} after {epoch} epochs")
                    print(f"Model loss was {batch_loss:.3f} after {epoch} epochs")
                    start = time.time()
                self.model.train()
            self.optimizer.zero_grad()
            self.batches_per_epoch = 0

    def test_duplicate(self,data):
        evaluation_metrics_results = {}
        self.run_logger.load_model(self.model)
        data_tuple = [item.to(self.device) for item in data.masking('test')]
        X_test_modified,M_test_modified,test_loss_indices_modified,orig_data_modified = self.duplicate(data_tuple)
        true_labels = orig_data_modified[:,-1].detach().cpu()
        self.model.eval()
        with torch.no_grad():
            z = self.model.forward(X_test_modified.to(self.device), M_test_modified.to(self.device))
            pred = z[data.target_col].detach().cpu()
            if data.target_type == 'continuous':
                pred *= data.stats[data.target_col]['std']
                pred += data.stats[data.target_col]['mean']
                true_labels *= data.stats[data.target_col]['std']
                true_labels += data.stats[data.target_col]['mean']
            for eval_metric, eval_metric_instance in self.eval_metrics.items():
                eval_loss = eval_metric_instance.compute(pred, true_labels, test_loss_indices_modified[:,-1].detach().cpu())
                evaluation_metrics_results[eval_metric] = eval_loss
        self.model.train()
        return evaluation_metrics_results

    def duplicate(self,data_tuple):
        X,M,loss_indices,real_data = data_tuple
        X_modified = torch.cat((X, real_data), 0)
        M_modified = torch.cat((M, torch.zeros(M.shape).to(self.device)), 0)
        loss_indices_modified = torch.cat(
            (loss_indices, torch.zeros(loss_indices.shape).to(self.device)), 0)
        real_data_modified = torch.cat((real_data, real_data), 0)
        return X_modified,M_modified,loss_indices_modified,real_data_modified

    def corruption_experiment(self, data, batch_size, cv=None):
        self.run_training(data, batch_size, cv)
        full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
        full_data = utils_data.TensorDataset(torch.tensor(full_set, requires_grad=False, dtype=torch.float))
        data_loader = utils_data.DataLoader(full_data, batch_size=batch_size, shuffle=True)
        for batch_data in data_loader:
            batch_data = batch_data[0]
            for i in range(len(batch_data)):
                permuted_X, M, real_label = permute(batch_data, i)
                scores = self.evaluate(data, permuted_X, M, real_label)
                for metric,score in scores.items():
                    print(f"The {metric} loss of the sample is {score}")

    def data_deletion(self, data, batch_size, cv=None):
        self.run_training(data, batch_size, cv)
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
# def run_duplicate(self, data, batch_size, cv=None):
    #     if cv == None:
    #         self.run_logger.define_cacher(improvements_necessary=self.cacher_improvements_necessary)
    #         epochs, run_batch_size = self.get_epochs_and_run_batch_size(data, self.params['max_steps'], batch_size)
    #         encoded_data = Preprocessing(data, self.p)
    #         self.build_npt(encoded_data)
    #         self.build_optimizer()
    #         self.build_loss_function(encoded_data)
    #         self.train(encoded_data, epochs, run_batch_size)
    #         test_results = self.test(encoded_data)
    #         print(f"The evaluation metric loss on the test set is {test_results:.4f}")
    #         test_results = np.array(test_results)
    #     else:
    #         test_results = []
    #         for i in range(1, cv + 1):
    #             print(f"Cross validation at split {i}/{cv}")
    #             data.next()
    #             epochs, run_batch_size = self.get_epochs_and_run_batch_size(data, self.params['max_steps'], batch_size)
    #             encoded_data = Preprocessing(data, self.p)
    #             self.run_logger.define_cacher(cv=i, improvements_necessary=self.cacher_improvements_necessary)
    #             self.build_npt(encoded_data)
    #             self.build_optimizer()
    #             self.build_loss_function(encoded_data)
    #             self.train(encoded_data, epochs, run_batch_size)
    #             test_eval = self.test(encoded_data)
    #             print(f"The evaluation metric loss on the test set is {test_eval:.4f}")
    #             test_results.append(test_eval)
    #         test_results = np.array(test_results)
    #     self.calculate_and_save_results(test_results)
    # def evaluate(self, data, eval_X, eval_M, true_labels):
    #     evaluation_metrics_results = {}
    #     self.model.eval()
    #     with torch.no_grad():
    #         z = self.model.forward(eval_X.to(self.device), eval_M.to(self.device))
    #         pred = z[data.target_col].detach().cpu()
    #         if data.target_type == 'continuous':
    #             pred *= data.stats[data.target_col]['std']
    #             pred += data.stats[data.target_col]['mean']
    #             true_labels *= data.stats[data.target_col]['std']
    #             true_labels += data.stats[data.target_col]['mean']
    #         for eval_metric in self.eval_metric:
    #             eval_loss = evaluation_metrics_dict[eval_metric].compute(pred, true_labels, eval_M[:, -1])
    #             evaluation_metrics_results[eval_metric] = eval_loss
    #     self.model.train()
    #     return evaluation_metrics_results
