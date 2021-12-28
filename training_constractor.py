import time
import torch.utils.data as utils_data
from util import permute
import math
from npt import NPT
from loss import Loss
from lookahead import Lookahead
from preprocessing import Preprocessing
from run_documentation import run_logger
from util import probs
from evaluation_metrics import *
from scipy.stats import wilcoxon
import numpy as np

MAX_BATCH_SIZE = 1024
evaluation_metrics_dict = {'acc': acc, 'nll': nll, 'rmse': rmse, 'mse': mse,'auc-roc':auc_roc}


class Trainer():
    def __init__(self, params_dict, data, device, experiment=None):
        self.params = params_dict
        self.eval_metrics = {eval_metric: evaluation_metrics_dict[eval_metric]() for eval_metric in
                             params_dict['evaluation_metrics']}
        self.eval_epochs = params_dict['eval_every_n_th_epoch']
        self.print_epochs = params_dict['print_every_n_th_epoch']
        self.clip = params_dict['clip']
        self.cv = params_dict['cv']
        self.p = probs(params_dict['features_mask_prob'], params_dict['label_mask_prob'])
        self.lr_scheduler = params_dict['lr_scheduler']
        self.tradeoff_scheduler =params_dict['tradeoff_scheduler']
        self.device = device
        self.run_logger = run_logger(data, experiment)
        self.cacher_improvements_necessary = params_dict['improvements_necessary']
        self.run_logger.record_hyperparameters(params_dict)
        self.step_each_n_batches = 1
        self.batches_per_epoch = 0

    def build_npt(self, encoded_data):
        self.model = NPT(encoded_data, self.params['model_layers'], self.params['heads'], self.params['rff_layers'],
                         self.device,
                         self.params['drop_out'])
        self.model.define_in_and_out_encoding(encoded_data,self.device,self.params)
        if self.clip is not None:
            for p in self.model.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -self.clip, self.clip))

    def build_loss_function(self, encoded_data):
        self.loss_function = Loss(encoded_data.data, self.params['max_steps'], self.params['init_tradeoff'])

    def build_optimizer(self):
        lookahead_optimizer = Lookahead(model_parameters=self.model.parameters(), init_lr=self.params['lr'],
                                        max_steps=self.params['max_steps'],
                                        warmup_proportion=self.params['warmup'],end_lr=self.params['end_lr'])
        self.optimizer = lookahead_optimizer

    def get_epochs_and_run_batch_size(self, data, max_steps, batch_size):
        if batch_size == -1:
            if data.orig_size <= MAX_BATCH_SIZE:
                return max_steps, data.orig_size
            else:
                while batch_size > MAX_BATCH_SIZE:
                    batch_size  = math.ceil(batch_size/2)
                    self.step_each_n_batches *=2
                return max_steps, batch_size
        else:
            if batch_size <= MAX_BATCH_SIZE:
                steps_per_epoch = math.ceil(data.orig_size / batch_size)
                epochs = math.ceil(max_steps / steps_per_epoch)
                return epochs, batch_size
            else:
                while batch_size > MAX_BATCH_SIZE:
                    batch_size = math.ceil(batch_size/2)
                    self.step_each_n_batches *= 2
                steps_per_epoch = math.ceil(data.orig_size / batch_size)
                epochs = math.ceil(max_steps / steps_per_epoch)
                return epochs, batch_size

    def run(self, data, batch_size, cv=None, experiment=None,loading_paths = None):
        if experiment is None:
            self.run_training(data, batch_size, cv,loading_paths)
        if experiment == 'duplicate':
            self.run_training(data, batch_size, cv, True,loading_paths)
        elif experiment == 'corruption':
            self.corruption_experiment(data, batch_size,loading_paths)
        elif experiment == 'deletion':
            self.deletion_experiment(data, batch_size,loading_paths)


    def run_training(self, data, batch_size, cv=None, duplicate=False,loading_paths = None):
        results = {}
        if cv == None:
            self.run_logger.define_cacher(improvements_necessary=self.cacher_improvements_necessary)
            encoded_data = Preprocessing(data, self.p)
            epochs, run_batch_size = self.get_epochs_and_run_batch_size(encoded_data, self.params['max_steps'],
                                                                        batch_size)
            self.build_npt(encoded_data)
            self.build_optimizer()
            self.build_loss_function(encoded_data)
            if loading_paths is not None:
                self.load_from_save(loading_paths[0])
            else:
                if duplicate:
                    self.duplicate_experiment(encoded_data, epochs, run_batch_size)
                else:
                    self.train(encoded_data, epochs, run_batch_size)
            if duplicate:
                run_results = self.test_duplicate(encoded_data,run_batch_size)
            else:
                run_results = self.test(encoded_data,run_batch_size)
            for metric, score in run_results.items():
                print(f"The {metric} loss on the test set is {score:.4f}")
                results[metric] = np.array(score)
        else:
            for i in range(1, cv + 1):
                print(f"Cross validation at split {i}/{cv}")
                data.next()
                self.run_logger.define_cacher(cv=i, improvements_necessary=self.cacher_improvements_necessary)
                encoded_data = Preprocessing(data, self.p)
                epochs, run_batch_size = self.get_epochs_and_run_batch_size(encoded_data, self.params['max_steps'],
                                                                            batch_size)
                self.build_npt(encoded_data)
                self.build_optimizer()
                self.build_loss_function(encoded_data)
                if loading_paths is not None:
                    self.load_from_save(loading_paths[i-1])
                if duplicate:
                    self.duplicate_experiment(encoded_data, epochs, run_batch_size)
                else:
                    self.train(encoded_data, epochs, run_batch_size)
                if duplicate:
                    run_result = self.test_duplicate(encoded_data,batch_size)
                else:
                    run_result = self.test(encoded_data,batch_size)
                for metric, score in run_result.items():
                    print(f"The {metric} loss on the test set is {score:.4f}")
                    if metric not in results.keys():
                        results[metric] = np.array(score)
                    else:
                        results[metric] = np.append(results[metric], score)
        self.calculate_and_save_results(results)

    def calculate_and_save_results(self, results):
        self.run_logger.create_results_file()
        for metric, scores in results.items():
            printable_results = f'''
            Run results for {metric} loss are:
            The mean of the results is {scores.mean():.4f}
            The std of the results is {scores.std():.4f}
            '''
            print(printable_results)
            self.run_logger.save_run_metric_results(printable_results)

    def train(self, encoded_data, epochs, batch_size):
        print(f"number of epochs is {epochs}")
        start = time.time()
        for epoch in range(1, epochs + 1):
            X_train, M_train, train_loss_indices, orig_data = encoded_data.masking('train')
            train = utils_data.TensorDataset(X_train, M_train,train_loss_indices,orig_data)
            train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=False)
            for batch_data in train_loader:
                self.batches_per_epoch += 1
                batch_X, batch_M, batch_loss_indices, batch_real_data = batch_data
                batch_loss = self.pass_through(batch_X, batch_M, batch_loss_indices, batch_real_data)
            if epoch % self.eval_epochs == 0:
                self.model.eval()
                X_val, M_val, val_loss_indices, orig_data = encoded_data.masking('val')
                validation = utils_data.TensorDataset(X_val, M_val,val_loss_indices,orig_data)
                validation_loader = utils_data.DataLoader(validation, batch_size=batch_size, shuffle=False)
                eval_loss = 0
                eval_num = 0
                for batch_data in validation_loader:
                    batch_X, batch_M, batch_loss_indices, batch_real_data = batch_data
                    eval_loss_dict = self.pass_through(batch_X, batch_M, batch_loss_indices, batch_real_data)
                    eval_loss += eval_loss_dict['loss']
                    eval_num += eval_loss_dict['predictions']
                eval_loss /= eval_num
                self.run_logger.check_improvement(self.model, eval_loss, epoch)
                if epoch % self.print_epochs == 0:
                    print(f"Time for {self.print_epochs} epochs is {time.time() - start:.3f} seconds")
                    print(f"Current lr is {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"The validation loss is {eval_loss:.3f} after {epoch} epochs")
                    print(f"Model loss was {batch_loss:.3f} after {epoch} epochs")
                    start = time.time()
                self.model.train()
            self.optimizer.zero_grad()
            self.batches_per_epoch = 0

    def pass_through(self, batch_X, batch_M, loss_indices, batch_real_data):
        if self.model.training:
            z = self.model.forward(batch_X.to(self.device), batch_M.to(self.device))
            batch_loss = self.loss_function.compute(z, batch_real_data.to(self.device), loss_indices.to(self.device))
            batch_loss /= self.step_each_n_batches
            batch_loss.backward()
            if self.batches_per_epoch % self.step_each_n_batches == 0:
                self.step()
            return batch_loss.item()
        else:
            with torch.no_grad():
                z = self.model.forward(batch_X.to(self.device), batch_M.to(self.device))
                batch_loss_dict = self.loss_function.val_loss(z, batch_real_data.to(self.device), loss_indices.to(self.device))
            return batch_loss_dict

    def step(self):
        self.optimizer.step()
        self.lr_scheduler_step()
        self.tradeoff_scheduler_step()
        self.optimizer.zero_grad()

    def lr_scheduler_step(self):
        if self.lr_scheduler == 'flat_then_anneal':
            self.optimizer.flat_then_anneal()
        elif self.lr_scheduler == 'polynomial_decay':
            self.optimizer.polynomial_decay()

    def tradeoff_scheduler_step(self):
        if self.tradeoff_scheduler == 'cosine':
            self.loss_function.Scheduler_cosine_step()

    def test(self, encoded_data,batch_size):
        evaluation_metrics_results = {}
        for eval_metric,_ in self.eval_metrics.items():
            evaluation_metrics_results[eval_metric] = {}
            evaluation_metrics_results[eval_metric]['loss'] = 0
            evaluation_metrics_results[eval_metric]['counter'] = 0
        self.run_logger.load_model(self.model)
        batches = encoded_data.masking('test',batch_size)
        self.model.eval()
        with torch.no_grad():
            for batch_data in batches:
                X_batch, M_batch, batch_loss_indices, orig_data_batch = batch_data
                batch_true_labels = orig_data_batch[:, -1]
                z = self.model.forward(X_batch.to(self.device), M_batch.to(self.device))
                pred = z[encoded_data.data.target_col].detach().cpu()
                if encoded_data.data.target_type == 'continuous':
                    pred *= encoded_data.stats[encoded_data.data.target_col]['std']
                    pred += encoded_data.stats[encoded_data.data.target_col]['mean']
                    batch_true_labels *= encoded_data.stats[encoded_data.data.target_col]['std']
                    batch_true_labels += encoded_data.stats[encoded_data.data.target_col]['mean']
                for eval_metric, eval_metric_instance in self.eval_metrics.items():
                    eval_loss = eval_metric_instance.compute(pred, batch_true_labels, batch_loss_indices[:, -1])
                    evaluation_metrics_results[eval_metric]['loss'] += eval_loss
                    evaluation_metrics_results[eval_metric]['counter'] += batch_loss_indices[:, -1].sum().item()
        for eval_metric in self.eval_metrics.keys():
            evaluation_metrics_results[eval_metric]['loss'] /= evaluation_metrics_results[eval_metric]['counter']
            evaluation_metrics_results[eval_metric] = evaluation_metrics_results[eval_metric]['loss']
        self.model.train()
        return evaluation_metrics_results

    def load_from_save(self,path):
        self.model.load_state_dict(torch.load(path))

    def duplicate_experiment(self, encoded_data, epochs, batch_size):
        start = time.time()
        for epoch in range(1, epochs + 1):
            X_train, M_train, train_loss_indices, orig_data = encoded_data.masking('train')
            train = utils_data.TensorDataset(X_train.to(self.device), M_train.to(self.device),
                                             train_loss_indices.to(self.device),
                                             orig_data.to(self.device))
            train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=False)
            for batch_data in train_loader:
                self.batches_per_epoch += 1
                batch_X_modified, batch_M_modified, train_loss_indices_modified, batch_real_data_modified = self.duplicate(
                    batch_data)
                batch_loss = self.pass_through(batch_X_modified, batch_M_modified, train_loss_indices_modified,
                                               batch_real_data_modified)
            if epoch % self.eval_epochs == 0:
                self.model.eval()
                data_tuple = [item.to(self.device) for item in encoded_data.masking('val')]
                X_val_modified, M_val_modified, val_loss_indices_modified, orig_data_modified = self.duplicate(
                    data_tuple)
                validation = utils_data.TensorDataset(X_val_modified, M_val_modified,val_loss_indices_modified,orig_data_modified)
                validation_loader = utils_data.DataLoader(validation, batch_size=batch_size, shuffle=False)
                eval_loss = 0
                eval_num = 0
                for batch_data in validation_loader:
                    batch_X, batch_M, batch_loss_indices, batch_real_data = batch_data
                    eval_loss_dict = self.pass_through(batch_X, batch_M, batch_loss_indices, batch_real_data)
                    eval_loss += eval_loss_dict['loss']
                    eval_num += eval_loss_dict['predictions']
                eval_loss /= eval_num
                self.run_logger.check_improvement(self.model, eval_loss, epoch)
                if epoch % self.print_epochs == 0:
                    print(f"Time for {self.eval_epochs} epochs is {time.time() - start:.3f} seconds")
                    print(f"Current lr is {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"The validation loss is {eval_loss:.3f} after {epoch} epochs")
                    print(f"Model loss was {batch_loss:.3f} after {epoch} epochs")
                    start = time.time()
                self.model.train()
            self.optimizer.zero_grad()
            self.batches_per_epoch = 0


    def test_duplicate(self, encoded_data,batch_size):
        evaluation_metrics_results = {}
        for eval_metric, _ in self.eval_metrics.items():
            evaluation_metrics_results[eval_metric] = {}
            evaluation_metrics_results[eval_metric]['loss'] = 0
            evaluation_metrics_results[eval_metric]['counter'] = 0
        self.run_logger.load_model(self.model)
        batches = encoded_data.masking('test', batch_size)
        self.model.eval()
        with torch.no_grad():
            for batch_data in batches:
                batch_data = [item.to(self.device) for item in batch_data]
                X_batch_modified, M_batch_modified, batch_loss_indices_modified, orig_data_batch_modified = self.duplicate(batch_data)
                batch_true_labels = orig_data_batch_modified[:, -1]
                z = self.model.forward(X_batch_modified.to(self.device), M_batch_modified.to(self.device))
                pred = z[encoded_data.data.target_col].detach().cpu()
                if encoded_data.data.target_type == 'continuous':
                    pred *= encoded_data.stats[encoded_data.data.target_col]['std']
                    pred += encoded_data.stats[encoded_data.data.target_col]['mean']
                    batch_true_labels *= encoded_data.stats[encoded_data.data.target_col]['std']
                    batch_true_labels += encoded_data.stats[encoded_data.data.target_col]['mean']
                for eval_metric, eval_metric_instance in self.eval_metrics.items():
                    eval_loss = eval_metric_instance.compute(pred, batch_true_labels, batch_loss_indices_modified[:, -1])
                    evaluation_metrics_results[eval_metric]['loss'] += eval_loss
                    evaluation_metrics_results[eval_metric]['counter'] += batch_loss_indices_modified[:, -1].sum().item()
        for eval_metric in self.eval_metrics.keys():
            evaluation_metrics_results[eval_metric]['loss'] /= evaluation_metrics_results[eval_metric]['counter']
            evaluation_metrics_results[eval_metric] = evaluation_metrics_results[eval_metric]['loss']
        self.model.train()
        return evaluation_metrics_results

    def duplicate(self, data_tuple):
        X, M, loss_indices, real_data = data_tuple
        X_modified = torch.cat((X, real_data), 0)
        M_modified = torch.cat((M, torch.zeros(M.shape).to(self.device)), 0)
        loss_indices_modified = torch.cat(
            (loss_indices, torch.zeros(loss_indices.shape).to(self.device)), 0)
        real_data_modified = torch.cat((real_data, real_data), 0)
        return X_modified, M_modified, loss_indices_modified, real_data_modified

    def corruption_experiment(self, data, batch_size, load_path):
        self.run_training(data, batch_size, loading_paths=load_path)
        encoded_data = Preprocessing(data,self.p)
        evaluation_metrics_results = {}
        for eval_metric, _ in self.eval_metrics.items():
            evaluation_metrics_results[eval_metric] = {}
            evaluation_metrics_results[eval_metric]['loss'] = 0
            evaluation_metrics_results[eval_metric]['counter'] = 0
        full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
        full_data = utils_data.TensorDataset(torch.tensor(full_set, requires_grad=False, dtype=torch.float))
        data_loader = utils_data.DataLoader(full_data, batch_size=batch_size, shuffle=True)
        for batch_data in data_loader:
            batch_data = batch_data[0]
            for i in range(len(batch_data)):
                permuted_X, M,label_loss_indices, real_label = permute(batch_data, i)
                z = self.model.forward(permuted_X,M)
                pred = z[data.target_col].detach().cpu()
                if data.target_type == 'continuous':
                    pred *= encoded_data.stats[data.target_col]['std']
                    pred += encoded_data.stats[data.target_col]['mean']
                    real_label *= encoded_data.stats[data.target_col]['std']
                    real_label += encoded_data.stats[data.target_col]['mean']
                for eval_metric, eval_metric_instance in self.eval_metrics.items():
                    eval_loss = eval_metric_instance.compute(pred, real_label, label_loss_indices)
                    evaluation_metrics_results[eval_metric]['loss'] += eval_loss
                    evaluation_metrics_results[eval_metric]['counter'] += label_loss_indices.sum().item()
        for eval_metric in self.eval_metrics.keys():
            evaluation_metrics_results[eval_metric]['loss'] /= evaluation_metrics_results[eval_metric]['counter']
            evaluation_metrics_results[eval_metric] = evaluation_metrics_results[eval_metric]['loss']
        self.model.train()
        results = {}
        for metric, score in evaluation_metrics_results.items():
            results[metric] = np.array(score)
        self.calculate_and_save_results(results)

    def deletion_experiment(self, data, batch_size, load_path):
        self.run_training(data, batch_size, loading_paths=load_path)
        self.model.eval()
        full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
        full_data = torch.tensor(full_set, dtype=torch.float)
        max_change = 0.1
        max_change_per_delete = 0.01
        N_max_retry = 50
        eps = 0.02
        N_retry = 0
        result_dict = {'kept':[],'thrown':[]}
        for i in range(len(full_set)):
            M = torch.zeros(full_data.size())
            M[i,-1] = 1
            X_i_masked = full_data.clone()
            if data.target_type == 'categorical':
                X_i_masked[i,-1] = -1
            else:
                X_i_masked[i,-1] = 0
            row_tested = X_i_masked[i]
            full_data_prediction = self.model.forward(X_i_masked.to(self.device),M.to(self.device))[data.target_col][i].item()

            R = np.concatenate((np.arange(0, i), np.arange(i + 1, len(full_set))))
            while True:
                deleted_index = np.random.randint(low=0, high=len(R))
                new_R = np.delete(R, deleted_index)

                selected_rows = full_data[torch.tensor(new_R).long()]
                X = torch.cat((selected_rows, row_tested.unsqueeze(dim=0)), dim=0)
                M = torch.zeros(X.size())
                M[-1, -1] = 1
                y_proposed = self.model.forward(X.to(self.device), M.to(self.device))[data.target_col][-1].item()
                delta_proposal = abs((y_proposed - full_data_prediction) / full_data_prediction)
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
                if len(R) < eps * len(full_set):
                    break
            result_dict['kept'].append(len(R))
            result_dict['thrown'].append(len(full_set)-len(R))
        self.run_logger.create_results_file()
        wilcoxon_results = wilcoxon(result_dict['kept'], result_dict['thrown'])
        printable_results = f'''Wilcoxon test statistic is {wilcoxon_results[0]} and the p value is {wilcoxon_results[1]}'''
        print(printable_results)
        self.run_logger.save_run_metric_results(printable_results)

