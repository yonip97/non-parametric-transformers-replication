from npt import NPT
from loss import Loss
import torch
from lamb import Lamb
import time
import torch.utils.data as utils_data
import numpy as np
from evaluation_metrics import *
from util import permute

def train_model(data, model, epochs, optimizer, batch_size, loss_function, eval_metric, eval_each_n_epochs, clip = None, use_lr_scheduler = False, use_tradeoff_scheduler = False):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    start = time.time()
    for epoch in range(epochs):
        X, M = data.create_mask()
        train = utils_data.TensorDataset(X.to(device), M.to(device), torch.tensor(data.train_data,requires_grad=False,dtype=torch.float).to(device))
        train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=True)
        for batch_data in train_loader:
            batch_X, batch_M, batch_real_data = batch_data
            z = model.forward(batch_X, batch_M)
            batch_loss = loss_function.compute(z, batch_real_data, batch_M)
            batch_loss.backward()
            #print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            #print(f"batch time {time.time()-start} seconds")
            if clip != None:
                optimizer.clip_gradient(model, clip)
            if use_lr_scheduler:
                optimizer.flat_then_anneal()
            if use_tradeoff_scheduler:
                loss_function.Scheduler_step()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % eval_each_n_epochs == 0:
            model.eval()
            true_labels = np.concatenate((data.train_data[:, -1], data.val_data[:, -1]))
            eval_X, eval_M = data.mask_targets('val')
            z = model.forward(eval_X.to(device), eval_M.to(device))
            if model.finalize:
                pred = z[:,-1].detach().cpu()
                #eval_acc = acc.compute(pred, true_labels, eval_M[:, -1])
            else:
                pred = z[data.target_col].detach().cpu()
                #eval_acc = acc.compute(pred,true_labels,eval_M[:,-1])
            eval_loss = eval_metric.compute(pred, true_labels, eval_M[:, -1])
            model.train()
            print(f"Epoch {epoch} the rmse loss is {eval_loss}")
            print(f"100 epochs time {time.time() - start}")
            print(f"epoch number {epoch} and the loss is {batch_loss}")
            start = time.time()
            #print(f"Epoch {epoch} the accuracy is {eval_acc}")
    model.eval()
    X_test, M_test = data.mask_targets('test')
    true_labels = np.concatenate((data.train_data[:,-1],data.val_data[:,-1],data.test_data[:,-1]))
    z = model.forward(X_test.to(device),M_test.to(device))
    if model.finalize:
        pred = z[:,-1].detach().cpu()
        #acc_score = acc.compute(pred[:,-1].detach().cpu(),true_labels,M_test[:,-1])
    else:
        pred = z[data.target_col].detach().cpu()
    nll_score = eval_metric.compute(pred, true_labels,M_test[:,-1])
            #acc_score = acc.compute(pred[data.target_col].detach().cpu(), true_labels, M_test[:, -1])
    print(f"rmse score {nll_score}")
    return nll_score


def duplicate_test(data,model,epochs,optimizer,batch_size,loss_function,eval_metric,eval_each_n_epochs,switch_to_pred=False,clip = None,cv_runs = 1):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    for epoch in range(epochs):
        for run in range(cv_runs):
            X, M = data.create_mask('train')
            train = utils_data.TensorDataset(X.to(device).float(), M.to(device).float(), torch.tensor(data.train_data,requires_grad=False).to(device).float())
            train_loader = utils_data.DataLoader(train, batch_size=batch_size, shuffle=True)
            for batch_data in train_loader:
                start = time.time()
                batch_X, batch_M, batch_real_data = batch_data
                batch_X_modified = torch.cat((batch_X,batch_real_data),0)
                batch_M_modified = torch.cat((batch_M,torch.zeros(batch_M.shape).to(device).float()),0)
                batch_real_data_modified = torch.cat((batch_real_data,batch_real_data),0)
                z_modified = model.forward(batch_X_modified, batch_M_modified)
                loss = loss_function.forward(z_modified, batch_real_data_modified, batch_M_modified)
                if loss != -1:
                    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                    loss.backward()
                    if clip != None:
                        optimizer.clip_gradient(model, clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    print(time.time()-start)
                    print(f"epoch number {epoch} and the loss is {loss}")
        if cv_runs != 1:
            data.restart_splitter()
    if switch_to_pred:
        model.eval()
    X_test,M_test = data.mask_targets()
    X_test_modified = torch.cat((X_test,torch.tensor(data.test_data)),0)
    M_test_modified = torch.cat((M_test,torch.zeros(M_test.shape)),0)
    with torch.no_grad():
        pred = model.forward(X_test_modified.to(device),M_test_modified.to(device))
        score = eval_metric.compute(pred[:,-1].detch().cpu().numpy(),data.test_data[:,-1])
        print(score)


def data_corruption(data,model,epochs,optimizer,batch_size,loss_function,eval_metric,eval_each_n_epochs,switch_to_pred=False,clip = None,use_lr_scheduler = False,use_tradeoff_scheduler = False ):
    train_model(data,model,epochs,optimizer,batch_size,loss_function,eval_metric,eval_each_n_epochs,switch_to_pred,clip,use_lr_scheduler,use_tradeoff_scheduler)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
    full_data = utils_data.TensorDataset(torch.tensor(full_set, requires_grad=False, dtype=torch.float))
    data_loader = utils_data.DataLoader(full_data, batch_size=batch_size, shuffle=True)
    for batch_data in data_loader:
        batch_data = batch_data[0]
        for i in range(len(batch_data)):
            permuted_X,M,real_label = permute(batch_data,i)
            if switch_to_pred:
                model.eval()
                z = model.forward(permuted_X.to(device), M.to(device))
                pred= z[-1,data.target_col].detach().cpu().view(1,-1)
                score = eval_metric.compute(pred,real_label.view(1),M[-1,data.target_col].cpu())
                model.train()
            else:
                with torch.no_grad():
                    z = model(permuted_X.to(device), M.to(device))
                    pred = z[data.target_col].detach().cpu()
                    score = eval_metric.compute(pred[-1].view(1,-1), real_label.view(1), M[-1,data.target_col])
            print(f"the nll loss of the sample is {score.item()}")
def data_deletion(data,model,epochs,optimizer,batch_size,loss_function,eval_metric,eval_each_n_epochs,switch_to_pred=False,clip = None,use_lr_scheduler = False,use_tradeoff_scheduler = False ):
    train_model(data,model,epochs,optimizer,batch_size,loss_function,eval_metric,eval_each_n_epochs,switch_to_pred,clip,use_lr_scheduler,use_tradeoff_scheduler)
    model.eval()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    full_set = np.concatenate((data.train_data, data.val_data, data.test_data), axis=0)
    data = torch.tensor(full_set,dtype=torch.float)
    max_change = 0.1
    max_change_per_delete = 0.01
    N_max_retry = 50
    eps = 0.02
    N_retry = 0
    for i in range(len(full_set)):
        y = data[i]
        true_label = y[-1].item()
        y_masked = y.reshape(1,-1)
        y_masked[:,-1] = 0
        R = np.concatenate((np.arange(0,i),np.arange(i+1,len(data))))
        while True:
            deleted_index = np.random.randint(low = 0,high=len(R))
            new_R = np.delete(R,deleted_index)
            selected_rows = data[torch.tensor(new_R).long()]
            X = torch.cat((selected_rows,y_masked),dim=0)
            M = torch.zeros(X.size())
            M[-1,-1] = 1
            y_proposed = model.forward(X.to(device),M.to(device))[-1,-1].item()
            delta_proposal = abs((y_proposed - true_label)/true_label)
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
            if len(R) < eps*len(data):
                break
        return R





