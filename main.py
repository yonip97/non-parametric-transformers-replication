import time

import numpy as np
import sklearn.model_selection
from torch import nn
import torch
from build_datasets import *
import torch.utils.data as utils_data
from npt import NPT
from lamb import Lamb
#from torch.optim.lr_scheduler import CosineAnnealingLr
from lookahead import Lookahead
from loss import Loss
from evaluation_metrics import acc,nll
from training_constractor import *
from util import probs

def main():
    p = probs(0.15,0.1)
    data = breast_cancer_dataset(embedding_dim=32,p = p)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    Epochs = 2000
    max_steps = 2000
    lr = 5e-4
    betas = (0.9,0.99)
    eps = 1e-6
    flat = 0.5
    k = 6
    alpha = 0.5
    model_layers = 4
    model_heads = 8
    rff_layers = 1
    drop_out = 0.1
    batch_size = -1
    cv = 10
    init_tradeoff = 1
    finalize = False
    params_dict = {'max_steps':max_steps,'lr':lr,'betas':betas,'eps':eps,'flat':flat,'k':k,'alpha':alpha,'model_layers':model_layers,'rff':rff_layers,'heads':model_heads,'drop':drop_out,'init_tradeoff':init_tradeoff,'finalize':finalize}
    trainer = Trainer(params_dict,nll(),100,device,1)
    trainer.run_training(data,batch_size,cv)
    # model = NPT(data.categorical, data.continuous, data.embedding_dim, data.input_dim,model_layers ,model_heads,rff_layers,data.h,device,drop_out,finalize)
    # loss_function = Loss(data,max_steps,init_loss_tradeoff)
    # optimizer = Lamb(model.parameters(),lr = lr,betas=betas,eps=eps)
    # lookahead_optimizer = Lookahead(optimizer,max_steps=max_steps,flat_proportion=flat,k = k,alpha=alpha)
    # Trainer()
    # trainer = Trainer(model,lookahead_optimizer,loss_function,nll(),100,device,1)
    # test_evals = trainer.run_training(data,max_steps,batch_size,cv)
    # print(test_evals)
    # print(f"mean of nlls is {np.mean(test_evals)}")
    # print(f"std of nlls is {test_evals.std()}")
    #duplicate_test(data,model,Epochs,lookahead,2048,loss_function,acc,1,True,1)
    # #flat = Flat_then_Anneal_Scheduler(optimizer,200000,1,0.7)
    # for epoch in range(Epochs):
    #     c = time.time()
    #     X, M = data.create_mask('train')
    #     print(time.time()-c)
    #     train = utils_data.TensorDataset(X.to(device).float(), M.to(device).float(), torch.tensor(data.train_data,requires_grad=False).to(device).float())
    #     train_loader = utils_data.DataLoader(train, batch_size=512, shuffle=True)
    #     for batch_data in train_loader:
    #         start = time.time()
    #         batch_X, batch_M, batch_real_data = batch_data
    #         #model_time = time.time()
    #         z = model.forward(batch_X, batch_M)
    #         #print(f"model time {time.time()-model_time} seconds")
    #         #loss_time = time.time()
    #         loss = loss_function.forward(z, batch_real_data, batch_M)
    #         if loss != -1:
    #             #print(loss)
    #             #print(f"loss time {time.time()-loss_time} seconds")
    #             print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    #             loss.backward()
    #             loss_function.Scheduler_step()
    #             #flat.step()
    #             lookahead.clip_gradient(model, 1)
    #             lookahead.step()
    #             lookahead.zero_grad()
    #             print(time.time()-start)
    #             print(f"epoch number {epoch} and the loss is {loss}")
    # model.eval()
    # X_test,M_test = data.mask_targets()
    # with torch.no_grad():
    #     pred = model.forward(X_test.to(device),M_test.to(device))
    #     score = acc(pred[:,-1],torch.tensor(torch.tensor(data.test_data[:,-1]).to(device)))
    #     print(score)
    #



if __name__ == '__main__':
    main()
