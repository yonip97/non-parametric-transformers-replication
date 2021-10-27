import torch

from build_datasets import *
from util import probs
from evaluation_metrics import *
from training_constractor import Trainer


def main():
    for seed in np.random.randint(0,100,size=4).tolist():
        torch_seed = seed
        numpy_seed = seed
        torch.manual_seed(torch_seed)
        np.random.seed(numpy_seed)
        dataset = 'breast_cancer'
        label_mask_prob = 1
        features_mask_prob = 0.15
        cv = 10
        data = breast_cancer_dataset(embedding_dim=32,cv= cv)
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
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
        init_tradeoff = 1
        clip = 1
        eval_every_n_th_epoch = 1
        improvements_necessary = 1
        params_dict = {'max_steps':max_steps,'lr':lr,'betas':betas,'eps':eps
                        ,'flat':flat,'k':k,'alpha':alpha,'model_layers':model_layers,'rff':rff_layers,
                       'heads':model_heads,'drop':drop_out,'batch_size':batch_size,'init_tradeoff':init_tradeoff,
                       'clip':clip,'eval_every_n_th_epoch':eval_every_n_th_epoch,'dataset':dataset,
                       'label_mask_prob':label_mask_prob,'features_mask_prob':features_mask_prob,'cv':cv,
                       'torch_seed':torch_seed,'numpy_seed':numpy_seed,'evaluation_metric':'nll',
                       'improvements_necessary':improvements_necessary}
        trainer = Trainer(params_dict=params_dict,eval_metric=nll(),data=data,device=device)
        trainer.run(data,batch_size,cv)














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
