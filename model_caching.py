from loss import Loss
import torch
import os
import time
import datetime
import glob

class Model_Cacher():
    def __init__(self,dataset,cv = None):
        self.min_val_loss = float('inf')
        parent_path = '/home/yehonatan-pe/replication/model_checkpoints/'
        dataset_name = dataset.name
        path = os.path.join(parent_path, dataset_name)
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path,run_time)
        os.mkdir(path)
        if cv is not None:
            path = os.path.join(path, str(cv))
        os.mkdir(path)
        self.caching_path = path
        self.best_model = None


    def check_improvement(self,model,val_loss,epoch):
        if self.min_val_loss > val_loss:
            print("improvement")
            path = os.path.join(self.caching_path,'*')
            files = glob.glob(path)
            for file in files:
                os.remove(file)
            path = os.path.join(self.caching_path,str(epoch))
            self.best_model = path
            torch.save(model.state_dict(), path)
            self.min_val_loss = val_loss
    def load_model(self,model):
        return model.load_state_dict(torch.load(self.best_model))
