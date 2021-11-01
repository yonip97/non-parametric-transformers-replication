from loss import Loss
import torch
import os
import time
import datetime
import glob

class Model_Cacher():
    def __init__(self,path,improvements_necessary):
        self.min_val_loss = float('inf')
        self.caching_path = path
        self.best_model = None
        self.improvements_necessary = improvements_necessary
        self.improvements_since_last_caching = 0

    def check_if_cache(self, val_loss):
        if self.min_val_loss > val_loss:
            self.improvements_since_last_caching +=1
            print("Improvement in  the model")
            if self.improvements_since_last_caching >= self.improvements_necessary:
                #print("Caching model")
                return True
        return False

    def cache(self,model,val_loss,epoch):
        print("Deleting previous weights")
        path = os.path.join(self.caching_path,'*')
        files = glob.glob(path)
        for file in files:
            os.remove(file)
        print("Caching model")
        path = os.path.join(self.caching_path,str(epoch)+'.pt')
        self.best_model = path
        torch.save(model.state_dict(), path)
        self.min_val_loss = val_loss
        self.improvements_since_last_caching = 0

    def load_model(self,model):
        model.load_state_dict(torch.load(self.best_model))
