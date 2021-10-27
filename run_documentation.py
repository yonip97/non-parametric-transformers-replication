from model_caching import Model_Cacher
import os
import datetime
import glob
import json
class run_logger():
    def __init__(self,dataset):
        parent_path = '/home/yehonatan-pe/replication/model_runs_information/'
        if not os.path.exists(parent_path):
            os.mkdir(parent_path)
        dataset_name = dataset.name
        path = os.path.join(parent_path, dataset_name)
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path,run_time)
        os.mkdir(path)
        self.run_path = path
        # if cv is not None:
        #     path = os.path.join(path, str(cv))
        # os.mkdir(path)
        # self.model_cacher = Model_Cacher(path,improvements_necessary)
    def record_hyperparameters(self,params):
        path = os.path.join(self.run_path,'run_params.json')
        with open(path,'w') as file:
            json.dump(params,file,indent = 2)
            file.close()


    def define_cacher(self,cv = None,improvements_necessary = 1):
        if cv is not None:
            path = os.path.join(self.run_path, str(cv))
        os.mkdir(path)
        self.model_cacher = Model_Cacher(path,improvements_necessary)


    def check_improvement(self,model,val_loss,epoch):
        if self.model_cacher.check_if_cache(val_loss):
            self.model_cacher.cache(model,val_loss,epoch)

    def load_model(self, model):
        self.model_cacher.load_model(model)

    def save_run_results(self,results):
        path = os.path.join(self.run_path,'results.txt')
        with open(path,'w') as file:
            file.write(results)
            file.close()
