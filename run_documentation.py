from model_caching import Model_Cacher
import os
import datetime
import json


class run_logger():
    def __init__(self,dataset,experiment):
        if experiment is None:
            parent_path = '/home/yehonatan-pe/replication/runs/regular_runs_information'
        else:
            parent_path = f'/home/yehonatan-pe/replication/runs/{experiment}_runs_information'
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


    def define_cacher(self,cv = None,improvements_necessary = 1,is_duplicate = False):
        if is_duplicate:
            self.run_path
        path = self.run_path
        if cv is not None:
            path = os.path.join(path, str(cv))
            os.mkdir(path)
        self.model_cacher = Model_Cacher(path,improvements_necessary)


    def check_improvement(self,model,val_loss,epoch):
        if self.model_cacher.check_if_cache(val_loss):
            self.model_cacher.cache(model,epoch)

    def load_model(self, model):
        self.model_cacher.load_model(model)

    def create_results_file(self):
        self.result_path = os.path.join(self.run_path, 'results.txt')
        with open(self.result_path,'w') as file:
            file.write('The results are:\n')
            file.close()

    def save_run_metric_results(self,results):
        with open(self.result_path,'a') as file:
            file.write(results)
            file.close()
