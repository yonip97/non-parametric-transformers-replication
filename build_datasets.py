import time

import numpy
import numpy as np
import pandas as pd
import sklearn.datasets
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston,load_breast_cancer
from sklearn.model_selection import KFold
from util import probs



class base_dataset():
    def _cutoff(self,df, cont_features):
        new_cols = {col: new_col for col, new_col in zip(df.columns, range(len(df.columns) + 1))}
        df = df.rename(columns=new_cols)
        unique_values = df.nunique()
        if cont_features != -1:
            continuous = unique_values.nlargest(cont_features)
            self.categorical = unique_values.drop(labels=continuous.index).to_dict()
            self.continuous = continuous.index.tolist()
            df[list(self.categorical.keys())] = df[self.categorical.keys()].astype('category')
            df[list(self.categorical.keys())] = df[self.categorical.keys()].apply(lambda x: x.cat.codes)
            df[list(self.categorical.keys())] = df[list(self.categorical.keys())].replace(to_replace=-1, value=np.nan)
        else:
            self.categorical = {}
            self.continuous = list(df.columns)
        self._create_sets(df.to_numpy())

    def _manual_preprocessing(self, df,categorical_cols,continuous_cols):
        new_cols = {col: new_col for col, new_col in zip(df.columns, range(len(df.columns) + 1))}
        df = df.rename(columns=new_cols)
        self.categorical = df[categorical_cols].nunique().to_dict()
        self.continuous = continuous_cols
        df[categorical_cols] = df[categorical_cols].astype('category')
        df[categorical_cols]  = df[categorical_cols].apply(lambda x:x.cat.codes)
        df[categorical_cols] = df[categorical_cols].replace(to_replace = -1,value = np.nan)
        self._create_sets(df.to_numpy(dtype = np.float))

    def _create_sets(self,data):
        train_val_data, self.test_data = train_test_split(data, test_size=self.split['test'])
        self.train_data, self.val_data = train_test_split(train_val_data, train_size=self.split['train'] / (
                self.split['val'] + self.split['train']))

    def _cv_split(self,cv):
        self.train_val_data = np.concatenate((self.train_data,self.val_data))
        self.cv = cv
        splitter = KFold(n_splits=cv)
        self.generator = splitter.split(self.train_val_data)


    def next(self):
        train_indices, val_indices = next(self.generator)
        self.val_data = self.train_val_data[val_indices]
        self.train_data = self.train_val_data[train_indices]

    def restart_splitter(self):
        splitter = KFold(n_splits=self.cv)
        self.generator = splitter.split(self.train_val_data)

    def create_mask(self):
        '''
        unchanged value get mask value of 0.
        masked out value of 1
        randomized value of 2
        :return:
        '''
        features = np.copy(self.train_data[:,:-1])
        labels = np.copy(self.train_data[:,-1])
        if self.target_type == 'categorical':
            labels_classes = self.categorical.pop(self.target_col, None)
        else:
            self.continuous.remove(self.target_col)
        flat_features = features.flatten()
        feature_mask = np.full(flat_features.shape,1)
        mask_features = np.random.choice(a=[0,1,2], size=np.count_nonzero(~np.isnan(flat_features)),
                                        p=[self.p.f_uc,self.p.f_mo, self.p.f_r])
        feature_mask[~np.isnan(flat_features)] = mask_features
        feature_mask = feature_mask.reshape(features.shape)
        flat_labels = labels.flatten()
        labels_mask = np.full(flat_labels.shape,1)
        mask_labels = np.random.choice(a=[0,1], size=np.count_nonzero(~np.isnan(flat_labels)),
                                       p=[self.p.l_uc,self.p.l_mo])
        labels_mask[~np.isnan(flat_labels)] = mask_labels
        labels_mask = labels_mask.reshape(labels.shape)
        features[feature_mask == 1] = 0
        feature_indices = feature_mask == 2
        for category in self.continuous:
            arr_size = np.count_nonzero(feature_indices[:, category] == True)
            features[feature_indices[:, category], category] = np.random.normal(0, 1, arr_size)
        for category, classes in self.categorical.items():
            arr_size = np.count_nonzero(feature_indices[:, category] == True)
            features[feature_indices[:, category], category] = np.random.randint(low=0, high=classes, size=arr_size)
        labels[labels_mask == 1] = 0
        label_indices = labels_mask == 2
        arr_size = np.count_nonzero(label_indices == True)
        if self.target_type == 'categorical':
            labels[label_indices] = np.random.randint(low=0, high=labels_classes, size=arr_size)
            self.categorical[self.target_col] = labels_classes
        else:
            labels[label_indices] = np.random.normal(0, 1, arr_size)
            self.continuous.append(self.target_col)
        X = np.hstack((features, labels.reshape((-1, 1))))
        mask = np.hstack((feature_mask, labels_mask.reshape((-1, 1))))
        M = mask == 1
        return torch.tensor(X,dtype=torch.float), torch.tensor(M,dtype=torch.float)


    def mask_targets(self,set):
        val_data = np.copy(self.val_data)
        val_M = np.zeros(self.val_data.shape)
        train_data = np.copy(self.train_data)
        train_M = np.zeros(self.train_data.shape)
        if set == 'val':
            val_data[:,-1] = 0
            val_M[:,-1] = 1
            X = np.concatenate((train_data,val_data))
            M = np.concatenate((train_M,val_M))
        else:
            test_data = np.copy(self.test_data)
            test_M = np.zeros(self.test_data.shape)
            test_data[:,-1] = 0
            test_M[:,-1] = 1
            X = np.concatenate((train_data,val_data,test_data))
            M = np.concatenate((train_M,val_M,test_M))
        return torch.tensor(X,dtype=torch.float), torch.tensor(M,dtype=torch.float)



class income_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/income_data/census-income.csv'
        df = pd.read_csv(path)
        self.split = {'train': 0.57, 'val': 0.1, 'test': 0.33}
        self._cutoff(df,6)
        self.embedding_dim = embedding_dim
        self.input_dim = 42 * embedding_dim
        self.target_col = 41
        self.h = 42
        self.target_type = 'categorical'
        self.p = p




class poker_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/poker_hand/poker-hand-testing.data'
        poker_1 = pd.read_csv(path,header=None)
        path = '/data/shared-data/UCI-income/poker_hand/poker-hand-training-true.data'
        poker_2 = pd.read_csv(path,header=None)
        df = pd.concat([poker_1,poker_2])
        self.split = {'train': 0.017, 'val': 0.003, 'test': 0.98}
        categorical_cols = [10]
        continuous_cols = np.arange(10).tolist()
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self.embedding_dim = embedding_dim
        self.input_dim = 11 * embedding_dim
        self.target_col = 10
        self.h = 11
        self.target_type = 'categorical'
        self.p = p



class boson_housing_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,p = probs(0.15,0.9,0.15)):
        features,labels = load_boston(return_X_y=True)
        data = np.hstack((features, labels.reshape((-1, 1))))
        df = pd.DataFrame(data)
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        categorical_cols = [3, 8]
        continuous_cols = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13]
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self._cv_split(10)
        self.embedding_dim = embedding_dim
        self.input_dim = 14 * embedding_dim
        self.target_col = 13
        self.h = 14
        self.target_type = 'continuous'
        self.p = p


class forest_cover_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/forest_cover/covtype.data.gz'
        df = pd.read_csv(path, compression='gzip', header=None)
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._cutoff(df,10)
        self.embedding_dim = embedding_dim
        self.input_dim = 55 * embedding_dim
        self.target_col = 54
        self.h = 55
        self.target_type = 'categorical'
        self.p = p

class higgs_boston_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/boston_higgs/HIGGS.csv.gz'
        df = pd.read_csv(path,compression='gzip',header=None)
        temp = df.pop(0)
        df[29] = temp
        self.split = {'train': 0.84, 'val': 0.12, 'test': 0.04}
        categorical_cols = [28]
        continuous_cols = np.arange(28).tolist()
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self.embedding_dim = embedding_dim
        self.input_dim = 29 * embedding_dim
        self.target_col = 28
        self.h = 29
        self.target_type = 'categorical'
        self.p = p


class kick_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/kick/kick.csv'
        df = pd.read_csv(path,header=None,skiprows=1)
        temp = df.pop(0)
        df[33] = temp
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        categorical_cols = [1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 24, 25, 26, 27, 28, 30, 32]
        continuous_cols = [0, 2, 3, 12, 16, 17, 18, 19, 20, 21, 22, 23, 29, 31]
        df = df.replace({'?':np.nan})
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self.embedding_dim = embedding_dim
        self.input_dim = 33 * embedding_dim
        self.target_col = 32
        self.h = 33
        self.target_type = 'categorical'
        self.p = p

class breast_cancer_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,p = probs(0.15,0.9,0.15)):
        features,labels = load_breast_cancer(return_X_y=True)
        data = np.hstack((features,labels.reshape(-1,1)))
        df = pd.DataFrame(data)
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        categorical_cols = [30]
        continuous_cols = np.arange(30).tolist()
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self._cv_split(10)
        self.embedding_dim = embedding_dim
        self.input_dim = 31 * embedding_dim
        self.target_col = 30
        self.h = 31
        self.target_type = 'categorical'
        self.p = p

class protein_dataset(base_dataset):
    def __init__(self,embedding_dim = 64, p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/protein/protein.csv'
        df = pd.read_csv(path)
        temp = df.pop('RMSD')
        df['RMSD'] = temp
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._cutoff(df, -1)
        self.embedding_dim = embedding_dim
        self.input_dim = 10 * embedding_dim
        self.target_col = 9
        self.h = 10
        self.target_type = 'continuous'
        self.p = p

class concrete_dataset(base_dataset):
    def __init__(self,embedding_dim = 64, p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/concrete/concrete_data.csv'
        df = pd.read_csv(path)
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._cutoff(df, -1)
        self._cv_split(10)
        self.embedding_dim = embedding_dim
        self.input_dim = 9 * embedding_dim
        self.target_col = 8
        self.h = 9
        self.target_type = 'continuous'
        self.p = p

class yacht_dataset(base_dataset):
    def __init__(self,embedding_dim = 64, p = probs(0.15,0.9,0.15)):
        path = '/data/shared-data/UCI-income/yacht/yacht_hydrodynamics.data'
        df = pd.read_fwf(path,header = None)
        categorical_cols = np.arange(5)
        continuous_cols = [5,6]
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self._cv_split(10)
        self.embedding_dim = embedding_dim
        self.input_dim = 7 * embedding_dim
        self.target_col = 6
        self.h = 7
        self.target_type = 'continuous'
        self.p = p





