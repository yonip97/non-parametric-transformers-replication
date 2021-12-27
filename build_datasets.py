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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml


class base_dataset():
    def _cutoff(self, df, num_of_cont_features,fixed_test = None):
        '''
        :param df: pandas dataframe containing the data
        :param num_of_cont_features: number of continuous features. If -1 then all are continuous.
        The function is based on the fact that continuous features will have more unique values.
        It defines the continuous features as the features with the most unique features. The rest are categorical
        '''
        df= df.dropna()
        new_cols = {col: new_col for col, new_col in zip(df.columns, range(len(df.columns) + 1))}
        df = df.rename(columns=new_cols)
        unique_values = df.nunique()
        if num_of_cont_features != -1:
            continuous = unique_values.nlargest(num_of_cont_features)
            self.categorical = unique_values.drop(labels=continuous.index).to_dict()
            self.continuous = continuous.index.tolist()
            df[list(self.categorical.keys())] = df[self.categorical.keys()].astype('category')
            df[list(self.categorical.keys())] = df[self.categorical.keys()].apply(lambda x: x.cat.codes)
        else:
            self.categorical = {}
            self.continuous = list(df.columns)
        if fixed_test is None:
            self._create_sets(df.to_numpy(dtype=np.float))
        else:
            fixed_test[list(self.categorical.keys())] = fixed_test[self.categorical.keys()].astype('category')
            fixed_test[list(self.categorical.keys())] = fixed_test[self.categorical.keys()].apply(lambda x: x.cat.codes)
            self._fixed_sets(df.to_numpy(dtype=np.float),fixed_test.to_numpy(dtype = np.float))

    def _manual_preprocessing(self, df,categorical_cols,continuous_cols,fixed_test = None):
        '''
        :param df: pandas dataframe containing the data.
        :param categorical_cols: Columns index of categorical features.
        :param continuous_cols: Columns index of continuous features.
        '''
        df = df.dropna()
        new_cols = {col: new_col for col, new_col in zip(df.columns, range(len(df.columns) + 1))}
        df = df.rename(columns=new_cols)
        self.categorical = df[categorical_cols].nunique().to_dict()
        self.continuous = continuous_cols
        df[categorical_cols] = df[categorical_cols].astype('category')
        df[categorical_cols] = df[categorical_cols].apply(lambda x:x.cat.codes)
        if fixed_test is None:
            self._create_sets(df.to_numpy(dtype=np.float))
        else:
            fixed_test[categorical_cols] = fixed_test[categorical_cols].astype('category')
            fixed_test[categorical_cols] = fixed_test[categorical_cols].apply(lambda x: x.cat.codes)
            self._fixed_sets(df.to_numpy(dtype=np.float),fixed_test)

    def _create_sets(self,data):
        '''
        splits data to training, validation and testing
        '''
        train_val_data, self.test_data = train_test_split(data, test_size=self.split['test'])
        self.train_data,self.val_data = train_test_split(train_val_data,train_size=self.split['train']/(self.split['train']+self.split['val']))

    def _fixed_sets(self,train_val_data,test_data):
        self.test_data = test_data
        self.train_data, self.val_data = train_test_split(train_val_data, train_size=self.split['train'] / (
                    self.split['train'] + self.split['val']))

    def _cv_split(self,cv):
        '''
        :param cv: number of cross validations
        In case of cross validation, creates generator for the cross validation from the training and validation sets.
        '''
        self.orig_train_val_data = np.concatenate((self.train_data,self.val_data))
        self.cv = cv
        splitter = KFold(n_splits=cv)
        self.generator = splitter.split(self.orig_train_val_data)

    def next(self):
        '''
        Resplits data for cross validation
        '''
        train_indices, val_indices = next(self.generator)
        self.val_data = self.orig_train_val_data[val_indices]
        self.train_data = self.orig_train_val_data[train_indices]







class income_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
        path = '/data/shared-data/UCI-income/income_data/census-income.csv'
        df = pd.read_csv(path)
        self.split = {'train': 0.57, 'val': 0.1, 'test': 0.33}
        self._cutoff(df,6)
        self.embedding_dim = embedding_dim
        self.input_dim = 42 * embedding_dim
        self.target_col = 41
        self.target_type = 'categorical'
        self.name = 'income_dataset'
        if cv !=None:
            self._cv_split(cv)




class poker_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
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
        self.target_type = 'categorical'
        self.name = 'poker_dataset'
        if cv != None:
            self._cv_split(cv)


class boson_housing_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv=None):
        features,labels = load_boston(return_X_y=True)
        data = np.hstack((features, labels.reshape((-1, 1))))
        df = pd.DataFrame(data)
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        categorical_cols = [3, 8]
        continuous_cols = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13]
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self.embedding_dim = embedding_dim
        self.input_dim = 14 * embedding_dim
        self.target_col = 13
        self.target_type = 'continuous'
        self.name= 'boston_housing_dataset'
        if cv != None:
            self._cv_split(cv)

class forest_cover_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
        path = '/data/shared-data/UCI-income/forest_cover/covtype.data.gz'
        df = pd.read_csv(path, compression='gzip', header=None)
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._cutoff(df,10)
        self.embedding_dim = embedding_dim
        self.input_dim = 55 * embedding_dim
        self.target_col = 54
        self.target_type = 'categorical'
        self.name = 'forest_cover_dataset'
        if cv != None:
            self._cv_split(cv)

class higgs_boston_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv =None):
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
        self.target_type = 'categorical'
        self.name = 'higgs_boston_dataset'
        if cv != None:
            self._cv_split(cv)

class kick_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
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
        self.target_type = 'categorical'
        self.name = 'kick_dataset'
        if cv != None:
            self._cv_split(cv)

class breast_cancer_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
        features,labels = load_breast_cancer(return_X_y=True)
        data = np.hstack((features,labels.reshape(-1,1)))
        df = pd.DataFrame(data)
        self.split = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        categorical_cols = [30]
        continuous_cols = np.arange(30).tolist()
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self.embedding_dim = embedding_dim
        self.input_dim = 31 * embedding_dim
        self.target_col = 30
        self.target_type = 'categorical'
        self.name = 'breast_cancer_dataset'
        if cv != None:
            self._cv_split(cv)

class protein_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
        path = '/data/shared-data/UCI-income/protein/protein.csv'
        df = pd.read_csv(path)
        temp = df.pop('RMSD')
        df['RMSD'] = temp
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._cutoff(df, -1)
        self.embedding_dim = embedding_dim
        self.input_dim = 10 * embedding_dim
        self.target_col = 9
        self.target_type = 'continuous'
        self.name = 'protein_dataset'
        if cv != None:
            self._cv_split(cv)

class concrete_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
        path = '/data/shared-data/UCI-income/concrete/concrete_data.csv'
        df = pd.read_csv(path)
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._cutoff(df, -1)
        self.embedding_dim = embedding_dim
        self.input_dim = 9 * embedding_dim
        self.target_col = 8
        self.target_type = 'continuous'
        self.name = 'concrete_dataset'
        if cv != None:
            self._cv_split(cv)

class yacht_dataset(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None):
        path = '/data/shared-data/UCI-income/yacht/yacht_hydrodynamics.data'
        df = pd.read_fwf(path,header = None)
        categorical_cols = np.arange(5)
        continuous_cols = [5,6]
        self.split = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        self._manual_preprocessing(df,categorical_cols,continuous_cols)
        self.embedding_dim = embedding_dim
        self.input_dim = 7 * embedding_dim
        self.target_col = 6
        self.target_type = 'continuous'
        self.name = 'yacht_dataset'
        if cv != None:
            self._cv_split(cv)




class MNIST(base_dataset):
    def __init__(self,embedding_dim = 64,cv = None,n_patches = 49):
        x, y = fetch_openml(
            'mnist_784', version=1, return_X_y=True,cache=False)
        self.fixed_test_index = 10000
        y = y.astype(int)
        test_x = x[:self.fixed_test_index]
        test_y = y[:self.fixed_test_index]
        train_x = x[self.fixed_test_index:]
        train_y = y[self.fixed_test_index:]
        train_data = np.concatenate((train_x,train_y.reshape(-1,1)),axis=1)
        test_data = np.concatenate((test_x,test_y.reshape(-1,1)),axis=1)
        test_prop = len(test_data)/len(train_data)+len(test_data)
        self.orig_dim = train_data.shape[1]
        self.split = {'train':0.9*(1-test_prop),'val':0.1*(1-test_prop),'test':test_prop}
        self.categorical = {self.orig_dim-1:10}
        self.continuous = [i for i in range(self.orig_dim-1)]
        self._fixed_sets(train_data,test_data)
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim*(n_patches+1)
        self.target_col = 784
        self.image_channels = 1
        self.n_classes = 10
        self.target_type = 'categorical'
        self.name = 'mnist'


