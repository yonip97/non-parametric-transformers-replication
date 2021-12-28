import math
import time

import torch
import numpy as np
from torch.nn import functional as F


# from sklearn.preprocessing import OneHotEncoder


# first obtain the mean and std of the numeric data. then build a one hot encoder without the -1 class which represents nan
# save the original data as tensors. the data is normalized by the training set and not one hot encoded.
# create mask and then transform all the columns. the training data has stocastic making while the validation and test set are static.
# after the masks are created we randomize all entries of 2 and zero all entries of 1. while in numeric data the value is set to zero, in categorical columns the entire row is zero, no class speciped.

class Preprocessing():
    def __init__(self, data, p):
        self.data = data
        self.split = data.split
        self.p = p
        self._preprocess()

    def _preprocess(self):
        self._obtain_stats()
        self._standrize_and_save()

    def _obtain_stats(self):
        self.stats = {}
        for col in self.data.continuous:
            self.stats[col] = {}
            self.stats[col]['mean'] = np.nanmean(self.data.train_data[:, col])
            self.stats[col]['std'] = np.nanstd(self.data.train_data[:, col])

    def _standrize_and_save(self):
        train_data = np.copy(self.data.train_data)
        self.orig_train_tensors = self.normalize(train_data)
        self.train_features_size,self.train_labels_size = self.get_sizes(self.orig_train_tensors.size())
        val_data = np.copy(self.data.val_data)
        self.orig_val_tensors = self.normalize(val_data)
        self.val_features_size,self.val_labels_size = self.get_sizes(self.orig_val_tensors.size())
        test_data = np.copy(self.data.test_data)
        self.orig_test_tensors = self.normalize(test_data)
        self.test_features_size,self.test_labels_size = self.get_sizes(self.orig_test_tensors.size())
        self.orig_size = self.orig_train_tensors.size(0)+self.orig_val_tensors.size(0)+self.orig_test_tensors.size(0)


    def normalize(self,data):
        for col in self.data.continuous:
            data[:, col] = (data[:, col] - self.stats[col]['mean'])
            if self.stats[col]['std'] >0:
                data[:,col] /=self.stats[col]['std']
        return torch.tensor(data,dtype=torch.float)

    def get_sizes(self,size):
        features = (size[0],size[1]-1)
        labels = (size[0],1)
        return features,labels

    def masking(self, set,batch_size=None):
        '''
        :param set:is the masking done on the train,validation or test set.
        if the masking is done on the train set then random indices and labels of the training set will be masked,
        while the validation and test tests will have exposed features and masked labels.
        if the masking is done on the validation set, then only the labels of the validaton and test set will be masked
        all else will be exposed
        if the masking is done on the test set then just the test labels will be masked
        :return:
        '''

        # test features are always exposed and test target are always masked

        mask_test_features = np.zeros(self.test_features_size)
        mask_test_labels = np.ones(self.test_labels_size).reshape((-1, 1))
        #  validation features are always exposed
        mask_val_features = np.zeros(self.val_features_size)
        mask_test = np.concatenate((mask_test_features, mask_test_labels), axis=1)
        if set == 'train':
            # if the model is training then we apply stochastic mask on the features an labels of the train set
            mask_train_features = np.random.choice(a=[0, 1, 2], size=self.train_features_size,
                                                   p=[self.p.f_uc, self.p.f_mo, self.p.f_r])
            mask_train_labels = np.random.choice(a=[0, 1], size=self.train_labels_size,
                                                 p=[self.p.l_uc, self.p.l_mo]).reshape((-1, 1))
            mask_train = np.concatenate((mask_train_features, mask_train_labels), axis=1)
            # validation targets are masked
            mask_val_labels = np.ones(self.val_labels_size).reshape((-1, 1))
            mask_val = np.concatenate((mask_val_features, mask_val_labels), axis=1)
            # the loss is calculated only on the features that have been masked out, not on the randomized features
            train_loss_indices = torch.tensor(mask_train == 1)
            # loss is calculated only on the train set. so the loss indices of the validation and test set are 0.
            val_loss_indices = torch.zeros(mask_val.shape)
            test_loss_indices = torch.zeros(mask_test.shape)
        elif set == 'val':
            # the train features and labels are exposed
            mask_train_features = np.zeros(self.train_features_size)
            mask_train_labels = np.zeros(self.train_labels_size).reshape(-1, 1)
            mask_train = np.concatenate((mask_train_features, mask_train_labels), axis=1)
            # validation targets are masked
            mask_val_labels = np.ones(self.val_labels_size).reshape((-1, 1))
            mask_val = np.concatenate((mask_val_features, mask_val_labels), axis=1)
            # loss indices are calculated just on the validation set
            train_loss_indices = torch.zeros(mask_train.shape)
            val_loss_indices = torch.tensor(mask_val.copy())
            test_loss_indices = torch.zeros(mask_test.shape)
        else:
            # the train features and labels are exposed
            mask_train_features = np.zeros(self.train_features_size)
            mask_train_labels = np.zeros(self.train_labels_size).reshape(-1, 1)
            mask_train = np.concatenate((mask_train_features, mask_train_labels), axis=1)
            # validation set targets are exposed
            mask_val_labels = np.zeros(self.val_labels_size).reshape((-1, 1))
            mask_val = np.concatenate((mask_val_features, mask_val_labels), axis=1)
            train_loss_indices = torch.zeros(mask_train.shape)
            val_loss_indices = torch.zeros(mask_val.shape)
            test_loss_indices = torch.tensor(mask_test.copy())
        if set == 'test':
            train_masked,mask_train = self._transform_data(self.orig_train_tensors.clone(),mask_train)
            train_masked,mask_train,train_loss_indices,orig_train_tensors = self.shuffle(train_masked,mask_train,train_loss_indices,self.orig_train_tensors.clone())
            val_masked,mask_val = self._transform_data(self.orig_val_tensors.clone(),mask_val)
            val_masked,mask_val,val_loss_indices,orig_val_tensors = self.shuffle(val_masked,mask_val,val_loss_indices,self.orig_val_tensors.clone())
            test_masked,mask_test = self._transform_data(self.orig_test_tensors.clone(),mask_test)
            test_masked,mask_test,test_loss_indices,orig_test_tensors = self.shuffle(test_masked,mask_test,test_loss_indices,self.orig_test_tensors.clone())
            train_indexes,val_indexes,test_indexes = self.define_mini_batches(len(train_masked),len(val_masked),len(test_masked),batch_size,self.split)
            batches = []
            for indexes in zip(train_indexes,val_indexes,test_indexes):
                masked_batch = self.construct_batches(indexes,(train_masked,val_masked,test_masked))
                batch_mask = self.construct_batches(indexes,(mask_train,mask_val,mask_test))
                batch_loss_indices = self.construct_batches(indexes,(train_loss_indices,val_loss_indices,test_loss_indices))
                batch_orig_data = self.construct_batches(indexes,(orig_train_tensors,orig_val_tensors,orig_test_tensors))
                batches.append((masked_batch,batch_mask,batch_loss_indices,batch_orig_data))
            return batches
        else:
            mask = torch.tensor(np.concatenate((mask_train, mask_val, mask_test), axis=0))
            loss_indices = torch.cat((train_loss_indices, val_loss_indices, test_loss_indices), dim = 0)
            data = torch.cat((self.orig_train_tensors,self.orig_val_tensors,self.orig_test_tensors),dim = 0).clone()
            masked_data, mask = self._transform_data(data.clone(),mask)
            return self.shuffle(masked_data, mask, loss_indices, data)

    def construct_batches(self,indexes,data):
        batch = []
        for index,data in zip(indexes,data):
            batch.append(data[index])
        return torch.cat(batch)

    def define_mini_batches(self,train,val,test,final_batch_size,split):
        train_batch_size = math.ceil(final_batch_size * split['train'])
        val_batch_size = math.ceil(final_batch_size * split['val'])
        test_batch_size = math.ceil(final_batch_size * split['test'])
        train_indexes = list(range(train))
        val_indexes = list(range(val))
        test_indexes = list(range(test))
        train_chunked = []
        val_chunked = []
        test_chunked = []
        for i in range(0,train,train_batch_size):
            train_chunked.append(train_indexes[i:i+train_batch_size])
        for i in range(0,val,val_batch_size):
            val_chunked.append(val_indexes[i:i+val_batch_size])
        for i in range(0,test,test_batch_size):
            test_chunked.append(test_indexes[i:i+test_batch_size])
        return train_chunked,val_chunked,test_chunked



    def shuffle(self, masked_data, mask, loss_indices, data):
        order = torch.randperm(data.size(0))
        return masked_data[order], mask[order], loss_indices[order], data[order]


    def _transform_data(self, data,mask):
        data_dict = {}
        to_zero_indices = mask == 1
        randomize_indices = mask == 2
        for col in self.data.continuous:
            temp = data[:, col]
            to_zero_col = to_zero_indices[:, col]
            randomize_col = randomize_indices[:, col]
            if to_zero_col.sum() != 0:
                temp[to_zero_col] = 0
            if randomize_col.sum() != 0:
                temp[randomize_col] = torch.normal(mean=0, std=1, size=[randomize_col.sum()])
            data_dict[col] = temp
        for col, classes in self.data.categorical.items():
            temp = data[:, col].long()
            to_zero_col = to_zero_indices[:, col]
            randomize_col = randomize_indices[:, col]
            if randomize_col.sum() != 0:
                temp[randomize_col] = torch.randint(low=0, high=classes, size=[randomize_col.sum()])
            if to_zero_col.sum() != 0:
                temp[to_zero_col] = -1
            data_dict[col] = temp
        stacked_data = torch.stack([x[1] for x in sorted(data_dict.items())], dim=1)
        return stacked_data, torch.tensor(mask == 1)

