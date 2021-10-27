import torch
import numpy as np
#from sklearn.preprocessing import OneHotEncoder







# first obtain the mean and std of the numeric data. then build a one hot encoder without the -1 class which represents nan
# save the original data as tensors. the data is normalized by the training set and not one hot encoded.
# create mask and then transform all the columns. the training data has stocastic making while the validation and test set are static.
# after the masks are created we randomize all entries of 2 and zero all entries of 1. while in numeric data the value is set to zero, in categorical columns the entire row is zero, no class speciped.

class Preprocessing():
    def __init__(self,data,p):
        self.data = data
        self.p = p
        self.target_col = data.target_col
        self.target_type = data.target_type
        self.categorical = data.categorical
        self.continuous = data.continuous
        self.embedding_dim = data.embedding_dim
        self.input_dim = data.input_dim
        self._standrize_and_mask()



    def _standrize_and_mask(self):
        self._obtain_stats()
        self._save_original_data_tensors()
        self._create_masks_and_tensors()

    def _obtain_stats(self):
        self.stats = {}
        for col in self.data.continuous:
            self.stats[col] = {}
            self.stats[col]['mean'] =np.nanmean(self.data.train_data[:,col])
            self.stats[col]['std'] = np.nanstd(self.data.train_data[:,col])
        self.encoders = {}
        # for col in self.data.categorical.keys():
        #     encoder = OneHotEncoder(sparse=False,categories='auto',handle_unknown='ignore')
        #     not_nan_indices = ~(self.data.train_data[:,col] == -1)
        #     self.encoders[col] = encoder.fit(self.data.train_data[not_nan_indices, col].reshape(-1, 1))



    def _save_original_data_tensors(self):
        data = np.copy(self.data.train_data)
        data = data[~np.any(data == -1, axis=1)]
        for col in self.data.continuous:
            data[:,col] = (data[:,col]-self.stats[col]['mean'])/self.stats[col]['std']
        self.orig_train_tensors = torch.tensor(data, dtype=torch.float)
        data = np.copy(self.data.train_data)
        data = np.concatenate((data,np.copy(self.data.val_data)))
        for col in self.data.continuous:
            data[:, col] = (data[:, col] - self.stats[col]['mean']) / self.stats[col]['std']
        self.orig_val_tensors = torch.tensor(data, dtype=torch.float)
        data = np.copy(self.data.train_data)
        data = np.concatenate((data, np.copy(self.data.val_data)))
        data = np.concatenate((data, np.copy(self.data.test_data)))
        for col in self.data.continuous:
            data[:, col] = (data[:, col] - self.stats[col]['mean']) / self.stats[col]['std']
        self.orig_test_tensors = torch.tensor(data, dtype=torch.float)

    def _create_masks_and_tensors(self):
        features = self.orig_train_tensors[:, :-1]
        labels = self.orig_train_tensors[:, -1]
        mask_features = np.random.choice(a=[0, 1, 2], size=features.size(),
                                         p=[self.p.f_uc, self.p.f_mo, self.p.f_r])
        mask_labels = np.random.choice(a=[0, 1], size=labels.size(), p=[self.p.l_uc, self.p.l_mo])
        train_mask = np.concatenate((mask_features, mask_labels.reshape(-1, 1)), axis=1)
        self.train_tensors,self.train_M = self._transform_data(self.orig_train_tensors, train_mask)
        val_mask = np.zeros(self.data.val_data.shape)
        val_mask[:,-1] = 1
        val_mask = np.concatenate((np.zeros(self.data.train_data.shape),val_mask))
        val_mask = self._find_nan(self.orig_val_tensors, val_mask)
        self.val_tensors,self.val_M = self._transform_data(self.orig_val_tensors, val_mask)
        test_mask = np.zeros(self.data.test_data.shape)
        test_mask[:,-1] = 1
        test_mask = np.concatenate((np.zeros(val_mask.shape),test_mask))
        test_mask = self._find_nan(self.orig_test_tensors, test_mask)
        self.test_tensors,self.test_M = self._transform_data(self.orig_test_tensors, test_mask)

    def _find_nan(self, data, mask):
        for col in self.data.continuous:
            nan_indices = torch.isnan(data[:,col])
            mask[nan_indices,col] = 1
        for col in self.data.categorical.keys():
            nan_indices = data[:,col] == -1
            mask[nan_indices, col] = 1
        return mask

    def _transform_data(self, data, mask):
        data_dict = {}
        for col in self.data.continuous:
            temp = data[:,col].clone()
            to_zero_indices = mask[:,col] == 1
            randomize_indices = mask[:,col] == 2
            if sum(to_zero_indices) != 0:
                temp[to_zero_indices] = 0
            if sum(randomize_indices) != 0:
                temp[randomize_indices] = torch.normal(mean = 0,std = 1,size = (1,sum(randomize_indices)))
            data_dict[col] = temp
        for col,classes in self.data.categorical.items():
            temp = data[:,col].long().clone()
            to_zero_indices = mask[:,col] == 1
            randomize_indices = mask[:,col] == 2
            if sum(randomize_indices) != 0:
                temp[randomize_indices] = torch.randint(low = 0,high = classes,size=(1,sum(randomize_indices)))
            if sum(to_zero_indices) != 0:
                temp[to_zero_indices] = -1
            data_dict[col] = temp
            #data_dict[col] = torch.tensor(self.one_hot_encoders[col].transform(temp.view(-1,1)),dtype=torch.long)
        stacked_data = torch.stack([x[1] for x in sorted(data_dict.items())],dim=1)
        return stacked_data,torch.tensor(mask == 1)

    def get(self,set):
        if set == 'train':
            return self.train_tensors,self.train_M,self.orig_train_tensors
        elif set == 'val':
            return self.val_tensors,self.val_M,self.orig_val_tensors
        elif set == 'test':
            return self.test_tensors,self.test_M,self.orig_test_tensors
        else:
            raise ValueError("the set must be train,val or test")
    def test(self):
        loss_train_mask = torch.zeros(self.orig_test_tensors.size())
        # features = self.orig_train_tensors[:, :-1]
        # labels = self.orig_train_tensors[:, -1]
        #
        # mask_features = np.random.choice(a=[0, 1, 2], size=features.size(),
        #                                  p=[self.p.f_uc, self.p.f_mo, self.p.f_r])
        # mask_labels = np.random.choice(a=[0, 1], size=labels.size(), p=[self.p.l_uc, self.p.l_mo])
        # train_masking = np.concatenate((mask_features, mask_labels.reshape(-1, 1)), axis=1)
        loss_train_mask[:self.train_M.shape[0]] = self.train_M
        train_mask = loss_train_mask.clone()
        train_mask[self.train_M.shape[0]:,-1] = 1
        loss_val_mask = torch.zeros(self.orig_test_tensors.size())
        loss_val_mask[:self.val_M.shape[0]] = self.val_M
        val_mask = loss_val_mask.clone()
        val_mask[self.val_M.shape[0]:,-1] = 1
        train_X,train_M = self._transform_data(self.orig_test_tensors,train_mask)
        val_X,val_M = self._transform_data(self.orig_test_tensors,val_mask)
        return train_X,train_M,loss_train_mask,val_X,val_M,loss_val_mask,self.orig_test_tensors

        # M = torch.zeros(self.orig_test_tensors.shape())
        # M[:,-1] = 1

    # def create_masks(self):
    #     features = self.orig_train_tensor[:,:-1]
    #     labels = self.orig_train_tensor[:,-1]
    #     mask_features = np.random.choice(a=[0,1,2], size=features.size(),
    #                                     p=[self.p.f_uc,self.p.f_mo, self.p.f_r])
    #     mask_labels = np.random.choice(a=[0, 1], size=labels.size(),p=[self.p.l_uc,self.p.l_mo])
    #     train_mask = np.concatenate((mask_features,mask_labels.reshape(-1,1)),axis = 1)
    #     train_dict = {}
    #     for col in self.data.continuous:
    #         temp = self.orig_train_tensor[:,col]
    #         to_zero_indices = train_mask[:,col] == 1
    #         randomize_indices = train_mask[:,col] == 2
    #         if sum(to_zero_indices) != 0:
    #             temp[to_zero_indices] = 0
    #         if sum(randomize_indices) != 0:
    #             temp[randomize_indices] = torch.normal(mean = 0,std = 1,size = (1,sum(randomize_indices)))
    #         train_dict[col] = temp
    #     for col,classes in self.data.categorical.items():
    #         temp = self.orig_train_tensor[:,col].long()
    #         to_zero_indices = train_mask[:,col] == 1
    #         randomize_indices = train_mask[:,col] == 2
    #         if sum(randomize_indices) != 0:
    #             temp[randomize_indices] = torch.randint(low = 0,high = classes,size=(1,sum(randomize_indices)))
    #         if sum(to_zero_indices) != 0:
    #             temp[to_zero_indices] = -1
    #         train_dict[col] = torch.tensor(self.one_hot_encoders[col].transform(temp.view(-1,1)),dtype=torch.long)
    #     M = train_mask == 1
    #     self.train_X = train_dict
    #     self.train_M = torch.tensor(M)
    #     val_mask = np.zeros(self.data.val_data.shape)
    #     val_mask[:,-1] = 1
    #     val_mask = np.concatenate((np.zeros(self.data.train_data.shape),val_mask))
    #     val_dict = {}
    #     for col in self.data.continuous:
    #         nan_indices = torch.isnan(self.orig_val_tensor[:,col])
    #         val_mask[nan_indices,col] = 1
    #         temp = self.orig_val_tensor[:,col]
    #         to_zero_indices = val_mask[:, col] == 1
    #         temp[to_zero_indices] = 0
    #         val_dict[col] = temp
    #     for col in self.data.categorical.keys():
    #         nan_indices = self.orig_val_tensor[:,col] == -1
    #         val_mask[nan_indices,col] = 1
    #         temp = self.orig_val_tensor[:, col]
    #         val_dict[col] = torch.tensor(self.one_hot_encoders[col].transform(temp.view(-1, 1)), dtype=torch.long)
    #     test_mask = np.zeros(self.data.test_data.shape)
    #     test_mask[:,-1] = 1
    #     test_mask = np.concatenate((np.zeros(val_mask.shape),test_mask))
    #     test_dict = {}
    #     for col in self.data.categorical.keys():
    #         nan_indices = self.orig_test_tensor[:,col] == -1
    #         test_mask[nan_indices,col] = 1
    #         temp = self.orig_test_tensor[:,col].long()
    #         to_zero_indices = test_mask[:, col] == 1
    #         temp[to_zero_indices] = -1
    #         test_dict[col] = torch.tensor(self.one_hot_encoders[col].transform(temp.view(-1,1)),dtype=torch.long)
    #     for col in self.data.categorical.keys():
    #         nan_indices = self.orig_test_tensor[:,col] == -1
    #         test_mask[nan_indices,col] = 1
    #         temp = self.orig_test_tensor[:, col]
    #         to_zero_indices = test_mask[:, col] == 1
    #         temp[to_zero_indices] = 0
    #         test_dict[col] = temp
    #     y = 6










# class Encoding():
#     def __init__(self,data,p):
#         self.data = data
#         self.p = p
#
#     def encode_and_mask(self):
#         train_mask = self.create_train_mask()
#         val_mask = self.create_static_mask()
#         test_mask = self.create_static_mask(True)
#         self.obtain_stats()
#         train_X,train_M = self.transform(self.data.train_data,train_mask)
#         val_X,val_M = self.transform(self.data.val_data,val_mask)
#         test_X,test_M = self.transform(self.data.test_data,test_mask)
#         self.train_data_tensor = train_X
#         self.train_mask_tensor = train_M
#         self.val_data_tensor = torch.cat((torch.tensor(train_data),val_X))
#         self.val_mask_tesnor = torch.cat((torch.zeros(self.train_mask_tensor),val_M))
#         self.test_data_tensor = torch.cat((torch.tensor(train_data),torch.tensor(val_data),test_X))
#         self.test_mask_tensor = torch.tensor(torch.zeros(self.val_mask_tesnor.shape),test_M)
#
#     def create_train_mask(self):
#         features = self.data.train_data[:,:-1]
#         labels = self.data.train_data[:,-1]
#         flat_features = features.flatten()
#         feature_mask = np.full(flat_features.shape,1)
#         mask_features = np.random.choice(a=[0,1,2], size=np.count_nonzero(~np.isnan(flat_features)),
#                                         p=[self.p.f_uc,self.p.f_mo, self.p.f_r])
#         feature_mask[~np.isnan(flat_features)] = mask_features
#         feature_mask = feature_mask.reshape(features.shape)
#         flat_labels = labels.flatten()
#         labels_mask = np.full(flat_labels.shape,1)
#         mask_labels = np.random.choice(a=[0,1], size=np.count_nonzero(~np.isnan(flat_labels)),
#                                        p=[self.p.l_uc,self.p.l_mo])
#         labels_mask[~np.isnan(flat_labels)] = mask_labels
#         labels_mask = labels_mask.reshape((labels.shape[0],1))
#         mask = np.concatenate((feature_mask,labels_mask),axis = 1)
#         return mask
#
#     def create_static_mask(self,test = False):
#         labels_mask = np.zeros((self.data.train_data.shape[0]))
#         features = np.concatenate((self.data.train_data[:,:-1],self.data.val_data[:,:-1]))
#         if not test:
#             labels_mask = np.concatenate((labels_mask,np.ones((self.data.val_data.shape[0]))))
#         else:
#             features = np.concatenate((features,self.data.test_data[:,:-1]))
#             labels_mask = np.concatenate((labels_mask,np.zeros((self.data.val_data.shape[0]))))
#             labels_mask = np.concatenate((labels_mask,np.ones((self.data.test_data.shape[0]))))
#         labels_mask = labels_mask.reshape((labels_mask.shape[0],1))
#         flat_features = features.flatten()
#         feature_mask = np.full(flat_features.shape,1)
#         feature_mask[~np.isnan(flat_features)] = 0
#         feature_mask = feature_mask.reshape(features.shape)
#         return np.concatenate((feature_mask, labels_mask), axis=1)
#
#     def obtain_stats(self):
#         self.stats = {}
#         for col in self.data.continuous:
#             self.stats[col] = {}
#             self.stats[col]['mean'] = np.nanmean(self.data.train_data[:,col])
#             self.stats[col]['std'] = np.nanstd(self.data.train_data[:,col])
#         self.one_hot_encoders = {}
#         for col in self.data.categorical.keys():
#             encoder = OneHotEncoder(sparse=False,categories='auto',handle_unknown='ignore')
#             not_non_indices = ~np.isnan(self.data.train_data[:,col])
#             self.one_hot_encoders[col] = encoder.fit(self.data.train_data[not_non_indices,col].reshape(-1,1))
#
#     def apply(self,data,mask):
#         data_dict = {}
#         for col in self.data.continuous:
#             normalize_indices = mask[:,col] == 0
#             zero_indices = mask[:,col] == 1
#             random_indices = mask[:,col] == 2
#             data[normalize_indices,col] = (data[normalize_indices,col]-self.stats[col]['mean'])/self.stats[col]['std']
#             data[zero_indices,col] = 0
#             data[random_indices,col] = np.random.normal(0,1,sum(random_indices))
#         for col in self.data.categorical:
#             data_dict[col] = self.one_hot_encoders[col].transform(data[:,col].reshape(-1,1))
#         return data_dict
#
#     def transform(self,orig_data,mask):
#         data = np.copy(orig_data)
#         data_dict = self.apply(data,mask)
#
#
#
#     # def create_training_mask(self,orig_data):
#     #     '''
#     #     unchanged value get mask value of 0.
#     #     masked out value of 1
#     #     randomized value of 2
#     #     :return:
#     #     '''
#     #
#     #
#     #     data = np.copy(orig_data)
#     #     features = data[:,:-1]
#     #     labels = data[:,-1]
#     #     flat_features = features.flatten()
#     #     feature_mask = np.full(flat_features.shape,1)
#     #     mask_features = np.random.choice(a=[0,1,2], size=np.count_nonzero(~np.isnan(flat_features)),
#     #                                     p=[self.p.f_uc,self.p.f_mo, self.p.f_r])
#     #     feature_mask[~np.isnan(flat_features)] = mask_features
#     #     feature_mask = feature_mask.reshape(features.shape)
#     #     flat_labels = labels.flatten()
#     #     labels_mask = np.full(flat_labels.shape,1)
#     #     mask_labels = np.random.choice(a=[0,1], size=np.count_nonzero(~np.isnan(flat_labels)),
#     #                                    p=[self.p.l_uc,self.p.l_mo])
#     #     labels_mask[~np.isnan(flat_labels)] = mask_labels
#     #     labels_mask = labels_mask.reshape(labels.shape)
#     #     features[feature_mask == 1] = 0
#     #     feature_indices = feature_mask == 2
#     #     if self.data.target_type == 'categorical':
#     #         labels_classes = self.data.categorical.pop(self.data.target_col, None)
#     #     else:
#     #         self.data.continuous.remove(self.data.target_col)
#     #     for category in self.data.continuous:
#     #         arr_size = np.count_nonzero(feature_indices[:, category] == True)
#     #         features[feature_indices[:, category], category] = np.random.normal(0, 1, arr_size)
#     #     for category, classes in self.data.categorical.items():
#     #         arr_size = np.count_nonzero(feature_indices[:, category] == True)
#     #         features[feature_indices[:, category], category] = np.random.randint(low=0, high=classes, size=arr_size)
#     #     labels[labels_mask == 1] = 0
#     #     if self.data.target_type == 'categorical':
#     #         self.data.categorical[self.data.target_col] = labels_classes
#     #     else:
#     #         self.data.continuous.append(self.data.target_col)
#     #     X = np.hstack((features, labels.reshape((-1, 1))))
#     #     mask = np.hstack((feature_mask, labels_mask.reshape((-1, 1))))
#     #     M = mask == 1
#     #     return torch.tensor(X,dtype=torch.float), torch.tensor(M,dtype=torch.float)
#
#     def mask_col(self,orig_data,col):
#         data = np.copy(orig_data)
#         mask_features = np.random.choice(a=[0,1,2], size=orig_data.shape[0],
#                                         p=[self.p.f_uc,self.p.f_mo, self.p.f_r])
#         data[mask_features == 1] = 0
#         feature_indices = mask_features == 2
#
#     def mask_targets(self,data):
#         X = np.copy(data)
#         if self.data.target_type == 'categorical':
#             X[:,:,-1] = 0
#         elif self.data.target_type == 'continuous':
#             X[:,-1] = 0
#         M = np.zeros(X.shape[:2])
#         M[:,-1] = 1
#         return torch.tensor(X,dtype=torch.float),torch.tensor(M,dtype=torch.float)
#
#
#         #
#         #
#         # if set == 'val':
#         #     val_data[:,-1] = 0
#         #     val_M[:,-1] = 1
#         #     X = np.concatenate((train_data,val_data))
#         #     M = np.concatenate((train_M,val_M))
#         # else:
#         #     test_data = np.copy(self.test_data)
#         #     test_M = np.zeros(self.test_data.shape)
#         #     test_data[:,-1] = 0
#         #     test_M[:,-1] = 1
#         #     X = np.concatenate((train_data,val_data,test_data))
#         #     M = np.concatenate((train_M,val_M,test_M))
#         # return torch.tensor(X,dtype=torch.float), torch.tensor(M,dtype=torch.float)
