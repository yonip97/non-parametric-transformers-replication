from torch import nn
import torch
from torch.nn import functional as F
import random
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
class Flatten(nn.Module):
    @staticmethod
    def forward(X):
        return X.reshape(1,X.shape[0],-1)
class Reshape(nn.Module):
    def __init__(self,input_dim):
        super(Reshape, self).__init__()
        self.E = input_dim
    def forward(self,X):
        return X.reshape(X.size(1),-1,self.E)

class probs():
    def __init__(self, p_features,p_labels):
        assert 0 <= p_labels <= 1
        assert 0 <= p_features <= 1
        self.f_mo = p_features * 0.9
        self.f_r = 0.1 * p_features
        self.f_uc = 1 - p_features
        self.l_mo = p_labels
        self.l_uc = 1 - p_labels

class Input_Embbeding(nn.Module):
    def __init__(self, encoded_data, device):
        '''
        :param categorical: a dictionary containing the columns of the categorical attributes of the data as keys and
        the number of classes in the category as key
        :param continuous: list of the columns of the continuous attributes of the data
        :param embedding_dim: the dimension to the original data is mapped
        :param device: cuda or cpu
        additional parameters:
        type learnable weights which are categorical (index 0) continuous (index 1)
        learnable weights of the columns
        '''
        super(Input_Embbeding, self).__init__()
        self.continuous = encoded_data.data.continuous
        self.categorical = encoded_data.data.categorical
        self.index_embedding = nn.Embedding(len(self.categorical.keys()) + len(self.continuous), encoded_data.data.embedding_dim)
        self.into_cat = nn.ModuleDict({str(col): nn.Linear(classes + 1, encoded_data.data.embedding_dim) for col, classes in self.categorical.items()})
        self.into_cont = nn.ModuleDict({str(col): nn.Linear(2, encoded_data.data.embedding_dim) for col in self.continuous})
        self.device = device
        self.type_embedding = nn.Embedding(2, encoded_data.data.embedding_dim)
        self.cat_index = torch.tensor(0).to(device)
        self.cont_index = torch.tensor(1).to(device)
        self.to(device)


    def forward(self,X,M):
        '''
        :param X: data
        :param M: mask
        :return: embedding of the data
        '''
        encodings = {}
        for col, classes in self.categorical.items():
            encoded_col = torch.zeros((X[:,col].shape[0],classes)).long().to(self.device)
            ind = X[:,col] != -1
            to_encode = X[ind,col].long()
            if to_encode.nelement() != 0:
                encoded_col[ind] = F.one_hot(to_encode,num_classes = classes)
            encodings[col] = self.into_cat[str(col)](torch.cat((encoded_col, M[:, col].unsqueeze(dim=1).long()), dim=1).float())
        for col in self.continuous:
            encodings[col] = self.into_cont[str(col)](torch.stack((X[:, col], M[:, col]), dim=1))
        return encodings


    def change_device(self,device):
        self.device = device
        self.cat_index = self.cat_index.to(self.device)
        self.cont_index = self.cont_index.to(self.device)

class Output_Encoding(nn.Module):
    def __init__(self, encoded_data, device):
        '''
        :param categorical: a dictionary containing the columns of the categorical attributes of the data as keys and
        the number of classes in the category as key
        :param continuous: list of the columns of the continuous attributes of the data
        :param embedding_dim: the dimension to the original data is mapped
        :param device: cuda or cpu
        '''
        super(Output_Encoding, self).__init__()
        self.cat = encoded_data.data.categorical.keys()
        self.cont = encoded_data.data.continuous
        self.out_cat = nn.ModuleDict({str(col): nn.Linear(encoded_data.data.embedding_dim, encoded_data.data.categorical[col]) for col in encoded_data.data.categorical.keys()})
        self.out_cont = nn.ModuleDict({str(col): nn.Linear(encoded_data.data.embedding_dim, 1) for col in encoded_data.data.continuous})
        self.to(device)

    def forward(self, H):
        '''
        :param H: network output
        :return: final output
        '''
        z = {}
        for col in self.cat:
            z[col] = self.out_cat[str(col)](H[:, col, :])
        for col in self.cont:
             z[col] = self.out_cont[str(col)](H[:, col, :]).squeeze()
        return z



def permute(X,i):
    temp_x = X[i, :].unsqueeze(dim = 0)
    real_label = X[i, -1]
    new_X = X[torch.arange(X.size(0)) != i]
    permuted_cols = []
    random_row = random.randint(0,X.size(0))
    for j in range(new_X.size(1)):
        random_permutation = torch.randperm(new_X.size(0))
        col = new_X[random_permutation,j].unsqueeze(dim = 1)
        permuted_cols.append(col)
    new_X = torch.cat(permuted_cols,dim=1)
    part_one_X = new_X[:random_row]
    part_two_X = new_X[random_row:]
    new_X = torch.cat((part_one_X,temp_x,part_two_X),dim=0)
    new_M = torch.zeros(new_X.shape)
    new_M[random_row, -1] = 1
    loss_indices = torch.zeros(X.size(0))
    loss_indices[random_row] = 1
    return new_X,new_M,loss_indices,real_label

class gradient_clipper():
    def __init__(self,clip):
        self.clip = clip
    def clip_gradient(self, model):
        for p in model.parameters():
            p.register_hook(
                lambda grad: torch.clamp(grad, -self.clip, self.clip))
