from torch import nn
import torch
from torch.nn import functional as F
import random

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
        self.f_uc = 1 - self.f_r - self.f_mo
        self.l_mo = p_labels * 0.9
        self.l_r = p_labels * 0.1
        self.l_uc = 1 - self.l_mo - self.l_r

class Input_Embbeding(nn.Module):
    def __init__(self, categorical: dict, continuous, embedding_dim, device):
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
        self.continuous = continuous
        self.categorical = categorical
        self.index_embedding = nn.Embedding(len(categorical.keys()) + len(continuous), embedding_dim)
        self.into_cat = nn.ModuleDict({str(col): nn.Linear(classes + 1, embedding_dim) for col, classes in categorical.items()})
        self.into_cont = nn.ModuleDict({str(col): nn.Linear(2, embedding_dim) for col in continuous})
        self.device = device
        self.type_embedding = nn.Embedding(2, embedding_dim)
        self.cat_index = torch.tensor(0).to(device)
        self.cont_index = torch.tensor(1).to(device)
        self.to(device)

    @staticmethod
    def standardize(X):
        '''
        :param X:input data
        :return: standardize data
        '''
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True)
        if std.item() != 0:
            return (X - mean) / std
        else:
            return X-mean

    def forward(self,X,M):
        '''
        :param X: data
        :param M: mask
        :return: embedding of the data
        '''
        encodings = {}
        for col, classes in self.categorical.items():
            encodings[col] = self.into_cat[str(col)](
                torch.cat((F.one_hot(X[:, col].long(), classes), M[:, col].unsqueeze(dim=1).long()), dim=1).float())
            encodings[col] += self.index_embedding(torch.tensor(col).to(self.device)) + self.type_embedding(self.cat_index)
        for col in self.continuous:
            encodings[col] = self.into_cont[str(col)](torch.stack((self.standardize(X[:, col]), M[:, col]), dim=1))
            encodings[col] += self.index_embedding(torch.tensor(torch.tensor(col).to(self.device))) + self.type_embedding(self.cont_index)
        encodings_list = sorted([(key, item) for key, item in encodings.items()])
        encodings_list = [item[1] for item in encodings_list]
        return torch.stack(encodings_list, dim=1)


class Output_Encoding(nn.Module):
    def __init__(self, categorical: dict, continuous, embedding_dim,finalize, device):
        '''
        :param categorical: a dictionary containing the columns of the categorical attributes of the data as keys and
        the number of classes in the category as key
        :param continuous: list of the columns of the continuous attributes of the data
        :param embedding_dim: the dimension to the original data is mapped
        :param device: cuda or cpu
        '''
        super(Output_Encoding, self).__init__()
        self.cat = categorical.keys()
        self.cont = continuous
        self.out_cat = nn.ModuleDict({str(col): nn.Linear(embedding_dim, categorical[col]) for col in categorical.keys()})
        self.out_cont = nn.ModuleDict({str(col): nn.Linear(embedding_dim, 1) for col in continuous})
        self.finalize = finalize
        self.to(device)

    def forward(self, H):
        '''
        :param H: network output
        :return: final output
        '''
        z = {}
        if self.finalize and not self.training:
            for col in self.cat:
                z[col] = torch.argmax(self.out_cat[str(col)](H[:, col, :]),dim=1).float()
        else:
            for col in self.cat:
                z[col] = self.out_cat[str(col)](H[:, col, :])
        for col in self.cont:
             z[col] = self.out_cont[str(col)](H[:, col, :]).squeeze()
        return z
        # else:
        #     z = {}
        #     for col in self.cat:
        #         z[col] = torch.argmax(self.out_cat[str(col)](H[:, col, :]),dim=1).float()
        #     for col in self.cont:
        #         z[col] = self.out_cont[str(col)](H[:, col, :]).squeeze()
        #     return torch.stack([x[1] for x in sorted(z.items())],dim=1)

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
    return new_X,new_M,real_label

