import time
from itertools import cycle
import torch.nn as nn
import torch
from mhsa import MHSA
from util import Flatten,Reshape,Input_Embbeding,Output_Encoding
from Image_patcher import LinearImagePatcher


class NPT(nn.Module):
    '''
    non parametric transformer
    '''
    def __init__(self,encoded_data, layers, heads, rff_layers, device,drop_out = None):
        super(NPT, self).__init__()
        self.image_patcher = None
        self.network = nn.Sequential()
        self.ln = nn.LayerNorm(encoded_data.data.embedding_dim)
        reshape = Reshape(encoded_data.data.embedding_dim)
        flatten = Flatten()
        for layer in range(layers):
            self.network.add_module(f"flatten_{layer+1}",flatten)
            self.network.add_module(f'abd_{layer+1}',MHSA(encoded_data.data.input_dim, rff_layers, heads,drop_out, device))
            self.network.add_module(f"reshape_{layer+1}",reshape)
            self.network.add_module(f'aba_{layer+1}',MHSA(encoded_data.data.embedding_dim, rff_layers, heads,drop_out, device))
        if drop_out is None:
            self.drop_out = None
        else:
            self.drop_out = nn.Dropout(p = drop_out)
        self.device = device
        self.to(device)

    def define_in_and_out_encoding(self,encoded_data,device,params):
        self.cat_index = torch.tensor(0).to(device)
        self.cont_index = torch.tensor(1).to(device)
        self.type_embedding = nn.Embedding(2, encoded_data.data.embedding_dim)
        if params['use_image_patcher']:
            self.categorical = {params['image_n_patches']:encoded_data.data.n_classes}
            self.index_embedding = nn.Embedding(params['image_n_patches']+1,
                                                encoded_data.data.embedding_dim)
            self.image_patcher = LinearImagePatcher(encoded_data,params['image_n_patches'],device)
        else:
            self.categorical = encoded_data.data.categorical
            self.index_embedding = nn.Embedding(len(encoded_data.data.categorical.keys()) + len(encoded_data.data.continuous),
                                                encoded_data.data.embedding_dim)
            self.input_embedding = Input_Embbeding(encoded_data,device)
            self.output_embedding = Output_Encoding(encoded_data, device)
        self.to(self.device)


    def forward(self, X, M):
        if self.image_patcher is not None:
            embedded_input = self.image_patcher.encode(torch.stack((X,M),dim=2))
        else:
            embedded_input = self.input_embedding(X,M)
        for i,info in embedded_input.items():
            feature_index_embeddings = torch.unsqueeze(
                self.index_embedding(torch.tensor(i).to(self.device)), 0)

            feature_index_embeddings = feature_index_embeddings.repeat(
                X.size(0), 1)
            info += feature_index_embeddings
            if i in self.categorical.keys():
                feature_type_embeddings = torch.unsqueeze(
                    self.type_embedding(self.cat_index), 0)
            else:
                feature_type_embeddings = torch.unsqueeze(
                    self.type_embedding(self.cont_index), 0)
            feature_type_embeddings = feature_type_embeddings.repeat(
                X.size(0), 1)
            info += feature_type_embeddings
        embedded_input = sorted([(key, item) for key, item in embedded_input.items()])
        embedded_input = [item[1] for item in embedded_input]
        embedded_input = torch.stack(embedded_input, dim=1)
        if self.ln is not None:
            embedded_input = self.ln(embedded_input)
        if self.drop_out is not None:
            embedded_input = self.drop_out(embedded_input)
        network_output = self.network.forward(embedded_input)
        if self.image_patcher is not None:
            output = self.image_patcher.decode(network_output)
        else:
            output = self.output_embedding(network_output)
        if self.drop_out is not None:
              output = {key:self.drop_out(value)for key,value in output.items()}
        return output



