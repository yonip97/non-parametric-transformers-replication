import time
from itertools import cycle
import torch.nn as nn
import torch
from mhsa import MHSA
from util import Input_Embbeding, Output_Encoding,Flatten,Reshape



class NPT(nn.Module):
    '''
    non parametric transformer
    '''
    def __init__(self, categorical, continuous, embedding_dim, input_dim, layers, heads, rff_layers,h, device,drop_out = None,finalize = False):
        super(NPT, self).__init__()
        self.input_embedding = Input_Embbeding(categorical, continuous, embedding_dim, device)
        self.output_embedding = Output_Encoding(categorical, continuous, embedding_dim,finalize, device)
        self.network = nn.Sequential()
        reshape = Reshape(embedding_dim)
        flatten = Flatten()
        for layer in range(layers):
            self.network.add_module(f"flatten_{layer+1}",flatten)
            self.network.add_module(f'abd_{layer+1}',MHSA(input_dim, rff_layers, heads,h,drop_out, device))
            self.network.add_module(f"reshape_{layer+1}",reshape)
            self.network.add_module(f'aba_{layer+1}',MHSA(embedding_dim, rff_layers, heads,h,drop_out, device))
        if drop_out == None:
            self.drop_out = drop_out
        else:
            self.drop_out = nn.Dropout(p = drop_out)
        self.device = device
        self.finalize = finalize

    def forward(self, X, M):
        if self.drop_out is not None:
            embedded_input = self.drop_out(self.input_embedding(X,M))
        else:
            embedded_input = self.input_embedding(X,M)
        network_output = self.network.forward(embedded_input)
        if self.drop_out is not None:
              output = {key:self.drop_out(value)for key,value in self.output_embedding(network_output).items()}
        else:
              output = self.output_embedding(network_output)
        if self.finalize and not self.training:
            return torch.stack([x[1] for x in sorted(output.items())],dim=1)
        else:
            return output
