import torch
from torch import nn
import math

class MHSA_layer(nn.Module):
    '''
    multi head self attention layer.
    '''
    def __init__(self, input_dim, output_dim,h,drop_out=None):
        super(MHSA_layer, self).__init__()
        self.key = nn.Linear(input_dim, output_dim)
        self.query = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)
        if drop_out == None:
            self.drop_out = None
        else:
            self.drop_out = nn.Dropout(p=drop_out)
        self.h = h

    def forward(self, X):
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        if self.drop_out is not None and self.training:
            #return self.drop_out(self.softmax(torch.einsum('ijl,ikl->ijk', Q, K) / Q.shape[-1]**0.5).bmm(V))
            return self.drop_out(self.softmax(Q @ torch.transpose(K, 1, 2) / self.h**0.5) @ V)
        else:
            #return self.softmax(torch.einsum('ijl,ikl->ijk', Q, K) / Q.shape[-1] ** 0.5).bmm(V)
            return self.softmax(Q @ torch.transpose(K, 1, 2) / self.h**0.5) @ V
#

# class MHSelfAtt(nn.Module):
#     '''
#     multi head self attention.
#     composed of a number of mhsa (multi head self attention blocks) and mix head layer
#     '''
#     def __init__(self, input_dim, head_num,h,drop_out, device):
#         super(MHSelfAtt, self).__init__()
#         self.input_dim = input_dim
#         self.split_dim = input_dim // head_num
#         self.device = device
#         self.head_num = head_num
#         self.blocks = nn.ModuleList(
#              MHSA_layer(self.input_dim, int(self.input_dim / self.head_num),h,drop_out) for i in range(head_num))
#         self.mix_heads = nn.Linear(input_dim, input_dim)
#         if drop_out is not None:
#             self.drop_out = nn.Dropout(p = drop_out)
#         else:
#             self.drop_out = None
#         self.to(self.device)
#
#
#     def forward(self, X):
#         if self.drop_out is not None:
#             return self.drop_out(self.mix_heads(torch.cat([block.forward(X) for block in self.blocks], dim=2)))
#         else:
#             return self.mix_heads(torch.cat([block.forward(X) for block in self.blocks], dim=2))


class Gelu(nn.Module):
    '''
    gelu function. returns x∗Φ(x) where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.
    '''

    def __init__(self):
        super(Gelu, self).__init__()
        self.gaussian = torch.distributions.normal.Normal(0, 1)

    def forward(self, input):
        return self.gaussian.cdf(input) * input


class rff(nn.Module):
    '''
    row wise feed forward network
    '''

    def __init__(self, input_dim, hidden_layers):
        super(rff, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module(f'linear_layer_0', module=nn.Linear(input_dim, 4 * input_dim))
        self.model.add_module(f'gelu_layer_0', module=Gelu())
        for i in range(hidden_layers):
            self.model.add_module(f'linear_layer_{i + 1}', module=nn.Linear(4 * input_dim, 4 * input_dim))
            self.model.add_module(f'gelu_layer_{i + 1}', module=Gelu())
        self.model.add_module('final_layer', module=nn.Linear(input_dim * 4, input_dim))

    def forward(self, input):
        return self.model.forward(input)


class MHSA(nn.Module):
    def __init__(self, input_dim, rff_layers, head_num, h, drop_out, device):
        '''
        :param input_dim: network input dim
        :param rff_layers: number of layers in the row wise feed forward network
        :param head_num: number of attention heads
        :param drop_out: dropout parameter
        '''
        super(MHSA, self).__init__()
        self.keys = nn.Linear(input_dim, input_dim)
        self.queries = nn.Linear(input_dim, input_dim)
        self.values = nn.Linear(input_dim, input_dim)

        #self.blocks = nn.ModuleList(MHSA_layer(input_dim,input_dim//head_num,h,drop_out)for i in range(head_num))
        self.rff = rff(input_dim, rff_layers)
        self.ln_0 = nn.LayerNorm(input_dim)
        self.ln_1 = nn.LayerNorm(input_dim)
        self.res = nn.Linear(input_dim, input_dim)
        self.mix_heads = nn.Linear(input_dim, input_dim)
        self.device = device
        self.softmax = nn.Softmax(dim=2)
        self.split_dim = input_dim // head_num
        if drop_out == None:
            self.drop_out = None
        else:
            self.drop_out = nn.Dropout(p=drop_out)
        self.h = input_dim
        self.to(self.device)




    def forward(self, input):
        # layer normalization
        X_multihead = self.ln_0.forward(input)
        # residual branch
        X_res = self.res.forward(input)
        # keys, queries and values of the heads
        K = torch.cat(torch.split(self.keys.forward(input), self.split_dim, 2), 0)
        Q = torch.cat(torch.split(self.queries.forward(X_multihead), self.split_dim, 2), 0)
        V = torch.cat(torch.split(self.values.forward(input), self.split_dim, 2), 0)
        A = self.softmax.forward(torch.einsum('ijl,ikl->ijk', Q, K) / (self.h ** 0.5))
        if self.drop_out != None:
            A = self.drop_out(A)
        multihead = A.bmm(V)
        #multihead = torch.cat([block.forward(input) for block in self.blocks])
        multihead = torch.cat(multihead.split(input.size(0), 0), 2)
        # mix heads
        H = self.mix_heads(multihead)
        if self.drop_out != None:
            H = self.drop_out(H)
        H = H + X_res
        # layer normalization
        H_rff = self.ln_1.forward(H)
        return H + self.rff.forward(H_rff)
        # output = self.res(input) + self.MHSelfAtt(self.ln(v))
        # return output + self.rff(self.ln(output))
        # #return self.just_residuals(input).add(self.rff(self.ln(self.pre_rff(input))))

# class MHSelfAtt_test(nn.Module):
#     '''
#     multi head self attention.
#     composed of a number of mhsa (multi head self attention blocks) and mix head layer
#     '''
#     def __init__(self, input_dim, head_num,drop_out, device):
#         super(MHSelfAtt_test, self).__init__()
#         self.input_dim = input_dim
#         self.split_dim = input_dim // head_num
#         self.device = device
#         self.head_num = head_num
#         self.mix_heads = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.query = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.softmax = nn.Softmax(dim=2)
#         if drop_out is not None:
#             self.drop_out = nn.Dropout(p = drop_out)
#         else:
#             self.drop_out = None
#         self.to(self.device)
#
