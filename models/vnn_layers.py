import torch
from torch import nn
import math


def filtering(X, H, C, bias=None):
    """
    Filtering operation with a graph convolutional filter with 
    shift operator C.
    """

    F_out, K, F_in = H.shape
    N, T, _ = X.shape

    Z = [X.clone()]

    for k in range(K-1):
        X = torch.matmul(C, X).reshape((N,T,F_in))
        Z.append(X.clone())

    Z = torch.stack(Z) # (K+1) x N x T x F_in

    U = H.reshape(F_out, (K)*F_in) @ Z.permute((0,3,1,2)).reshape(((K)*F_in,N * T))
    U = U.reshape((F_out, N, T)).permute((1,2,0)) # N x T x F_out

    if bias is not None:
        U = U + bias

    return U


class CovFilter(nn.Module):
    """
    Class for Covariance/Precision filter.
    """

    def __init__(self, in_feat, out_feat, K, bias = True):
        """
        Input:
            in_feat: number of input features

            out_feat: number of output features

            K: order of filter

            bias (bool): whether to use bias or not
        """

        super().__init__()
        self.K = K
        self.H = nn.parameter.Parameter(torch.Tensor(out_feat, K + 1, in_feat))

        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_feat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.K)
        self.H.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, X, C):

        U = filtering(X, self.H, C, bias=self.bias)
        return U

