import sys
sys.path.insert(0, '..')
from models.base import *
import torch
import torch.nn as nn
# import tqdm
# import time
import math
from models.util import sparse_split2

non_linearities = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}
##
## SparseLinear Layer
##
class SparseLinear(nn.Module):
    """
    Linear layer with sparse input tensor, and dense output.
        in_features    size of input
        out_features   size of output
        bias           whether to add bias
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) / math.sqrt(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        out = torch.mm(input, self.weight)
        if self.bias is not None:
            return out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.weight.shape[0], self.weight.shape[1], self.bias is not None
        )

##
## InputNet Segment 
##
class SparseInputNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        if conf.input_size_freq is None:
            conf.input_size_freq = conf.input_size
        
        assert conf.input_size_freq <= conf.input_size, f"Number of high important features ({conf.input_size_freq}) should not be higher input size ({conf.input_size})."
        self.input_splits = [conf.input_size_freq, conf.input_size - conf.input_size_freq]
        self.net_freq   = SparseLinear(self.input_splits[0], conf.hidden_sizes[0])
        
        print(f" input_size_freq: {conf.input_size_freq}    input_size: {conf.input_size}")
        print(f" self.input_splits: {self.input_splits}")
        # print(f" tail_hidden_size is {conf.tail_hidden_size}")
        
        if self.input_splits[1] == 0:
            self.net_rare = None
        else:
            ## TODO: try adding nn.ReLU() after SparseLinear
            ## Bias is not needed as net_freq provides it
            self.net_rare = nn.Sequential(
                SparseLinear(self.input_splits[1], conf.tail_hidden_size),
                nn.Linear(conf.tail_hidden_size, conf.hidden_sizes[0], bias=False),
            )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == SparseLinear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            m.bias.data.fill_(0.1)

        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def forward(self, X):
        if self.input_splits[1] == 0:
            return self.net_freq(X)
        Xfreq, Xrare = sparse_split2(X, self.input_splits[0], dim=1)
        return self.net_freq(Xfreq) + self.net_rare(Xrare)

##
## MiddleNet Segment 
##
class MiddleNet(nn.Module):
    """
    Additional hidden layers (hidden_sizes[1:]) are defined here 
    """
    def __init__(self, conf):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(conf.hidden_sizes) - 1):
            self.net.add_module(f"layer_{i}", nn.Sequential(
                non_linearities[conf.middle_non_linearity](),
                nn.Dropout(conf.middle_dropout),
                nn.Linear(conf.hidden_sizes[i], conf.hidden_sizes[i+1], bias=True),
            ))
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def forward(self, H):
        return self.net(H)

##
## LastNet Segment 
##
class LastNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.non_linearity = conf.last_non_linearity
        non_linearity = non_linearities[conf.last_non_linearity]
        self.net = nn.Sequential(
            non_linearity(),
            nn.Dropout(conf.last_dropout),
            nn.Linear(conf.hidden_sizes[-1], conf.output_size),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("sigmoid"))
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def forward(self, H):
        return self.net(H)

##
##  SparseFFN : Complete network
##  
class SparseFFN(nn.Module):
    def __init__(self, conf):
        super().__init__()
        print(f" defining SparseFFN")
        if hasattr(conf, "class_output_size"):
            self.class_output_size = conf.class_output_size
            self.regr_output_size  = conf.regr_output_size
        else:
            self.class_output_size = None
            self.regr_output_size  = None

        self.net = nn.Sequential(
            SparseInputNet(conf),
            MiddleNet(conf),
            LastNet(conf),
        )

    @property
    def has_2heads(self):
        return self.class_output_size is not None

    def forward(self, X, last_hidden=False):
        if last_hidden:
            H = self.net[:-1](X)                ## pass through all except the last layer of SparseFFN (LastNet)
            return self.net[-1].net[:-1](H)     ## pass through all except the last layer of LastNet
        out = self.net(X)
        ## single output 
        if self.class_output_size is None:
            return out
        ## splitting to class and regression outputs
        return out[:, :self.class_output_size], out[:, self.class_output_size:]

