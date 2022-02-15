# Copyright (c) 2020 KU Leuven
import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F

from utils.util import print_heading, timestring, print_dbg, debug_off_m, debug_on, debug_off

non_linearities = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


##--------------------------------------------------------------------------------
## SparseLinear Layer
##--------------------------------------------------------------------------------
class SparseLinear(torch.nn.Module):
    """
    Linear layer with sparse input tensor, and dense output.
        in_features    size of input
        out_features   size of output
        bias           whether to add bias
    """
    def __init__(self, in_features, out_features, bias=True, verbose = False):
        super(SparseLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(in_features, out_features) / math.sqrt(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        print_dbg(f" SparseLinear - self.weight: {self.weight.type()}  {self.weight.shape}", verbose = verbose)
        print_dbg(f" SparseLinear - self.bias:   {self.bias.type()}    {self.bias.shape}", verbose = verbose)

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
## SparseChemBlock
##
class SparseChemBlock(nn.Module):
    expansion = 1

    def __init__(self, input_sz, output_sz, non_linearity, dropout, bias = True):
        super(SparseChemBlock, self).__init__()
        self.linear = nn.Linear(input_sz, output_sz, bias=bias )
        self.non_linear = non_linearity
        self.dropout = nn.Dropout(dropout) 
        # self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)

    def forward(self, x):
        out = self.linear(x)
        out = self.non_linear(out)
        # y = self.bn2(out)
        out = self.dropout(out)

        return out


##
## SparseLinear Classification Module
##
class SparseChem_Classification_Module(nn.Module):
    def __init__(self, inplanes, num_classes, rate=12):
        super(SparseChem_Classification_Module, self).__init__()
        self.linear  = nn.Linear(inplanes, num_classes)
        # Currently final layer of Sparse Chem doesn't have Relu and dropout. 
        # We can decide if we want them later
        # self.relu    = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.linear(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        return x



## 
##  SparseChem_Backbone: 
## 
class SparseChem_Backbone(torch.nn.Module):

    def __init__(self, conf, block, layers, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.version = 1.0

        print_heading(f" {self.name}  Ver: {self.version} Init() Start ", verbose = verbose)
        if hasattr(conf, "class_output_size"):
            self.class_output_size = conf['class_output_size']
            self.regr_output_size  = conf['regr_output_size']
        else:
            self.class_output_size = None
            self.regr_output_size  = None

        # if conf.input_size_freq is None:
        #     conf.input_size_freq = conf.input_size

        # self.net = nn.Sequential()
        self.layer_config = layers

        print_dbg(f" layer config : {self.layer_config} \n", verbose = verbose)
        ##----------------------------------------------------
        ## Input Net        
        ##----------------------------------------------------
        # assert conf.input_size_freq <= conf.input_size, \
        #         f"Number of high important features ({conf.input_size_freq}) should not be higher input size ({conf.input_size})."
        # self.input_splits = [conf.input_size_freq, conf.input_size - conf.input_size_freq]

        # self.input_net_freq = nn.Sequential(SparseLinear(self.input_splits[0], conf.hidden_sizes[0]))
        # print(f" input_size_freq  : {conf.input_size_freq}")
        # print(f" self.input_splits: {self.input_splits}")
        # self.input_layer = nn.Module()
        # self.net.add_module("InputNet",SparseLinear(conf['input_size'], conf['hidden_sizes'][0]))
        # self.net.add_module("InputNet_nonlinearity", non_linearities[conf['first_non_linearity']]() )
        # self.input_layer.add_module("Input_linear",SparseLinear(conf['input_size'], conf['hidden_sizes'][0]))
        # self.input_layer.add_module("Input_nonlinear", non_linearities[conf['first_non_linearity']]() )

        # self.Input_nonlinear =  non_linearities[conf['first_non_linearity']]() 
        # self.Input_nonlinear =  nn.ReLU(inplace=True)

        print_dbg(f" Input Layer  - Input: {conf['input_size']}  output: {conf['hidden_sizes'][0]}"
                  f"  non-linearity:{non_linearities[conf['first_non_linearity']]}", verbose = verbose)

        self.Input_linear = SparseLinear(conf['input_size'], conf['hidden_sizes'][0])
 

        ##----------------------------------------------------
        ## Middle Layers        
        ##----------------------------------------------------
        self.blocks = []

        for i in range(1, len(conf['hidden_sizes'])):
            print_dbg(f" Hidden layer {i} - Input: {conf['hidden_sizes'][i-1]}   output:{conf['hidden_sizes'][i]}", verbose = verbose)
            blk = self._make_layer(block = block,
                                   input_sz   = conf['hidden_sizes'][i-1], 
                                   output_sz  = conf['hidden_sizes'][i], 
                                   non_linearity = non_linearities[conf['middle_non_linearity']](),
                                   dropout    = conf['middle_dropout'])
                                           
            self.blocks.append(nn.ModuleList(blk))
    

        ##----------------------------------------------------
        ## Final Hidden layer  
        ##----------------------------------------------------
        i+= 1
        print_dbg(f" Hidden layer {i} : Input{conf['hidden_sizes'][i-1]}   output:{conf['tail_hidden_size']}", verbose = verbose)
        blk = self._make_layer(block = block,
                               input_sz   = conf['hidden_sizes'][i-1], 
                               output_sz  = conf['tail_hidden_size'], 
                               non_linearity = non_linearities[conf['last_non_linearity']](),
                               dropout    = conf['last_dropout'])        
        
        self.blocks.append(nn.ModuleList(blk))
        
        # self.net.add_module(f"layer_{i}_linear"   , nn.Linear(conf['hidden_sizes'][i-1], conf['tail_hidden_size'], bias=True) )
        # self.net.add_module(f"layer_{i}_nonlinear", non_linearity())
        # self.net.add_module(f"layer_{i}_dropout"  , nn.Dropout(conf['last_dropout']))

        ##----------------------------------------------------
        ## Individual head layer is defined in SparseChem_Classification_Layer 
        ##----------------------------------------------------
        #   self.add_module(f"layer_{i}_linear",        nn.Linear(conf.hidden_sizes[-1], conf.output_size))

        # print_heading(f" SparseChem Backbone -- Final configuration(1) : \n"
        #               f"                     self.blocks: {type(self.blocks)}  len:{len(self.blocks)}", verbose = verbose)
        # for i,blk in enumerate(self.blocks,1):
        #     print_heading(f"block # {i}  type:{type(blk)} ", verbose =verbose)
        #     print(f" {blk} \n")
        # print('\n')
            
        self.blocks = nn.ModuleList(self.blocks)

        if self.verbose :
            print_heading(f" SparseChem Backbone -- Final configuration(2) : \n" 
                          f"                     self.blocks: {type(self.blocks)}  len:{len(self.blocks)}", verbose = True)
            for i,blk in enumerate(self.blocks,1):
                print_heading(f"block # {i}  type:{type(blk)} len:{len(blk)}", verbose = True)
                print(f" {blk} \n")
            print('\n')
            print_heading(f" SparseChem backbone initialize weights ", verbose = True)

        self.apply(self.init_weights)
        # print_heading(f" Initialize Input_linear layer#  type:{type(self.Input_linear)} ", verbose =verbose)            
        # self.Input_linear.apply(self.init_weights)
        # for i, blk in enumerate(self.blocks,1):
            # print_heading(f" Initialize block # {i}  type:{type(blk)} ", verbose =verbose)            
            # blk.apply(self.init_weights)
        print_heading(f" {self.name} Init() End ", verbose = verbose)
        return 

    def init_weights(self, m, verbose = False):
        print_dbg(f"   SparseChem Backbone - init_weights - module {m} ", verbose = verbose)

        if type(m) == SparseLinear:
            print_dbg(f"    >>> apply xavier_uniform to SparseLinear module {m} \n", verbose = verbose)
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            m.bias.data.fill_(0.1)
        
        if type(m) == nn.Linear:
            print_dbg(f"    >>> apply xavier_uniform to linear module {m} \n", verbose = verbose)
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            if m.bias is not None:
                m.bias.data.fill_(0.1)


    def _make_layer(self, block, input_sz, output_sz, non_linearity, dropout, bias = True, verbose = False):
        '''
            Build a layer consisting of SparseChemBlock(s)
            Currently each layer only contains one block

            block :      Block type (SparseChemBlock)
            input_sz:    input_features
            output_sz:   output_features
            non_linear:  non linearity
            dropout:     droput rate
            bias 
        '''
        print_dbg(f"\t _make_layer() using block: {block}", verbose = verbose)
        layers = []
        layers.append(block(input_sz, output_sz, non_linearity, dropout, bias))

        return layers

    @debug_off
    def forward(self, x, policy=None, last_hidden=False, task_id = '', verbose = False):
        if self.verbose:
            print_heading(f" {timestring()} - SparseChem backbone forward start  for task {task_id} ", verbose = verbose)
            print_dbg(f"\t  Input : X shape: {x.shape}   last_hidden:{last_hidden}", verbose = verbose)
            print_dbg(f"  policy used for forward pass: \n {policy}", verbose = verbose)
        
        x = self.Input_linear(x)
        print_dbg(f"\t Output of Input Linear Layer: {x.shape}", verbose = verbose)
 

        if policy is None:
            # forward through the all blocks without dropping
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_
                    residual = x
                    x = F.relu(residual + self.blocks[segment][b](x))
                    print_dbg(f"\t Segment{segment} num_blocks: {num_blocks}   block {b} -  output: {x.shape}", verbose = verbose)

                    # x  =  self.blocks[segment][b](x)

        # do the block dropping (based on policy)
        else:
            t = 0

            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    print_dbg(f" segment: {segment}  num_block: {num_blocks}  t: {t}  b: {b} ", verbose = verbose)
                    print_dbg(f" policy[{t},0]: {policy[t,0]:5f}   policy[{t},1]: {policy[t,1]:5f} ", verbose = verbose)
                    
                    residual = x
                    block_x = self.blocks[segment][b](x)
                    fx = F.relu(residual + block_x)
                    
                    # Policy[t,0] : layer selected  
                    # Policy[t,1] : layer IS NOT selected.
        
                    x  = (fx * policy[t, 0] )+ (residual * policy[t, 1])
                    
                    print_dbg(f" residual: {residual.shape}  block_out: {block_x.shape}  ", verbose = verbose)
                    print_dbg(f" fx: {fx.shape}   ", verbose = verbose)
                    print_dbg(f" new x: {x.shape}   ", verbose = verbose)

                    # residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    # fx = F.relu(residual + self.blocks[segment][b](x))                    
                    #
                    # if policy.ndimension() == 1:
                        # x = fx * policy[t] + residual * (1-policy[t])
                    # elif policy.ndimension() == 2:
                        # x = fx * policy[t, 0] + residual * policy[t, 1]
                    # ndim() == 3 used for instance-based policy
                    # elif policy.ndimension() == 3:
                    #     x = fx * policy[:, t, 0].contiguous().view(-1, 1, 1, 1) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    
                    t += 1

        # if last_hidden:
            # H = self.net[:-1](X)
            # return self.net[-1].net[:-1](H)
        
        # out = self.net(X)
        
        # if self.class_output_size is not None:
            ## splitting to class and regression
            # return out[:, :self.class_output_size], out[:, self.class_output_size:]
        print_heading(f" {timestring()} - SparseChem backbone forward end", verbose = verbose)   
        return x



    @property
    def has_2heads(self):
        return self.class_output_size is not None
    
    @property
    def name(self):
        return 'SparseChem_Backbone'