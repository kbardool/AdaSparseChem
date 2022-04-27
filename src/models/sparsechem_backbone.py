# Copyright (c) 2020 KU Leuven
from collections import OrderedDict
import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F

from utils.util import print_heading, print_underline,timestring, print_dbg, debug_off_m, debug_on, debug_off

non_linearities = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


##--------------------------------------------------------------------------------
## SparseLinear Layer
##--------------------------------------------------------------------------------
class SparseLinear(nn.Module):
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
            out = out + self.bias

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

    def __init__(self, input_sz, output_sz, non_linearity, dropout, bias = True, verbose = False):
        super(SparseChemBlock, self).__init__()
        print_dbg(f"           SparseChemBlock.init(): input_size: {input_sz} output_sz: {output_sz}  "
                  f" non_linearity: {non_linearity} dropout: {dropout} bias: {bias}", verbose = verbose)        
        self.linear = nn.Linear(input_sz, output_sz, bias=bias )
        self.non_linear = non_linearity
        self.dropout = nn.Dropout(dropout) 
        # self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)

    def forward(self, x):
        out = self.linear(x)
        out = self.non_linear(out)
        out = self.dropout(out)
        # out = self.bn2(out) 

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

    def __init__(self, conf, block, layers, verbose=True):
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

        self.layer_config = layers
        self.skip_residual_layers  = conf['skip_residual']
        self.skip_hidden  = conf['skip_hidden']
        self.blocks = []
        self.residuals= []
        print_dbg(f" layer config   : {self.layer_config} \n", verbose = verbose)
        print_dbg(f" residual layers: {self.skip_residual_layers} \n", verbose = verbose)

        ##----------------------------------------------------
        ## Input Net        
        ##----------------------------------------------------
        # OrderedDict allows naming of layers
        self.Input_Layer  = nn.Sequential(OrderedDict([
                                          ('linear'    , SparseLinear(in_features   = conf['input_size'], 
                                                                      out_features  = conf['hidden_sizes'][0]) ),
                                          ('non_linear', non_linearities[conf['first_non_linearity']]()),
                                          ('dropout'   , nn.Dropout(conf['first_dropout']))
                                          ]))
  
        print_dbg(f" SparseChem_BackBone() Input Layer  - Input: {conf['input_size']}  output: {conf['hidden_sizes'][0]}"
                  f"  non-linearity:{non_linearities[conf['first_non_linearity']]}", verbose = True)

        if self.skip_hidden:
            pass
        else:
            ##----------------------------------------------------
            ## HIDDEN LAYERS
            ##----------------------------------------------------
            i = 0
            for i in range(len(conf['hidden_sizes']) -1 ):
                print_dbg(f" Hidden layer {i} - Input: {conf['hidden_sizes'][i]}   output:{conf['hidden_sizes'][i+1]}", verbose = verbose)
                blk, res = self._make_layer(block = block,
                                    input_sz   = conf['hidden_sizes'][i], 
                                    output_sz  = conf['hidden_sizes'][i+1], 
                                    non_linearity = non_linearities[conf['middle_non_linearity']](),
                                    dropout    = conf['middle_dropout'], verbose = verbose)
                                            
                # self.blocks.append(nn.ModuleList(blk))
                self.blocks.append(blk)
                self.residuals.append(res)

            ##----------------------------------------------------
            ## Final Hidden layer  
            ##----------------------------------------------------
            # i+= 1
            print_dbg(f" Final Hidden layer {i} : Input size: {conf['hidden_sizes'][-1]}   output size:{conf['tail_hidden_size'][0]}", verbose = verbose)
            blk, res = self._make_layer(block         = block,
                                        input_sz      = conf['hidden_sizes'][-1], 
                                        output_sz     = conf['tail_hidden_size'][0], 
                                        non_linearity = non_linearities[conf['last_non_linearity']](),
                                        dropout       = conf['last_dropout'], verbose = verbose)        
            self.blocks.append(blk)
            self.residuals.append(res)
        # endif 


        ## Create Module List 
        print_dbg(f" Module List ", verbose = verbose)
        self.blocks    = nn.ModuleList(self.blocks)
        self.residuals = nn.ModuleList(self.residuals)

        ##----------------------------------------------------------------------
        ## Individual task heads are defined in SparseChem_Classification_Layer 
        ##----------------------------------------------------------------------

        ##----------------------------------------------------------------------
        ## Initialize weights and display configuration 
        ##----------------------------------------------------------------------
        print_heading(f" Initialize weights ", verbose = True)
        self.apply(self.init_weights)

        if verbose :
            print_heading(f" SparseChem Backbone -- Final configuration(2) : \n" 
                          f"                     self.blocks: {type(self.blocks)}  len:{len(self.blocks)}", verbose = True)
            print_underline(f" Input_Layer  type:{type(self.Input_Layer)}  ", verbose =True)  
            print(f"self.Input_layer")          
            
            print_heading(f"Layers/Blocks    : {type(self.blocks)}   len:{len(self.blocks)} \n"
                          f"Resdiual layers  : {type(self.residuals)}   len:{len(self.residuals)}", verbose=verbose)

            for i, (blk, res) in enumerate( zip(self.blocks, self.residuals) ,1):
                print_heading(f" Layer #: {i}  type:{type(blk)} ", verbose = True)
                print(f" {blk} \n")
                res_len = 'n/a' if res is None else len(res)
                print(f" Residual Layer #: {i}  type:{type(res)} ")
                print(f" {res}")
            print('\n\n')
            print(f" {self.name} Init() End " )
        return 

    def init_weights(self, m, verbose = False):
        print_dbg(f"   SparseChem Backbone - init_weights - module {m} ", verbose = verbose)

        if type(m) == SparseLinear:
            print_dbg(f"    >>> apply xavier_uniform to SparseLinear module {m} \n", verbose = verbose)
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            m.bias.data.fill_(0.1)
        
        if type(m) == nn.Linear:
            print_dbg(f"    >>> apply xavier_uniform to linear module {m} \n", verbose = verbose)
            # torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("sigmoid"))
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
        print_dbg(f"    _make_layer() using block: {block}", verbose = verbose)
        print_dbg(f"           input_size: {input_sz} output_sz: {output_sz}  non_linearity: {non_linearity} dropout: {dropout} bias: {bias}", verbose = verbose)
        residual = None

        # layers.append(block(input_sz, output_sz, non_linearity, dropout, bias, verbose = verbose))
        layers = block(input_sz, output_sz, non_linearity, dropout, bias, verbose = verbose)

        if (not self.skip_residual_layers) and (input_sz != output_sz):
            residual = nn.Sequential(block(input_sz, output_sz, non_linearity, dropout, bias, verbose = verbose))

        return layers, residual

    def forward(self, x, policy=None, last_hidden=False,  task_id = '', verbose = False):
        """Sparsechem Backbone forward pass"""
        # if self.verbose:
        #     print_heading(f" {timestring()} - SparseChem backbone forward start  for task {task_id} ", verbose = verbose)
        #     print_dbg(f"\t  Input : X shape: {x.shape}   last_hidden:{last_hidden}", verbose = verbose)
        #     print_dbg(f"  policy used for forward pass: \n {policy}", verbose = verbose)
        
        x = self.Input_Layer(x)
        # print_dbg(f"\t Output of Input Linear Layer: {x.shape}", verbose = verbose)

        if policy is None:
            # forward through the all blocks without dropping
            for layer, _ in enumerate(self.layer_config):
                # apply the residual skip out of _make_layers_
 
                if self.skip_hidden:
                    print_dbg('skip_hidden is true - return x', verbose = False)
                    pass
                elif self.skip_residual_layers:
                    x = self.blocks[layer](x)
                else:
                    residual = x  if self.residuals[layer] is None else self.residuals[layer](x)
                    x = F.relu(residual + self.blocks[layer](x))
                    
                # x = residual + self.blocks[layer](x)
                # x  =  self.blocks[segment][b](x)
                # print_dbg(f"\t Segment{segment} num_blocks: {num_blocks}   block {b} -  output: {x.shape}", verbose = verbose)


            # for layer, num_blocks in enumerate(self.layer_config):
            #     for b in range(num_blocks):
            #         # apply the residual skip out of _make_layers_
            #         # residual = self.residuals[layer](x) if b == 0 and self.residuals[layer] is not None else x
            #         # x = F.relu(residual + self.blocks[layer][b](x))
            #         # x = residual + self.blocks[layer][b](x)
            #         x = self.blocks[layer][b](x)

            #         # print_dbg(f"\t Segment{segment} num_blocks: {num_blocks}   block {b} -  output: {x.shape}", verbose = verbose)
            #         # x  =  self.blocks[segment][b](x)

        # else:
        #     # t = 0
        #     ##----------------------------------------------------
        #     ## Apply policies to each layer  
        #     ##----------------------------------------------------
        #     for layer, num_blocks in enumerate(self.layer_config):
        #         for b in range(num_blocks):
        #             # print_dbg(f" segment: {segment}  num_block: {num_blocks}  t: {t}  b: {b} ", verbose = verbose)
        #             # print_dbg(f" policy[{t},0]: {policy[t,0]:5f}   policy[{t},1]: {policy[t,1]:5f} ", verbose = verbose)
                    
        #             residual = x
        #             block_x = self.blocks[layer][b](x)
        #             fx = F.relu(residual + block_x)
                    
        #             # Policy[t,0] : layer selected  
        #             # Policy[t,1] : layer IS NOT selected.
        
        #             x  = (fx * policy[layer, 0] )+ (residual * policy[layer, 1])
                    
                    # print_dbg(f" residual: {residual.shape}  block_out: {block_x.shape}  ", verbose = verbose)
                    # print_dbg(f" fx: {fx.shape}   ", verbose = verbose)
                    # print_dbg(f" new x: {x.shape}   ", verbose = verbose)

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
                    
                    # t += 1

        # if last_hidden:
            # H = self.net[:-1](X)
            # return self.net[-1].net[:-1](H)
        
        # print_heading(f" {timestring()} - SparseChem backbone forward end", verbose = verbose)   
        return x



    @property
    def has_2heads(self):
        return self.class_output_size is not None
    
    @property
    def name(self):
        return 'SparseChem_Backbone'