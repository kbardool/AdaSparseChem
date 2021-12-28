import torch
from torch import nn
import torch.nn.functional as F
from utils.util import print_heading

affine_par = True

class Deeplab_ResNet_Backbone_Dev(nn.Module):

    def __init__(self, block, layers):
        '''
        block: identifies type of block used in the ResNet construction (BasicBlock / Bottleneck)
        layers: List - number of blocks used in each section of the Backbone, eg. [3, 4, 23, 3]
        '''
        super(Deeplab_ResNet_Backbone_Dev, self).__init__()
        
        self.blocks   = []
        self.ds       = []
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change



        # layers passed are  [3, 6, 4, 3]
        filt_sizes = [64, 128, 256, 512]
        strides    = [1, 2, 1, 1]
        dilations  = [1, 1, 2, 4]

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            print_heading(f" Deeplab_ResNet_Backbone_Dev : making layer {idx}  filter size: {filt_size}  num-blocks in layer:{num_blocks}  strides: {stride} dilation: {dilation}")
            
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            
            # print(f" blocks: {type(blocks)} \n {blocks}")
            # print(f" ds    : {type(ds)} \n {ds}")
            # print('\n\n')
            
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

            # print(f" blocks: {type(self.blocks)}   len:{len(self.blocks)} ")
            # print(f" {type(self.blocks[-1])} \n {self.blocks[-1]}")
            # print(f"\n downsamples : {type(self.ds)}    len:{len(self.ds)} ")
            # print(f"| {type(self.ds[-1])}     \n {self.ds[-1]}")
            # print('\n\n')

        print_heading(f" Deeplab Resent Backbone -- Final configuration :")
        print_heading(f" self.blocks: {type(self.blocks)}  len:{len(self.blocks)}")
        for i,blk in enumerate(self.blocks,1):
            print_heading(f"block # {i} ")
            print(f" {type(blk)}:")
            print(f"{blk}")


        print_heading(f"self.downsamples    : {type(self.ds)}   len:{len(self.ds)}")
        for i in self.ds:
            print(f" {type(i)}:  {i} \n")
        print('\n\n')
            
        self.blocks = nn.ModuleList(self.blocks)
        self.ds     = nn.ModuleList(self.ds)
        self.layer_config = layers

        print(f" layer config : {self.layer_config}")

        print_heading(f"self.blocks ")
        print(f" {self.blocks} \n\n")

        print_heading(f"self.ds (downsampling ")
        print(f" {self.ds} \n\n")


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        '''
            Build a residual layer
            block :  Resnet Block type (BasicBlock, BottleNeck, BasicBlock2, Bottleneck2)
            planes:  Number of channels
            block.expansion is 1 
        '''
        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion) or (dilation == 2) or (dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine = affine_par))
            # for i in downsample._modules['1'].parameters():
            #     i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample



    def forward(self, x, policy=None):

        if policy is None:
            # forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # do the block dropping (based on policy)
            x = self.seed(x)
            t = 0

            # layer_config is  [3,6,4,3]
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    fx = F.relu(residual + self.blocks[segment][b](x))

                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(-1, 1, 1, 1) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1-policy[t])
                    
                    t += 1
        return x

    

