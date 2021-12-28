# import sys
# sys.path.insert(0, '..')
from models.base import *
import torch.nn.functional as F
# from scipy.special import softmax
# from models.util import count_params, compute_flops
# import torch
# import tqdm
# import time
import math

def get_shape(shape1, shape2):
    out_shape = []
    for d1, d2 in zip(shape1, shape2):
        out_shape.append(min(d1, d2))
    return out_shape


def resnet_dev(num_class=10, blocks=BasicBlock):
    return ResNet_Dev(blocks, [1, 1, 1], num_class)    

class ResNet_Dev(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(ResNet_Dev, self).__init__()

        factor = 1
        self.in_planes = int(32 * factor)
        self.conv1 = conv3x3(3, int(32 * factor))
        self.bn1 = nn.BatchNorm2d(int(32 * factor))
        self.relu = nn.ReLU(inplace=True)

        strides = [2, 2, 2]
        filt_sizes = [64, 128, 256]
        self.blocks, self.ds = [], []
        self.parallel_blocks, self.parallel_ds = [], []

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.parallel_blocks.append(nn.ModuleList(blocks))
            self.parallel_ds.append(ds)
        self.parallel_blocks = nn.ModuleList(self.parallel_blocks)
        self.parallel_ds = nn.ModuleList(self.parallel_ds)

        self.bn2 = nn.Sequential(nn.BatchNorm2d(int(256 * factor)), nn.ReLU(True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(256 * factor), num_class)

        self.layer_config = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.bn1(self.conv1(x))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return layers, downsample

    def forward(self, x):
        t = 0
        x = self.seed(x)
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                output = self.blocks[segment][b](x)
                res_shape = residual.shape
                out_shape = output.shape
                new_shape = get_shape(res_shape, out_shape)
                x = F.relu(residual[:new_shape[0], :new_shape[1], :new_shape[2], :new_shape[3]] +
                           output[:new_shape[0], :new_shape[1], :new_shape[2], :new_shape[3]])
                t += 1

        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
