import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from dev.deeplab_resnet_dev import Deeplab_ResNet_Backbone_Dev
from dev.resnet_dev import resnet_dev
from models.base import Classification_Module
from scipy.special import softmax
# from models.deeplab_resnet import * 

class MTL_Instance_Dev(nn.Module):
    """

    """
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0):
        super(MTL_Instance_Dev, self).__init__()
    
        self.backbone = Deeplab_ResNet_Backbone_Dev(block, layers)
        self.num_tasks = len(num_classes_tasks)

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'policynet' in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'policynet' in name:
                params.append(param)
        return params

    def get_logits(self, img):
        batch_size = img.shape[0]
        self.task_logits = self.policynet(img).contiguous().view(batch_size, -1, 2)

    def train_sample_policy(self, img, temperature, hard_sampling):
        print(f' MTL_Instance TRAIN SAMPLE POLICY')
        self.get_logits(img)
        task_logits_shape = self.task_logits.shape
        policy = F.gumbel_softmax(self.task_logits.contiguous().view(-1, 2), temperature, hard=hard_sampling)
        policy = policy.contiguous().view(task_logits_shape)
        policys = list(torch.split(policy,  sum(self.layers) - self.skip_layer, dim=1))
        return policys

    def test_sample_policy(self, img,  hard_sampling):
        print(f' MTL_Instance TEST SAMPLE POLICY')
        self.policys = []

        if  hard_sampling:
            raise ValueError('hard sample is not supported')
        else:
            self.get_logits(img)
            task_logits_shape = self.task_logits.shape
            cuda_device = self.task_logits.get_device()
            logits = self.task_logits.contiguous().view(-1, 2)
            # logits = self.task_logits.detach().cpu().numpy().reshape(-1, 2)
            distribution = F.softmax(logits, dim=-1)
            s1 = distribution.shape[0]
            esl = torch.rand(s1).float().to(cuda_device)
            sampled = torch.where(esl < distribution[:, 0], torch.ones(s1, device=cuda_device),
                              torch.zeros(s1, device=cuda_device) )
            policy = [sampled, 1 - sampled]
            policy = torch.stack(policy, dim=-1).contiguous().view(task_logits_shape)
            policys = list(torch.split(policy, sum(self.layers) - self.skip_layer, dim=1))
        

        return policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        self.policynet = resnet_dev(num_class=2 * self.num_tasks * (num_layers - self.skip_layer))
        self._arch_parameters = self.policynet.parameters()

    def forward(self, img, temperature, is_policy, policys=None, num_train_layers=None, hard_sampling=False, mode='train'):
        # print('deeplab_resnet.forward() num_train_layers in mtl forward = ', num_train_layers)

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        
        # Generate features
        cuda_device = img.get_device()
        if is_policy:
            if mode == 'train':
                self.policys = self.train_sample_policy(img, temperature, hard_sampling)
            elif mode == 'eval':
                self.policys = self.test_sample_policy(img, hard_sampling)
            elif mode == 'fix_policy':
                assert(policys is not None)
                self.policys = policys
                # import pdb
                # pdb.set_trace()
                for p in policys:
                    print(f"   fix_policy - {p.shape}")
            else:
                raise NotImplementedError('mode %s is not implemented' % mode)

            ## build policy array to pass to backbone

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            print(f'         MTL2_Dev forward  - skip layer (for policy padding): {sum(self.layers)- num_train_layers}')

            batch_size = img.shape[0]
            if cuda_device != -1:
                padding = torch.ones(batch_size, skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(batch_size, skip_layer, 2)

            padding[:, :, 1] = 0
            padding_policys = []
            feats = []

            ## padding policy is the concatenation of [1, 0] for the layers that we have not policy trainined +
            ## gumbel_softmax dist for the layers that have been policy trained (indicated by num_train_layers)
            
            for t_id in range(self.num_tasks):
                padding_policy = torch.cat((padding.float(), self.policys[t_id][:, -num_train_layers:].float()), dim=1).contiguous()
                print(f"      task id: {t_id}  padding_policy shape: {padding_policy.shape}   vals: {padding_policy[:,-num_train_layers:]}")
                padding_policys.append(padding_policy)

                ## pass input and policy throiugh the backbone
                feats.append(self.backbone(img, padding_policy))

            if mode == 'fix_policy':
                logits = [None] * self.num_tasks
            else:
                logits = list(torch.split(self.task_logits, sum(self.layers) - self.skip_layer, dim=1))
                
            self.policys = padding_policys

        ## if is_policy is false - simply pass img through backbones
        ## for each task, a feature set is generated.
        else:
            feats = [self.backbone(img)] * self.num_tasks
            logits = [None] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats[t_id])
            outputs.append(output)


        return outputs, self.policys, logits
