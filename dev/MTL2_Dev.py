import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from dev.deeplab_resnet_dev import Deeplab_ResNet_Backbone_Dev
from utils.util import timestring, print_heading
from models.base import Classification_Module
from scipy.special import softmax
# from models.deeplab_resnet import * 

class MTL2_Dev(nn.Module):
    """
    Create the architecture based on the Deep lab ResNet backbone

    block:  block type used in ResNet (BasicBlock, Bottleneck)
    layers: List - number of blocks used in each section of the Backbone, eg. [3, 4, 23, 3]
    """
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0):
        super(MTL2_Dev, self).__init__()
        self.num_tasks = len(num_classes_tasks)

        print( '\n block            : ', block,
               '\n layers           : ', layers,
               '\n num_classes_tasks: ', num_classes_tasks, 
               '\n init_method      : ', init_method, 
               '\n init_neg_logits  : ', init_neg_logits,
               '\n skip_layer       : ', skip_layer)        
        
        ## Build Network Backbone
        self.backbone = Deeplab_ResNet_Backbone_Dev(block, layers)
        

        ## Build Task specific heads. Each task head has 4 sub-heads, each with a different dilation rate. 
        ## the sub-heads are summed on the forward pass. 
        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

        ## Display Task specific heads info
        print('-'*120)
        print(f" Task Specific Heads :")
        print('-'*120)
        for t_id, num_class in enumerate(num_classes_tasks):
            print(f" Task {t_id+1} :")
            print('-'*120, '\n')
            print(f"task{t_id+1}_fc1_c0:  {getattr(self,'task%d_fc1_c0' % (t_id + 1))}"  )
            print(f"task{t_id+1}_fc1_c1:  {getattr(self,'task%d_fc1_c1' % (t_id + 1))}"  )
            print(f"task{t_id+1}_fc1_c2:  {getattr(self,'task%d_fc1_c2' % (t_id + 1))}"  )
            print(f"task{t_id+1}_fc1_c3:  {getattr(self,'task%d_fc1_c3' % (t_id + 1))}"  )
            print('\n', '-'*120)
            
        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        
        ## initialize logits

        self.reset_logits()

        print(f"\n Append policies : \n")
        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)


    def arch_parameters(self):
        """
        return policy network parameters
        """
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'logits' in name:
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
            if not ('task' in name and 'logits' in name):
                params.append(param)
        return params


    def train_sample_policy(self, temperature, hard_sampling):
        """
        Using task_logits -  Samples from the Gumbel-Softmax distribution and optionally discretizes.
        """
        print(f"        MTL2 TRAIN SAMPLE POLICY - temp: {temperature}  hard_sampling: {hard_sampling}")
        policys = []
        for t_id in range(self.num_tasks):
            task_logits_attribute = "t_id{:d}_logits".format(t_id+1)
            # task_logits_before    = getattr(self, task_logits_attribute)
            # print(f"        {task_logits_attribute}_logits: \n {task_logits_before}  \n")
            
            task_logits = getattr(self, 'task%d_logits' % (t_id + 1))
            policy = F.gumbel_softmax( task_logits, temperature, hard=hard_sampling)

            print(f"        {task_logits_attribute} Logits: \n{task_logits}  \n gumbel_softmax: \n{policy}")
            policys.append(policy)
            
        return policys


    def test_sample_policy(self, hard_sampling):

        print(f'       MTL3 TEST SAMPLE POLICY')
        self.policys = []
        
        if hard_sampling:

            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to('cuda:%d' % cuda_device)
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)

            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to('cuda:%d' % cuda_device)
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))

        else:

            for t_id in range(self.num_tasks):
                task_logits = getattr(self, 'task%d_logits' % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []

                ## Sample between (1,0) based on the probablites returned by 
                ## distribution. Initially the lower layers have probability of [1, 0]
                ## meaning the layer will always be selected. As the policy training 
                ## progresses, a different trained distribution starts to take shape
                
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)

                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                
                # setattr(self, 'policy%d' % t_id, policy)
                self.policys.append(policy)

        return self.policys


    def reset_logits(self):
        print_heading(f" Reset Logits - num layers : {sum(self.layers)}")

        num_layers = sum(self.layers)
        
        for t_id in range(self.num_tasks):
            if self.init_method == 'all_chosen':
                assert(self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer, 2)
                task_logits[:, 0] = 0

            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randn(num_layers-self.skip_layer, 2)
            
            elif self.init_method == 'equal':
                task_logits = 0.5 * torch.ones(num_layers-self.skip_layer, 2)
            
            else:
                raise NotImplementedError('Init Method %s is not implemented' % self.init_method)

            print(f" Task: {t_id}  Parm name: task{(t_id + 1)}_logits  shape: {task_logits.shape}")
            print(f" Task: {t_id}  Parm name: task{(t_id + 1)}_logits  Init Method: {self.init_method}   task_logits: {task_logits.shape}")
            self._arch_parameters = []
        
            self.register_parameter('task%d_logits' % (t_id + 1), nn.Parameter(task_logits, requires_grad=True))
            
            self._arch_parameters.append(getattr(self, 'task%d_logits' % (t_id + 1)))


    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):

        print(f" {timestring()}  -  MTL2_Dev forward  pass start: ")

        print(f" MTL2_Dev    num_train_layers: {num_train_layers}    hard_sampling:{hard_sampling}  mode:{mode} "
              f" temperature:{temperature}  is_policy:{is_policy}  self.skip_layer:{self.skip_layer}   sum(layers):{sum(self.layers)}")
        
        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer
        
        print(f'         MTL2_Dev self.layers: {sum(self.layers)}   self.skip_layer: {self.skip_layer}    num_train_layers: {num_train_layers}')

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)

        print(f'         MTL2_Dev num_training_layers  = min(sum(self.layers) - self.skip_layer, num_train_layers) =  {num_train_layers} ')
        
        # Generate features
        cuda_device = img.get_device()
        # print(f'Cuda Device is : {cuda_device}')

        ##--------------------------------------------------------------------------
        ## if is_policy is True - we pass the policy array along with the input to 
        ## the forward() routine 
        ##--------------------------------------------------------------------------
        if is_policy:
            if mode == 'train':
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == 'eval':
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == 'fix_policy':
                self.policys = self.test_sample_policy(hard_sampling)
                for p in self.policys:
                    assert(p is not None)
            else:
                raise NotImplementedError('mode %s is not implemented' % mode)

            ## build policy array to pass to backbone 
            ## nnum_train_layers: number of layers that have been policy trained.
            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            print(f'         MTL2_Dev forward  - skip layer (for policy padding): {sum(self.layers)- num_train_layers}')
            

            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            
            padding[:, 1] = 0
            padding_policys = []
            feats = []
            
            ## padding policy is the concatenation of [1, 0] for the layers 
            ## that we have NOT policy trainined concatenated with the gumbel_softmax
            ## dist for the layers that HAVE been policy trained.  
            ##  The are the last N layers where (N = num_train_layers)

            for t_id in range(self.num_tasks):
                padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()), dim=0)
                print(f"      task id: {t_id}  padding_policy shape: {padding_policy.shape}   vals: {padding_policy[-num_train_layers:]}")
                padding_policys.append(padding_policy)

                ## pass input and policy through the backbone
                feats.append(self.backbone(img, padding_policy))

        ##--------------------------------------------------------------------------
        ## if is_policy is false - simply pass img through backbones
        ## for each task, a feature set is generated.
        ##--------------------------------------------------------------------------
        else:
            feats = [self.backbone(img)] * self.num_tasks

        print(f"         MTL2_forward() feature set shape: {feats[0].shape}")


        #---------------------------------------------------------------------------
        # Get the output by passing results of backbone through task specific heads
        #---------------------------------------------------------------------------
        
        outputs = []
        
        # for t_id in range(self.num_tasks):
        #     output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id]) + \
        #              getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats[t_id]) + \
        #              getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats[t_id]) + \
        #              getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats[t_id])
        #     outputs.append(output)

        for t_id in range(self.num_tasks):
            o = []
            for c_id in [0,1,2,3]:
                o.append( getattr(self, 'task%d_fc1_c%d' % (t_id + 1, c_id))(feats[t_id]))
                # print(f"           task {t_id+1}   c: {c_id}   output  shape: {o[-1].shape}")
            output = sum(o)
            outputs.append(output)

        print('         MTL2_Dev forward  pass end: ', timestring())
        return outputs, self.policys, [None] * self.num_tasks
