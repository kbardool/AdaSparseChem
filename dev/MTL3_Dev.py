import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from dev.deeplab_resnet_dev import Deeplab_ResNet_Backbone_Dev
from utils.util             import timestring, print_heading, print_dbg, print_underline, debug_on, debug_off
from dev.sparsechem_adashare_dev  import SparseChem_Backbone, SparseChem_Classification_Module
from scipy.special          import softmax
# from models.deeplab_resnet import * 

class MTL3_Dev(nn.Module):
    """
    Create the multi-task architecture based on the SparseChem backbone

    block          :    block type used in ResNet (BasicBlock, Bottleneck)
    layers         :    List - number of blocks used in each section of the Backbone, eg. [3, 4, 23, 3]
    skip_layer     :    normally 0 is passed  
    init_method    :
    init_neg_logits:
    num_classes_tasks: 
    """
    def __init__(self, conf, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0, verbose = False):
        super(MTL3_Dev, self).__init__()
        self.num_tasks = len(num_classes_tasks)
        self.verbose = verbose


        if self.verbose:
            print_heading(f" Create {self.name} init() Start ", verbose = True)
            print_dbg(f"\n\t block            :  {block}"
                      f"\n\t layers           :  {layers}"
                      f"\n\t num_classes_tasks:  {num_classes_tasks,}"
                      f"\n\t init_method      :  {init_method}"
                      f"\n\t init_neg_logits  :  {init_neg_logits}"
                      f"\n\t skip_layer       :  {skip_layer}", verbose = True)
        
        ## Build Network Backbone
        self.backbone = SparseChem_Backbone(conf, block, layers, verbose = verbose)
        

        ## Build Task specific heads.
        for t_id, num_classes in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), SparseChem_Classification_Module(conf['tail_hidden_size'], num_classes))

        self.layers = layers
        self.num_layers = sum(layers)
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        
        ## initialize logits

        self.reset_logits()

        # print(f"\n Arch parameters : \n")
        # print(f"\t {self.arch_parameters()}")
        # print()
 
        ## initialize policys
        self.policys = [None] * self.num_tasks

        ## Display Task specific heads info
        if self.verbose:
            print_heading(f" Task Specific Heads :", verbose = True)
            for t_id, num_classes in enumerate(num_classes_tasks):
                print_underline(f" Task {t_id+1} :", verbose = True)
                print_dbg(f"task{t_id+1}_fc1_c0:  {getattr(self,'task%d_fc1_c0' % (t_id + 1))}    num classes: {num_classes} \n", verbose = True)
                
        if self.verbose:
            print_heading(f" Initialize policies ",verbose=True)
            print_dbg(self.policys, verbose = True)
            print_heading(f" {self.name} init() End ", verbose = True)

        return 


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

    ##----------------------------------------------------
    ##  Train policy 
    ##----------------------------------------------------

    def train_sample_policy(self, temperature, hard_sampling, verbose = None):
        """
        Using task_logits -  Samples from the Gumbel-Softmax distribution and optionally discretizes.
        """
        if verbose is None:
            verbose = self.verbose

        print_dbg(f"  MTL3_Dev train_sample_policy START - temperature: {temperature}  hard_sampling: {hard_sampling}  verbose: {verbose}", verbose = verbose)
        policys = []
        
        ## For each task
        for t_id in range(self.num_tasks):
            task_logits_attribute = f"task{t_id+1}_logits"
            task_logits = getattr(self, task_logits_attribute)
 
            # cuda_device  = task_logits.get_device()
            # logits       = task_logits.detach().cpu().numpy()
            policy = F.gumbel_softmax( task_logits, temperature, hard=hard_sampling)
            policys.append(policy)

            if verbose: 
                print_underline(f" task{t_id+1}", verbose= True)
                print_dbg(f" task_logits  \n {task_logits}", verbose = True)
                print_dbg(f" sampled policy from  gumbel_softmax distribution (temp:{temperature}): \n {policy}", verbose = True)
    
        print_dbg(f"  MTL3_Dev train_sample_policy END - temperature: {temperature}  hard_sampling: {hard_sampling}  verbose: {self.verbose}", verbose = verbose)

        return policys

    ##----------------------------------------------------
    ##  Test Policy  
    ##----------------------------------------------------
    def test_sample_policy(self, hard_sampling, verbose = None):
        """
        if hard_sampling == True
                Task Logits --> Argmax

        if hard_sampling == False
                Task Logits --> Softmax --> random.choice((1,0), P = softmax)
        """
        if verbose is None:
            verbose = self.verbose

        print_dbg(f" MTL3_Dev test_sample_policy() START -  hard_sampling: {hard_sampling}", verbose = verbose)
        policys = []

        for t_id in range(self.num_tasks):
            task_logits_attribute = f"task{t_id+1}_logits"
            task_logits  = getattr(self, task_logits_attribute)
            cuda_device  = task_logits.get_device()
            logits       = task_logits.detach().cpu().numpy()

            if hard_sampling:
                task_policy  = np.argmax(logits, axis=-1)
                task_policy  = np.stack((1 - task_policy, task_policy), axis=-1)
            else:
                ## Sample between (1,0) based on the probablites returned by softmax    
                ## Initially the lower layers have probability of [1, 0] - meaning the
                ## layer will always be selected. As the policy training progresses
                ## a different trained distribution starts to take shape
                task_policy = []
                distribution = softmax(logits, axis=-1)
                
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    layer_policy = [sampled, 1 - sampled]
                    task_policy.append(layer_policy)
                    # print_dbg(f" sampled policy from  distribution with P:{tmp_d[0]:.4f} {tmp_d[1]:.4f} "
                    #            "- sampled: {sampled} --> layer policy: {layer_policy}", verbose = verbose )
            ## endif

            if cuda_device != -1:
                task_policy = torch.from_numpy(np.array(task_policy)).to('cuda:%d' % cuda_device)
            else:
                task_policy = torch.from_numpy(np.array(task_policy))
            
            policys.append(task_policy)

            if verbose:
                print_underline(f"task{t_id+1} logits", verbose = verbose )
                print_dbg(f" {logits}", verbose =  verbose )
                if  hard_sampling:
                    print_underline(f"task{t_id+1} argmax /hard_sampled policy", verbose = verbose )
                    print_dbg(f" {task_policy}", verbose =  verbose )
                    print()
                else:
                    print_underline(f" task{t_id+1} softmax:", verbose = verbose )
                    print_dbg(f" {distribution}", verbose = verbose )
                    print_underline(f"task {t_id+1} sampled policy :", verbose = verbose)
                    print_dbg(f" {task_policy}", verbose = verbose )
                    print()         
            

        print_dbg(f" MTL3_Dev test_sample_policy() END -  hard_sampling: {hard_sampling}", verbose= verbose)
        return policys


    ##----------------------------------------------------
    ##  Reset logits 
    ##----------------------------------------------------
    def reset_logits(self, verbose = None):
        if verbose is None:
            verbose = self.verbose

        print_dbg('\n', verbose = verbose)
        print_heading(f" Reset Logits  num layers: {self.num_layers}  skip_layers: {self.skip_layer}    Init method: {self.init_method}", verbose = verbose)

        # num_layers = s um(self.layers)
        num_layers = self.num_layers
        
        for t_id in range(self.num_tasks):
            task_logits_attribute = f"task{(t_id + 1)}_logits"
            
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

            self._arch_parameters = []
        
            self.register_parameter(task_logits_attribute, nn.Parameter(task_logits, requires_grad=True))
            
            self._arch_parameters.append(getattr(self, task_logits_attribute))

            print_dbg(f" Task: {t_id}  Parm name: {task_logits_attribute}  shape: {task_logits.shape} "
                  f"_arch_parameters: {self._arch_parameters[-1]}", verbose = verbose)


    ##----------------------------------------------------
    ##  forward routine
    ##----------------------------------------------------
    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, policy_sampling_mode=None, verbose = None):
        '''
        input parameters:

        policy_sampling_mode:   specifies the policy sampling method 
                    'train' : Use gumbel_softmax on task_logits
                    'eval'  : when hard_sampling == False, Uses softmax on task logits 
                              When hard_sampling == True , applies argmax(task logits) 
        
        '''
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print_heading(f" {timestring()} -  MTL3_Dev.forward() START - verbose: {verbose}", verbose = verbose)
            print_dbg(f" num_train_layers: {num_train_layers}    hard_sampling:{hard_sampling}  policy sampling mode:{policy_sampling_mode} "
                f" temperature:{temperature}  is_policy:{is_policy}  self.skip_layer:{self.skip_layer}  "
                f"   num_layers:{self.num_layers}", verbose)
        
        if num_train_layers is None:
            num_train_layers = self.num_layers - self.skip_layer
            # num_train_layers = s um(self.layers) - self.skip_layer
        
        print_dbg(f" MTL3_Dev.forward() self.layers: {self.num_layers}   self.skip_layer: {self.skip_layer}    num_train_layers: {num_train_layers}", verbose = verbose)

        num_train_layers = min(self.num_layers - self.skip_layer, num_train_layers)
        # num_train_layers = min(s um(self.layers) - self.skip_layer, num_train_layers)

        print_dbg(f" MTL3_Dev.forward() num_training_layers  = min(num_layers - self.skip_layer, num_train_layers) =  {num_train_layers} \n", verbose = verbose)
        
        # Generate features
        cuda_device = img.get_device()
        # print(f'\t Cuda Device is : {cuda_device}')

        logits = [None] * self.num_tasks

        ##-----------------------------------------------------------------------------
        ## if is_policy == True - we are in policy mode, use the appropriate sampling 
        ## methofd based on policy_sampling_mode. 
        ##-----------------------------------------------------------------------------
        if is_policy:
            if policy_sampling_mode == 'train':
                self.policys = self.train_sample_policy(temperature, hard_sampling, verbose = verbose)
            elif policy_sampling_mode == 'eval':
                self.policys = self.test_sample_policy(hard_sampling, verbose = verbose)
            elif policy_sampling_mode == 'fix_policy':
                self.policys = self.test_sample_policy(hard_sampling, verbose = verbose)
                # pass
                for p in self.policys:
                    assert(p is not None)
            else:
                raise NotImplementedError(f"policy_sampling_mode [{policy_sampling_mode}] is not implemented" )

            ## place policys on appropriate device array to pass to backbone 
            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = self.num_layers - num_train_layers
            # skip_layer = s um(self.layers) - num_train_layers
            print_dbg(f" MTL3_Dev.forward()  skip_layer - first {skip_layer} layers  will NOT be included in policy training \n", verbose = verbose)
            

            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            
            ## padding[layers, 1] = P(layer,dont_select) = 0
            padding[:, 1] = 0
            padding_policys = []
            feats = []
            
            ##----------------------------------------------------------------------
            ## Pass input and Policy to the backbone
            ## padding policy is the concatenation of [1, 0] for the layers that we 
            ## are NOT policy training AND the GUMBEL_SOFTMAX dist for the layers   
            ## which WILL BE policy trained (indicated by num_train_layers)
            ##----------------------------------------------------------------------
            for t_id in range(self.num_tasks):

                padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()), dim=0)
                print_dbg(f"\n MTL3_Dev.forward() task id: {t_id}   policy after padding {skip_layer} "
                          f"non-training layers : shape: {padding_policy.shape}   \n {padding_policy}", verbose = verbose)
                padding_policys.append(padding_policy)

                ## pass input and policy through the backbone network for the task
                task_feats = self.backbone(img, padding_policy, task_id = f"task_{t_id+1}")
                feats.append(task_feats)
        
        ##--------------------------------------------------------------
        ## if IS_POLICY == False - simply pass img through backbones
        ## for each task, a feature set is generated.
        ##--------------------------------------------------------------
        else:
            feats = [self.backbone(img)] * self.num_tasks

        # print(f"\t MTL3_forward() feature set shape: {len(feats)} {feats[0].shape}")

        ##---------------------------------------------------------------
        ## Build network outputs
        ## pass results of backbone through task specific heads
        ##---------------------------------------------------------------
        outputs = []
        for t_id in range(self.num_tasks):
            o = []
            for c_id in [0]:
                o.append( getattr(self, 'task%d_fc1_c%d' % (t_id + 1, c_id))(feats[t_id]))
                # print(f" MTL3_Dev.forward()  task {t_id+1}   c: {c_id}   output  shape: {o[-1].shape}")
            output = sum(o)
            outputs.append(output)

        print_heading(f" {timestring()} - MTL3_Dev forward() END", verbose = verbose) 
        return outputs, self.policys, [None] * self.num_tasks

    @property
    def name(self):
        return 'MTL3'