import os
import pickle
# import time
import pandas as pd
import numpy  as np
from scipy.special import softmax
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
# from models.base   import Bottleneck, BasicBlock
# from dev.MTL_Instance_Dev import MTL_Instance_Dev
from dev.MTL3_Dev      import MTL3_Dev
from models.sparsechem import censored_mse_loss, censored_mae_loss
from dev.base_env_dev  import BaseEnv
from utils.util        import timestring, print_heading, print_dbg, print_underline, debug_on, debug_off
from tqdm.notebook     import tqdm,trange
from dev.sparsechem_utils_dev    import aggregate_results, compute_metrics
from dev.sparsechem_adashare_dev import SparseChemBlock
import pprint
pp = pprint.PrettyPrinter(indent=4)

    



class SparseChemEnv_Dev(BaseEnv):
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_name, tasks_num_class, 
                 device=0, 
                 is_train=True, 
                 init_neg_logits=-10, 
                 init_temperature=5.0, 
                 temperature_decay=0.965,
                 opt=None, 
                 verbose = None):
        """
        :param num_class: int, the number of classes in the dataset
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """

        print_heading(f"* {self.name}  Initializtion - verbose: {verbose}", verbose = True)
        
        self.init_neg_logits = init_neg_logits

        self.temp            = init_temperature
        self._tem_decay      = temperature_decay
        self.num_tasks       = len(tasks_num_class)
        self.input_size      = opt['input_size']
        self.norm_loss       = opt['SC']['normalize_loss']
        
        super(SparseChemEnv_Dev, self).__init__(log_dir, checkpoint_dir, exp_name, tasks_num_class, device,
                                                is_train, opt, verbose)

        print( f" is_train       :  {is_train} \n"
               f" init_neg_logits:  {self.init_neg_logits} \n"
               f" init temp      :  {self.temp} \n"
               f" decay temp     :  {self._tem_decay} \n"
               f" input_size     :  {self.input_size} \n"
               f" normalize loss :  {self.norm_loss} \n"
               f" num_tasks      :  {self.num_tasks} \n"
               f" policys        :  {self.networks['mtl-net'].policys}")



        print_heading(f"* {self.name} environment successfully created", verbose = True)



    # ##################### define networks / optimizers / losses ####################################

    def define_networks(self, tasks_num_class, verbose = None):
        if verbose is None:
            verbose = self.verbose
  
        print_heading(f" {self.name} Define Networks  ", verbose = verbose)

        if self.opt['backbone'] == 'SparseChem':
            init_method = self.opt['train']['init_method']
            block = SparseChemBlock    ## This is used by blockdrop_env to indicate the number of resnet blocks in each layer. 
            layers = [1 for _ in self.opt['hidden_sizes']]
            # layers = [1, 1, 1, 1]      ## This is used by blockdrop_env to indicate the number of resnet blocks in each layer. 

            if self.opt['policy_model'] == 'task-specific':
                self.networks['mtl-net'] = MTL3_Dev(conf = self.opt, 
                                                    block = block, 
                                                    layers = layers, 
                                                    num_classes_tasks = tasks_num_class, 
                                                    init_method = init_method, 
                                                    init_neg_logits = self.init_neg_logits, 
                                                    skip_layer = self.opt['skip_layer'], verbose = verbose)

            elif self.opt['policy_model'] == 'instance-specific':
                raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])
                # print(f'Create MTL_Instance with \n block: {block} \n layers: {layers} \n tasks_num_class: {tasks_num_class} \n init_method: {init_method}')
                # self.networks['mtl-net'] = MTL_Instance_Dev(block, layers, tasks_num_class, init_method, self.init_neg_logits, self.opt['skip_layer'])

            else:
                raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])

        else:
            raise NotImplementedError('backbone %s is not implemented' % self.opt['backbone'])


    def define_loss(self, verbose = None):
        """
        ignore_index – Specifies a target value that is ignored and does not contribute to the input gradient. 
                        When size_average is True, the loss is averaged over non-ignored targets.
        """
        if verbose is None:
            verbose = self.verbose

        print_heading(f" {self.name} Define Losses  ", verbose = verbose)
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss  = nn.L1Loss()
        self.l1_loss2 = nn.L1Loss(reduction='none')
        self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=255)
        self.cross_entropy2         = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        
        if self.dataset == 'Chembl_23_mini':
            self.loss_class = torch.nn.BCEWithLogitsLoss(reduction="none")
            self.loss_class_sum = torch.nn.BCEWithLogitsLoss(reduction="sum")
            self.loss_class_mean = torch.nn.BCEWithLogitsLoss(reduction="mean")
            self.loss_regr  = censored_mse_loss

        else:
            raise NotImplementedError('Dataset %s is not implemented' % self.dataset)


    def define_optimizer(self, policy_learning=False, verbose = None):
        """"
        if we are in policy learning phase - use SGD
        otherwise, we use ADAM
        """
        if verbose is None:
            verbose = self.verbose

        task_specific_params = self.get_task_specific_parameters()
        arch_parameters      = self.get_arch_parameters()
        backbone_parameters  = self.get_backbone_parameters()

        # TODO: add policy learning to yaml

        print_heading(f" {self.name} Define Optimizers  - policy_learning: {policy_learning} ", verbose = verbose)

        #----------------------------------------
        # weight optimizers
        #----------------------------------------
        if policy_learning:
            self.optimizers['weights'] = optim.SGD([{'params': task_specific_params, 'lr': self.opt['train']['task_lr']},
                                                    {'params': backbone_parameters , 'lr': self.opt['train']['backbone_lr']}],
                                                     momentum=0.9, weight_decay=0.0001)
        else:
            self.optimizers['weights'] = optim.Adam([{'params': task_specific_params, 'lr': self.opt['train']['task_lr']},
                                                     {'params': backbone_parameters , 'lr': self.opt['train']['backbone_lr']}],
                                                    betas=(0.5, 0.999), weight_decay=0.0001)
        
        print_dbg(f" define the weights optimizer - learning mode: {'policy' if policy_learning else 'non-policy'}", verbose = verbose)
        print_dbg(f" optimizers for weights : \n {self.optimizers['weights']}", verbose = verbose)

        #---------------------------------------
        # optimizers for alpha (logits??)
        #---------------------------------------
        if self.opt['train']['init_method'] == 'all_chosen':
            self.optimizers['alphas'] = optim.Adam(arch_parameters, lr=self.opt['train']['policy_lr'], weight_decay=5*1e-4)
        else:
            self.optimizers['alphas'] = optim.Adam(arch_parameters, lr=self.opt['train']['policy_lr'], weight_decay=5*1e-4)

        print_dbg(f"\ndefine the logits optimizer (init_method: {self.opt['train']['init_method']})", verbose = verbose)
        print_dbg(f" optimizers for alphas : \n {self.optimizers['alphas']}", verbose = verbose)
            

    def define_scheduler(self, policy_learning=False, verbose = None):
        if verbose is None:
            verbose = self.verbose

        print_heading(f" {self.name} Define Scheduler  - policy_learning: {policy_learning} ", verbose = verbose)

        if policy_learning:
            if ('policy_decay_lr_freq' in self.opt['train'].keys())  and \
               ('policy_decay_lr_rate' in self.opt['train'].keys()) :
                self.schedulers['weights'] = scheduler.StepLR(self.optimizers['weights'],
                                                              step_size=self.opt['train']['policy_decay_lr_freq'],
                                                              gamma=self.opt['train']['policy_decay_lr_rate'])
        else:
            if ('decay_lr_freq' in self.opt['train'].keys()) and \
               ('decay_lr_rate' in self.opt['train'].keys()):
                self.schedulers['weights'] = scheduler.StepLR(self.optimizers['weights'],
                                                              step_size=self.opt['train']['decay_lr_freq'],
                                                              gamma=self.opt['train']['decay_lr_rate'])
        print_dbg(self.schedulers['weights'], verbose = verbose)


    def get_sparsity_loss(self, num_train_layers, verbose = None):
        '''
        Compute policy network Sparsity Loss

        num_train_layers:  Number of layers policy network is training for

        Returns 
        self.losses['sparsity']
        '''
        if verbose is None:
            verbose = self.verbose

        print_heading(f"{timestring()} -  get_sparsity_loss START   num_train_layers: {num_train_layers}  ", verbose = verbose)
        
        self.losses['sparsity'] = {}
        self.losses['sparsity']['total'] = 0
        num_policy_layers = None

        if self.opt['policy_model'] == 'task-specific':

            for t_id in range(self.num_tasks):
                task_key = f"task{t_id+1}_logits"
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, task_key)
                else:
                    logits = getattr(self.networks['mtl-net'], task_key)

                if num_policy_layers is None:
                    num_policy_layers = logits.shape[0]
                else:
                    assert (num_policy_layers == logits.shape[0])

                if num_train_layers is None:
                    num_train_layers = num_policy_layers

                print_underline(f" Compute sparsity error for task {t_id+1}", verbose = verbose)
                print_dbg(f" num_train_layers: {num_train_layers}     num_policy_layers: {num_policy_layers}     logits shape:{logits.shape} \n", verbose = verbose)
                
                num_blocks = min(num_train_layers, logits.shape[0])
                
                ##---------------------------------------------------------------------------------------
                ## To enforce sparsity, we are make the assumption that the correct action is to
                ## NOT select the layer. Therefore for logits in each layer , [p1, p2] where: 
                ##  p1: probability layer is selected,
                ##  p2: layer NOT being selected
                ##  
                ##  the corresponding Ground Truth gt is [1]] 
                ##---------------------------------------------------------------------------------------
                gt = torch.ones((num_blocks)).long().to(self.device)

                if self.opt['diff_sparsity_weights'] and not self.opt['is_sharing']:
                    
                    ## Assign higher weights to higher layers 
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                    self.losses['sparsity'][task_key] = 2 * (loss_weights[-num_blocks:] * self.cross_entropy2(logits[-num_blocks:], gt)).mean()
                   
                    print_dbg(f" loss_weights :  {loss_weights}", verbose = verbose)
                    print_dbg(f" cross_entropy:  {self.cross_entropy2(logits[-num_blocks:], gt)}   ", verbose = verbose)
                    print_dbg(f" loss[sparsity][{task_key}]: {self.losses['sparsity'][task_key] } ", verbose = verbose)
               
                else:
                    print_dbg(f" Compute CrossEntropyLoss between \n\t Logits   : {logits[-num_blocks:]} \n\t and gt: {gt} \n", verbose = verbose)
                    self.losses['sparsity'][task_key]  = self.cross_entropy_sparsity(logits[-num_blocks:], gt)
                    self.losses['sparsity'][task_key] *= self.opt['train']['Lambda_sparsity'] 
                
                print_dbg(f"\t loss[sparsity][{task_key}]: {self.losses['sparsity'][task_key]:.6f}  \n", verbose = verbose)

                self.losses['sparsity']['total'] += self.losses['sparsity'][task_key]

            print_underline(f" loss[sparsity][total]: {self.losses['sparsity']['total'] }", verbose = verbose)
        
        else:
            raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])

        print_heading(f"{timestring()} -  get_sparsity_loss END   num_train_layers: {num_train_layers}  ", verbose = verbose)
        return


    def get_hamming_loss(self, num_policy_layers = None, verbose = None):
        """
        Compute policy network sharing loss \
             Compute hamming distance
        """
        if verbose is None:
            verbose = self.verbose

        print_heading(f"{timestring()} -  get_hamming_loss START  ", verbose = verbose)        
        
        self.losses['sharing'] = {}
        self.losses['sharing']['total'] = 0
        total_sharing_loss = 0 
        
        ## Get logits for task I 
        for t_id in range(self.num_tasks):

            task_i_attr = f"task{t_id + 1}_logits" 
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                logits_i = getattr(self.networks['mtl-net'].module, task_i_attr )
            else:
                logits_i = getattr(self.networks['mtl-net'], task_i_attr)

            ## Calculate (L - l)/L
            if num_policy_layers is None:
                num_policy_layers = logits_i.shape[0]
                if self.opt['diff_sparsity_weights']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                else:
                    loss_weights = (torch.ones((num_policy_layers)).float()).to(self.device)
            else:
                assert (num_policy_layers == logits_i.shape[0]) 

            task_i_loss = 0.0
            print_underline(f" {task_i_attr} Sharing (Hamming) loss: ",verbose=verbose)

            ## Get logits for all other tasks  
            for t_j in range(t_id, self.num_tasks):
                task_j_attr = f"task{t_j + 1}_logits" 
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits_j = getattr(self.networks['mtl-net'].module, task_j_attr)
                else:
                    logits_j = getattr(self.networks['mtl-net'], task_j_attr)

                if num_policy_layers is None:
                    num_policy_layers = logits_j.shape[0]
                else:
                    assert (num_policy_layers == logits_j.shape[0])

                task_i_j_loss = torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0]))
                task_i_loss += task_i_j_loss
                total_sharing_loss  +=  task_i_j_loss
                
                print_underline(f" between {task_i_attr} and {task_j_attr} : ",verbose=verbose)
                print_dbg(f" {task_i_attr:12s}: {logits_i[:,0]}  "
                          f"\n {task_j_attr:12s}: {logits_j[:,0]}  "
                          f"\n abs diff    : {(torch.abs(logits_i[:, 0] - logits_j[:, 0]))} "
                          f"\n loss_weights: {loss_weights}"
                          f"\n\t sum       : {torch.sum(torch.abs(logits_i[:, 0] - logits_j[:, 0])):.5f} "
                          f"\n\t weighted  : {torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0])):.5f} ", verbose = verbose)

            print_underline(f" Total Sharing loss for {task_i_attr} :  {task_i_loss} ", verbose=verbose)

        self.losses['sharing']['total']  = total_sharing_loss * self.opt['train']['Lambda_sharing'] 
        
        print_underline(f" Total Unweighted Sharing loss for all tasks:  {total_sharing_loss:.5f} ", verbose=verbose)
        print_underline(f" Total Weighted   Sharing loss for ALL TASKS:  {self.losses['sharing']['total']:.5f} ", verbose=verbose)
        print_heading(f"{timestring()} -  get_hamming_loss END  ", verbose = verbose)        
        return 


    def get_task_specific_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            task_specific_params = self.networks['mtl-net'].module.task_specific_parameters()
        else:
            task_specific_params = self.networks['mtl-net'].task_specific_parameters()

        return task_specific_params


    def get_arch_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            arch_parameters = self.networks['mtl-net'].module.arch_parameters()
        else:
            arch_parameters = self.networks['mtl-net'].arch_parameters()

        return arch_parameters


    def get_network_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            network_parameters = self.networks['mtl-net'].module.network_parameters()
        else:
            network_parameters = self.networks['mtl-net'].network_parameters()
        return network_parameters


    def get_backbone_parameters(self):
        if self.opt['backbone'] == 'WRN':
            network_parameters = []
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for name, param in self.networks['mtl-net'].module.named_parameters():
                    if name.startswith('task') and ('logits' in name or 'fc' in name):
                        continue
                    else:
                        network_parameters.append(param)
            else:
                for name, param in self.networks['mtl-net'].named_parameters():
                    if name.startswith('task') and ('logits' in name or 'fc' in name):
                        continue
                    else:
                        network_parameters.append(param)
            return network_parameters
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                backbone_parameters = self.networks['mtl-net'].module.backbone_parameters()
            else:
                backbone_parameters = self.networks['mtl-net'].backbone_parameters()
            return backbone_parameters


    def optimize(self, lambdas, is_policy=False, flag='update_w', num_train_layers=None, hard_sampling=False, verbose = None):
        """
        1) Make forward pass 
        2) compute losses based on tasks 
        3)
        """
        if verbose is None:
            verbose = self.verbose

        print_heading(f" {timestring()} - SparseChem network optimize() start ")
        print_dbg(f"\t flag: {flag}      num_train_layers: {num_train_layers}     is_policy: {is_policy}      hard_sampling: {hard_sampling}"
              f"    task lambdas:  {lambdas}", verbose = False)
        
        self.task_lambdas = lambdas
        ## reset losses & get training hyper parms
        self.losses = {}
        self.losses['parms'] = {}
        self.losses['parms']['gumbel_temp'] = self.temp
        self.losses['parms']['train_layers'] = 0 if num_train_layers is None else num_train_layers 
        curr_lrs = self.schedulers['weights'].get_last_lr()
        for i in range(len(curr_lrs)):
            self.losses['parms'][f'lr_{i}'] = curr_lrs[i]
     
        
        ## Forward Pass 
        self.forward(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling, verbose = verbose)     
        
        ## Compute Task Losses
        self.compute_task_losses()              

        ## backward pass - Weights
        if flag == 'update_w':
            self.backward_network()

        ## Backward pass - alphas
        elif flag == 'update_alpha':
            self.backward_policy(num_train_layers)

            
        else:
            raise NotImplementedError('Training flag %s is not implemented' % flag)

        print_heading(f" {timestring()} - SparseChem network optimize() end ", verbose = verbose)


    def optimize_fix_policy(self, lambdas, num_train_layer=None, verbose = None):
        if verbose is None:
            verbose = self.verbose

        # print('num_train_layers in optimize = ', num_train_layers)
        self.task_lambdas = lambdas 

        self.forward_fix_policy(num_train_layer)
        
        self.get_classification_loss()

        self.backward_network()


    def val_fix_policy(self, num_train_layers=None):
        
        self.forward_fix_policy(num_train_layers)
        # self.resize_results()

        self.get_classification_loss()

        # for t_id, task in enumerate(self.tasks):
        #     task_num_class = self.tasks_num_class[t_id]
        #     task_key = f"task{t_id+1}"
        #     self.metrics[task_key] = {}

        #     pred, gt, pixelAcc, err = self.seg_error(task_num_class)
        #     self.metrics['seg']['pred'] = pred
        #     self.metrics['seg']['gt'] = gt
        #     self.metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
        #     self.metrics['seg']['err'] = err
                                           
        # if 'seg' in self.tasks:
        #     self.metrics['seg'] = {}
        #     seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
        #     pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
        #     self.metrics['seg']['pred'] = pred
        #     self.metrics['seg']['gt'] = gt
        #     self.metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
        #     self.metrics['seg']['err'] = err
            

        return  

 
    def forward(self, is_policy, num_train_layers = None, policy_sampling_mode = "train" , hard_sampling = False, verbose = None):
        '''
        mode: Policy sampling mode (train, eval, fix_policy)
        '''
        if verbose is None:
            verbose = self.verbose

        if verbose: 
            print_heading(f" {timestring()} - SparseChem network FORWARD() start ", verbose = True)
            print_dbg(f"\t num_train_layers:{num_train_layers}   is_policy: {is_policy}   policy_sampling_mode: {policy_sampling_mode}    hard_sampling: {hard_sampling}", verbose = True)

        outputs, policys, logits = self.networks['mtl-net'](img              = self.input, 
                                                            temperature      = self.temp, 
                                                            is_policy        = is_policy, 
                                                            num_train_layers = num_train_layers, 
                                                            hard_sampling    = hard_sampling,
                                                            policy_sampling_mode  = policy_sampling_mode)

        for t_id,  task in enumerate(self.tasks):
            print_dbg(f" set attributes: task id {t_id+1}  task: {task}", verbose = verbose)
            print_dbg(f"    output[{t_id+1}]:      {outputs[t_id].shape}", verbose = verbose)
            print_dbg(f"    policy{t_id+1}:        {policys[t_id]}", verbose = verbose)
            print_dbg(f"    logits{t_id+1}:        {logits[t_id]} ", verbose = verbose)
            print_dbg(f"    task{t_id+1}_logits:   {getattr(self.networks['mtl-net'], f'task{t_id+1}_logits')}", verbose = verbose)

            setattr(self, 'task%d_pred' % (t_id+1), outputs[t_id])
            setattr(self, 'policy%d' % (t_id+1), policys[t_id])
            setattr(self, 'logit%d' % (t_id+1), logits[t_id])
    
        print_heading(f" {timestring()} - SparseChem network FORWARD() end ", verbose = verbose)
        return 


    def forward_fix_policy(self, is_policy = True, num_train_layers = None, hard_sampling = False, verbose = None):

        if verbose is None:
            verbose = self.verbose

        print_heading(f"forward_fix_policy - policy model: {self.opt['policy_model']}  ", verbose = verbose)

        ## if Task-Specific we call MTL2.forward()
        if self.opt['policy_model'] == 'task-specific':
            outputs, _, _ = self.networks['mtl-net'](self.img, 
                                                     self.temp, 
                                                     is_policy,  
                                                     num_train_layers=num_train_layers, 
                                                     hard_sampling = hard_sampling , 
                                                     mode='fix_policy')

        ## if Instance-specific we call MTL2_Instance.forward(), which also takes the policy as an input parameter
        # elif self.opt['policy_model'] == 'instance-specific':
        #     policys = []
        #     for task in self.opt['tasks']:
        #         policys.append(getattr(self, '%s_policy' % task))
        #         print(' policy : {policys[-1]}')
        #
        #     outputs, _, _ = self.networks['mtl-net'](self.img, self.temp, True, policys=policys, num_train_layers=num_train_layers, mode='fix_policy')

        else:
            raise ValueError('policy model = %s is not supported' % self.opt['policy_model'])

        for t_id, task in enumerate(self.tasks):
            # setattr(self, '%s_pred' % task, outputs[t_id])
            setattr(self, 'task%d_pred' % (t_id+1), outputs[t_id])


    def backward_policy(self, num_train_layers, verbose = None):
        """
        Compute losses on policy and back-propagate
        """
        if verbose is None:
            verbose = self.verbose

        print_heading(f" {timestring()} - SparseChem backward Policy start", verbose = verbose )
        
        self.optimizers['alphas'].zero_grad()
        self.losses['losses'] = {}
        self.losses['losses_mean'] = {}
        self.losses['total'] = {}
        self.losses['total_mean'] = {}

        total_loss = 0
        total_tasks_loss =  0 
        total_tasks_loss_mean = 0 

        if self.opt['is_sharing']:
            self.get_hamming_loss()   
            sharing_loss =  self.losses['sharing']['total']

        if self.opt['is_sparse']:
            self.get_sparsity_loss(num_train_layers)   # places sparsity loss into self.losses[sparsity][total]
            sparsity_loss = self.losses['sparsity']['total']

        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            # print_dbg(f"\t backward_policy - {task_key}   loss: {self.task_lambdas[t_id]} * {self.losses[task_key]['total']:.4f}"
                #   f" = {task_loss:.4f}" , verbose = verbose )            
            self.losses['losses'][task_key] = self.losses[task_key]['cls_loss']
            self.losses['losses_mean'][task_key] = self.losses[task_key]['cls_loss_mean'] 
            total_tasks_loss += self.losses[task_key]['cls_loss']
            total_tasks_loss_mean += self.losses[task_key]['cls_loss_mean']

        self.losses['losses']['total'] = total_tasks_loss   
        self.losses['losses_mean']['total'] = total_tasks_loss_mean

        self.losses['total']['total'] = total_tasks_loss + sparsity_loss + sharing_loss
        self.losses['total']['total_mean'] = total_tasks_loss_mean + sparsity_loss + sharing_loss
                
        if True:
            print(f" Task losses: {total_tasks_loss:8.4f}     mean: {total_tasks_loss_mean:8.4f}     Sharing: {sharing_loss:.5e}     Sparsity: {sparsity_loss:.5e}"
                  f"     Total: {self.losses['total']['total']:8.4f}     mean: {self.losses['total']['total_mean']:8.4f}")
            # print() 
            # print(f" Total Task losses                                             : {total_tasks_loss:.4f}")
            # print(f" Total Task losses mean                                        : {total_tasks_loss_mean:.4f}")
            # print(f" Sharing loss  (Lambda_sharing * Sharing (Hamming) Loss)       : {self.opt['train']['Lambda_sharing']:.2f} * {self.losses['sharing']['total']:.4f}"
            #       f" = {sharing_loss:.5e}")
            # print(f" Sparsity loss (Lambda_sharing * Sparsity (CrossEntropy) Loss) : {self.opt['train']['Lambda_sparsity']:.2f} * {self.losses['sparsity']['total']:.4f}"
            #       f" = {sparsity_loss:.5e}")
            # print(f" Backward Policy        Total Loss (Task + Sharing + Sparsity) : {self.losses['total']['total']:.4f}      mean: {self.losses['total']['total_mean']:.4f}")



        self.losses['total']['total'].backward()

        self.optimizers['alphas'].step()
        
        ## Added 1/11/22 - currently alpha has no schedulers.    
        if 'alphas' in self.schedulers.keys():
            self.schedulers['alphas'].step()

        print_heading(f" {timestring()} - SparseChem backward Policy end  " , verbose = verbose )
        return


    def backward_network(self, verbose = None):
        '''
        Aggregate losses and back-propagate
        '''
        if verbose is None:
            verbose = self.verbose
        
        print_heading(f" {timestring()} - SparseChem backward Network start" , verbose = verbose)
        
        self.optimizers['weights'].zero_grad()
        self.losses['losses'] = {}
        self.losses['losses_mean'] = {}
        self.losses['total'] = {}
        self.losses['total_mean'] = {}

        
        total_loss = 0
        total_tasks_loss =  0 
        total_tasks_loss_mean =  0 
        
        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            # print_dbg(f"\t backward_policy - {task_key}   loss: {self.task_lambdas[t_id]} * {self.losses[task_key]['total']:.4f}"
                #   f" = {task_loss:.4f}" , verbose = verbose )            
            self.losses['losses'][task_key] = self.losses[task_key]['cls_loss']
            self.losses['losses_mean'][task_key] = self.losses[task_key]['cls_loss_mean']
            total_tasks_loss += self.losses[task_key]['cls_loss']
            total_tasks_loss_mean += self.losses[task_key]['cls_loss_mean']

        self.losses['losses']['total'] = total_tasks_loss        
        self.losses['losses_mean']['total'] = total_tasks_loss_mean

        self.losses['total']['total'] = total_tasks_loss
        self.losses['total']['total_mean'] = total_tasks_loss_mean

        if verbose:
            print(f" Task losses: {total_tasks_loss:8.4f}     mean: {total_tasks_loss_mean:8.4f} ")

        print_dbg(f"\t backward Network - Sum of weighted task losses :  {self.losses['losses']['total']:.4f} ", verbose=verbose)
        print_dbg(f"\t {timestring()} - backward pass & optimize step begin  ", verbose=verbose )


        self.losses['total']['total'].backward()

        self.optimizers['weights'].step()

        if 'weights' in self.schedulers.keys():
            self.schedulers['weights'].step()

        print_heading(f" {timestring()} - BlockDrop backward Network end  ", verbose = verbose)
        return 


    def decay_temperature(self, decay_ratio=None, verbose = False):
        tmp = self.temp
        if decay_ratio is None:
            self.temp *= self._tem_decay
        else:
            self.temp *= decay_ratio
        print_dbg(f"Change temperature from {tmp:.5f} to{self.temp:.5f}", verbose = verbose )


    def get_sample_policy(self, hard_sampling):
        '''
        ## TODO: rename from Sample Policy to get_sample_policy
        Sample network policy 
        if hard_sampling == True
                Network Task Logits --> Argmax
        if hard_sampling != True
                Network Task Logits --> Softmax --> random.choice((1,0), P = softmax)
        '''
        # dist1, dist2 = self.get_policy_prob()
        # print(np.concatenate((dist1, dist2), axis=-1))
        policys = self.networks['mtl-net'].test_sample_policy(hard_sampling)
        return policys
            

    def set_sample_policy(self, policys):
        '''
        save list of policys as policy attributes 
        if hard_sampling == True
                Network Task Logits --> Argmax
        if hard_sampling != True
                Network Task Logits --> Softmax --> random.choice((1,0), P = softmax)
        '''
        for t_id, p in enumerate(policys):
            setattr(self, 'policy%d' % (t_id+1), p)         
       

    def get_policy_prob(self, verbose = False):
        distributions = []

        if self.opt['policy_model'] == 'task-specific':
            for t_id in range(self.num_tasks):
                task_attr = f"task{t_id+1}_logits"
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, task_attr).detach().cpu().numpy()
                else:
                    logits = getattr(self.networks['mtl-net'], task_attr).detach().cpu().numpy()

                print_underline(f" task {t_id+1} logits: ",verbose = verbose)
                print_dbg(f"{logits}", verbose = verbose)
                distributions.append(softmax(logits, axis=-1))

        elif self.opt['policy_model'] == 'instance-specific':
            for t_id in range(self.num_tasks):
                task_attr = f"task{t_id+1}_logits"
                logit = getattr(self, task_attr).detach().cpu().numpy()
                distributions.append(logit.mean(axis=0))
        else:
            raise ValueError('policy mode = %s is not supported' % self.opt['policy_model']  )

        return distributions


    def get_policy_logits(self, verbose = False):
        logits_list = []

        if self.opt['policy_model'] == 'task-specific':
            for t_id in range(self.num_tasks):
                task_attr = f"task{t_id+1}_logits"
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, task_attr).detach().cpu().numpy()
                else:
                    logits = getattr(self.networks['mtl-net'], task_attr).detach().cpu().numpy()

                print_underline(f" task {t_id+1} logits: ",verbose = verbose)
                print_dbg(f"{logits}", verbose = verbose)
                logits_list.append(logits)

        elif self.opt['policy_model'] == 'instance-specific':
            for t_id in range(self.num_tasks):
                task_attr = f"task{t_id+1}_logits"
                logit = getattr(self, task_attr).detach().cpu().numpy()
                logits_list.append(logit.mean(axis=0))
        else:
            raise ValueError('policy mode = %s is not supported' % self.opt['policy_model']  )

        return logits_list


    def get_current_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.detach().cpu().numpy()
            policys.append(policy)

        return policys


    def get_current_state(self, current_iter):
        current_state = super(SparseChemEnv_Dev, self).get_current_state(current_iter)
        current_state['temp'] = self.temp
        return current_state


    def save_policy(self, label):
        policy = {}
        for t_id in range(self.num_tasks):
            tmp = getattr(self, 'policy%d' % (t_id + 1))
            policy['task%d_policy' % (t_id + 1)] = tmp.cpu().data
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        with open(save_path, 'wb') as handle:
            pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        with open(save_path, 'rb') as handle:
            policy = pickle.load(handle)

        for t_id in range(self.num_tasks):
            print(f"setting policy{t_id+1} attribute ....")
            setattr(self, 'policy%d' % (t_id + 1), policy['task%d_policy' % (t_id+1)])
            print(getattr(self, 'policy%d' % (t_id + 1)))


    def check_exist_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        return os.path.exists(save_path)


    def load_snapshot(self, snapshot):
        super(SparseChemEnv_Dev, self).load_snapshot(snapshot)
        self.temp = snapshot['temp']
        return snapshot['iter']


    def fix_weights(self):
        print_heading(f" Fix Weights - disable gradient flow through main computation graph     ")

        if self.opt['backbone'] == 'WRN':
            network_params = self.get_network_parameters()
            for param in network_params:
                param.requires_grad = False
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for param in self.networks['mtl-net'].module.backbone.parameters():
                    param.requires_grad = False

            else:
                for param in self.networks['mtl-net'].backbone.parameters():
                    param.requires_grad = False

            task_specific_parameters = self.get_task_specific_parameters()
            for param in task_specific_parameters:
                param.requires_grad = False


    def free_weights(self, fix_BN):
        '''
        allow gradient computation through weight parameters
        '''
        print_heading(f" Free Weights - allow gradient flow through the main computation graph ")
        
        if self.opt['backbone'] == 'WRN':
            network_params = self.get_network_parameters()
            for param in network_params:
                param.requires_grad = True
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for name, param in self.networks['mtl-net'].module.backbone.named_parameters():
                    param.requires_grad = True

                    if fix_BN and 'bn' in name:
                        param.requires_grad = False
            else:
                for name, param in self.networks['mtl-net'].backbone.named_parameters():
                    param.requires_grad = True
                    if fix_BN and 'bn' in name:
                        param.requires_grad = False

            task_specific_parameters = self.get_task_specific_parameters()

            for param in task_specific_parameters:
                param.requires_grad = True


    def fix_alpha(self):
        """Fix architecture parameters - disable gradient flow through alpha computation graph"""
        print_heading(f" Fix Alpha - disable gradient flow through alpha computation graph  (policy network)")

        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = False


    def free_alpha(self):
        """Fix architecture parameters - allow gradient flow through alpha computation graph"""
        print_heading(f" Free Alpha - allow gradient flow through alpha computation graph  (policy network)")
        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = True


    def cuda(self, gpu_ids):
        super(SparseChemEnv_Dev, self).cuda(gpu_ids)
        policys = []

        for t_id in range(self.num_tasks):
            if not hasattr(self, 'policy%d' % (t_id+1)):
                return
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.to(self.device)
            policys.append(policy)

        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            setattr(self.networks['mtl-net'].module, 'policys', policys)
        else:
            setattr(self.networks['mtl-net'], 'policys', policys)


    def cpu(self):
        super(SparseChemEnv_Dev, self).cpu()
        print(f'sparsechem_env.cpu()')
        policys = []

        for t_id in range(self.num_tasks):
            if not hasattr(self, 'policy%d' % (t_id+1)):
                return
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.to(self.device)
            policys.append(policy)

        ## nn.DataParallel only applies to GPU configurations
        
        # if isinstance(self.networks['mtl-net'], nn.DataParallel):
        #     setattr(self.networks['mtl-net'].module, 'policys', policys)
        # else:
        setattr(self.networks['mtl-net'], 'policys', policys)
        print(f'environ cpu policy: {policys}')


    def display_parameters(self):

        task_specific_params = self.get_task_specific_parameters()
        arch_parameters      = self.get_arch_parameters()
        backbone_parameters  = self.get_backbone_parameters()
        
        print('-----------------------')
        print(' task specific parms  :')
        print('-----------------------')
        for i,j  in enumerate(task_specific_params):
            print(i, type(j), j.shape)
        print('\n')
        print('-----------------------')
        print('\n arch_parameters    :')
        print('-----------------------')
        for i,j in enumerate(arch_parameters):
            print(i, type(j), j.shape)
        print('\n')
        print('-----------------------')
        print('\n backbone parameters:')
        print('-----------------------')
        for i,j in enumerate(backbone_parameters):
            print(i, type(j), j.shape)
        
        return

    @property
    def name(self):
        return 'SparseChemEnv_Dev'


    def set_inputs(self, batch, input_size):
        """
        :param batch: {'images': a tensor [batch_size, c, video_len, h, w], 'categories': np.ndarray [batch_size,]}
        """
        self.batch = batch
        self.input = torch.sparse_coo_tensor(batch["x_ind"],
                                             batch["x_data"],
                                             size = [batch["batch_size"], input_size], 
                                             dtype = torch.float,
                                             device = self.device)

    def resize_results(self):
        pass


    def compute_task_losses(self,  instance=False, verbose = False):
        """
        seg_pred: semantic segmentation predictions. shape: (BatchSize, NumClasses, W, H)
        """
        if verbose is None:
            verbose = self.verbose

        print_heading(f" {timestring()} - SparseChem network compute_task_losses start ", verbose = verbose)
        # print(f" Tasks num classes: {self.tasks_num_class}    ")
        self.y_hat = {}


        for t_id,  task in enumerate(self.tasks):
            task_key     = f"task{t_id+1}"
            task_pred    = task_key+"_pred"
            task_ind     = task_key+"_ind"
            task_data    = task_key+"_data"
            task_weights = task_key+"_weights"

            self.losses[task_key] = {}
            # ## build ground truth 
            yc_ind  = self.batch[task_ind].to(self.device, non_blocking=True)
            yc_data = self.batch[task_data].to(self.device, non_blocking=True)                
            trn_weights =( self.batch[task_weights].training_weight).to(self.device, non_blocking=True)                 
            yc_w    = trn_weights[yc_ind[1]]

            ## yc_hat_all shape : [batch_size , # classes]         
            yc_hat_all = getattr(self, task_pred)
            ## yc_hat shape : ( batch_size x # classes,) (one dimensional)
            yc_hat  = yc_hat_all[yc_ind[0], yc_ind[1]]

            # self.losses[task_key][task_ind]  = yc_ind
            # self.losses[task_key][task_data] = yc_data
            # self.y_hat[task_key] = yc_hat
            
            # print(f" {task_key} Classification loss calculation ")
            # print(f"--------------------------------------------")
            # print(f"\t yc_ind[0]   ({ yc_ind[0].shape}) : {yc_ind[0]}")
            # print(f"\t yc_ind[1]   ({ yc_ind[1].shape}) : {yc_ind[1]}")
            # print(f"\t yc_data     ({   yc_data.shape}) : {yc_data}")
            # print(f"\t yc_hat_all  ({yc_hat_all.shape}) : {yc_hat_all}")
            # print(f"\t yc_hat      ({    yc_hat.shape}) : {yc_hat}")
            # print(f"\t yc_w[ind[1]({trn_weights.shape}) : {trn_weights}")
            # print(f"\t yc_w        ({      yc_w.shape}) : {yc_w}")
            # print(f"\t yc_loss        : {self.loss_class(yc_hat, yc_data)}")
            # print(f"\t yc_loss * yc_w : {self.loss_class(yc_hat, yc_data) * yc_w}")
            # print(f"\t yc_w.sum()     : {yc_w.sum()}")            

            ##---------------------------------------------------------------------------------------
            ## SparseChem normalizes the loss prior to backproping. The default is the batchsize
            ##---------------------------------------------------------------------------------------
            ## norm_loss : Is the Normalization constant to divide the loss (default uses batch size)
            ##
            ## if self.norm_loss is None:
            ##     norm = self.batch['batch_size']
            ## else:
            ##     norm = self.norm_loss
            ##
            ## Normalize loss we want to back prop - total_tasks_loss or whatever
            ##  
            ##   loss_normalized  = total_task_loss / norm
            ##   loss_normalzied.backward()
            ##
            ##---------------------------------------------------------------------------------------
            # task_loss_none = self.loss_class(yc_hat, yc_data)
            task_loss_sum  = self.loss_class_sum(yc_hat, yc_data)
            task_loss_mean = self.loss_class_mean(yc_hat, yc_data)

            if self.norm_loss is None:
                norm = self.batch['batch_size']
            else:
                norm = self.norm_loss
            
            ## Normalize loss we want to back prop - total_tasks_loss or whatever
             
            self.losses[task_key]['cls_loss']      = self.task_lambdas[t_id] * task_loss_sum / norm
            self.losses[task_key]['cls_loss_mean'] = self.task_lambdas[t_id] * task_loss_mean

            
            print_dbg(f"  task {t_id+1} "  
                      f"  BCE Mean : {self.losses[task_key]['cls_loss_mean']:.4f} = {task_loss_mean:.4f} * {self.task_lambdas[t_id]}" 
                      f"  BCE Norm : {self.losses[task_key]['cls_loss']:.4f}      = {task_loss_sum:.4f}  * {self.task_lambdas[t_id]} / {norm} ", verbose = verbose)            
            

            
            # print_dbg(f"      task training weight {self.batch[task_weights].training_weight.shape}   {self.batch[task_weights].training_weight}" , verbose = True )   
            # print_dbg(f"      task training weight {yc_w.shape}  {yc_w}" , verbose = True )   
            # print_dbg(f"      task {t_id+1} y_hat shape: {yc_hat.shape}    y_data.shape{yc_data.shape}" , verbose = True )   
            # print_dbg(f"      " , verbose = verbose )   
            # print_dbg(f"  BCE None: {task_loss_none.shape}  view(batch_size,-1): {task_loss_none.view(self.batch['batch_size'], -1).shape}"
            #           f"   mean: {task_loss_none.view(self.batch['batch_size'], -1).mean(dim=-1).shape}" , verbose = verbose )   
            # print_dbg(f"      yc_hat: {yc_hat.shape}  task_loss_none: {task_loss_none.shape}  " , verbose = verbose )   

            # print(f"\t Loss for task id {t_id+1} after  normalization: {self.losses[task_key]['total']} ")            
            # print('\n\n')
            
            
        print_heading(f" {timestring()} - SparseChem network compute_task_losses end ", verbose = verbose)
        return


    def classification_metrics(self, verbose= None ):
        if verbose is None:
            verbose = self.verbose

        print_heading(f" {timestring()} - SparseChem classification_metrics start ", verbose = verbose)
        self.metrics = {}

        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            task_pred    = task_key+"_pred"
            task_ind     = task_key+"_ind"
            task_data    = task_key+"_data"
            task_weights = task_key+"_weights"
 
            yc_hat_all = getattr(self, task_pred)

            self.metrics[task_key] =  {}        
            self.metrics[task_key]['task_num_class']  = self.tasks_num_class[t_id]
            self.metrics[task_key]['task_lambda']     = self.opt['lambdas'][t_id]
            self.metrics[task_key]['cls_loss']            = self.losses[task_key]['cls_loss'] 
            self.metrics[task_key]['cls_loss_mean']       = self.losses[task_key]['cls_loss_mean'] 
            self.metrics[task_key]['yc_ind']          = self.batch[task_ind].to(self.device, non_blocking=True)
            self.metrics[task_key]['yc_data']         = self.batch[task_data].to(self.device, non_blocking=True)    
            self.metrics[task_key]['yc_hat']          = yc_hat_all[self.metrics[task_key]['yc_ind'][0], self.metrics[task_key]['yc_ind'][1]]

            ## batch_size x number of populated outputs in the yc batch , indiccated by yc_ind. 
            self.metrics[task_key]['yc_wghts_sum']    = self.batch[task_weights].training_weight[self.metrics[task_key]['yc_ind'][1]].sum()
            
            ## weights of each indivudal class within a classification task
            self.metrics[task_key]['yc_trn_weights']  = ( self.batch[task_weights].training_weight).to(self.device, non_blocking=True)  
            
            ## weights of each indivudal class within a regression task
            self.metrics[task_key]['yc_aggr_weights'] = ( self.batch[task_weights].aggregation_weight)

        print_heading(f" {timestring()} - SparseChem classification_metrics end ", verbose = verbose)

        return


    def evaluate(self, dataloader, tasks_class, is_policy= False, num_train_layers=None, hard_sampling=False,
                 device = None, progress=True, eval_iter=-1, leave = False, verbose = False):

        self.val_metrics = {}
        
        task_loss_sum = {}
        task_loss_sum_mean = {}
        
        all_tasks_loss_sum = 0.0
        all_tasks_loss_mean = 0.0
        
        task_loss_avg= {}
        task_class_weights  = {}
        all_tasks_class_weights = 0.0
        
        data = {}
        num_class_tasks  = dataloader.dataset.class_output_size
        # num_regr_tasks   = dataloader.dataset.regr_output_size
        # class_w = tasks_class.aggregation_weight

        self.eval()

        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            data[task_key] = { "yc_ind":  [],
                        "yc_data": [],
                        "yc_hat":  [],
                        }
            task_loss_sum[task_key] = 0.0
            task_loss_sum_mean[task_key] = 0.0
            task_loss_avg[task_key] = 0.0
            task_class_weights[task_key]  = 0.0

        if eval_iter == -1:
            eval_iter = len(dataloader)


        with torch.no_grad():

            ## Note: len(tt) is equal to len(dataloader)
            # with tqdm(dataloader, total=eval_iter, initial=0, leave=True, disable=(progress==False), position=0, desc = "validation") as tt:
            #     for batch_idx, batch in enumerate(tt,1) :
            #         # if batch_idx > eval_iter :
            #         #     break

            with trange(1,eval_iter+1, total=eval_iter, initial=0, leave=leave, disable=(progress==False), position=0, desc = "validation") as t_validation:
                
                for batch_idx in t_validation:
            
                    print_dbg(f"\n + Validation Loop Start - batch_idx:{batch_idx}  eval_iter: {eval_iter}  t_validation: {len(t_validation)} \n",verbose=verbose)

                    batch = next(dataloader)
                    self.set_inputs(batch, dataloader.dataset.input_size)  
                    self.val2(is_policy, num_train_layers, hard_sampling, verbose = verbose)            
                    
                    for t_id, _ in enumerate(self.tasks):
                        task_key = f"task{t_id+1}"
                        all_tasks_loss_sum      += self.metrics[task_key]['cls_loss']
                        all_tasks_loss_mean     += self.metrics[task_key]['cls_loss_mean']

                        task_loss_sum[task_key]      += self.metrics[task_key]['cls_loss']                    
                        task_loss_sum_mean[task_key] += self.metrics[task_key]['cls_loss_mean']
                        
                        task_class_weights[task_key]  += self.metrics[task_key]['yc_wghts_sum']
                        
                        data[task_key]['yc_trn_weights']  = self.metrics[task_key]['yc_trn_weights']
                        data[task_key]['yc_aggr_weights'] = self.metrics[task_key]['yc_aggr_weights']
                        
                        ## storing Y data for EACH TASK (for metric computations)
                        for key in ["yc_ind", "yc_data", "yc_hat"]:
                            if (key in self.metrics[task_key]) and (self.metrics[task_key][key] is not None):
                                data[task_key][key].append(self.metrics[task_key][key].cpu())            
                    
                    t_validation.set_postfix({'bch_idx': batch_idx, 
                                    'loss'   : f"{all_tasks_loss_sum.item()/batch_idx:.4f}" ,
                                    'row_ids': f"{batch['row_id'][0]} - {batch['row_id'][-1]}" ,
                                    'task_wght': f"{self.metrics[task_key]['yc_wghts_sum']}"})

                    if verbose:                
                        for t_id, _ in enumerate(self.tasks):
                            task_key = f"task{t_id+1}"
                            print_dbg(f" + Validation Loop - batch_idx:{batch_idx}  eval_iter: {eval_iter}\n"
                                    f"    task {t_id+1:3d}: loss     : {self.metrics[task_key]['cls_loss']:.4f}   sum(err)     : {task_loss_sum[task_key]:.4f}    avg(err)=sum(err)/batch_id: {task_loss_sum[task_key]/batch_idx:.4f}   \n"
                                    f"    task {t_id+1:3d}: loss_mean: {self.metrics[task_key]['cls_loss_mean']:.4f}   sum(err_mean): {task_loss_sum_mean[task_key]:.4f}    avg(err_mean): {task_loss_sum_mean[task_key]/batch_idx:.4f}",
                                    verbose = True)
                            print_dbg(f"    self.metrics[task_key][yc_wghts_sum] {self.metrics[task_key]['yc_wghts_sum']}  task_weights[task_key]: {task_class_weights[task_key]}  ", verbose = True)
                        
                        print_dbg(f"\n + Validation Loop end - batch_idx:{batch_idx}  eval_iter: {eval_iter} \n"
                                f"    all tasks: loss_sum     : {all_tasks_loss_sum:.4f}   avg(loss_sum)     : {all_tasks_loss_sum/batch_idx:.4f} \n"
                                f"    all tasks: loss_sum_mean: {all_tasks_loss_mean:6.4f}    avg(loss_sum_mean): {all_tasks_loss_mean / batch_idx:.4f}", verbose = True)
                    

            ##-----------------------------------------------------------------------
            ## All Validation batches have been feed to network - calcualte metrics
            ##-----------------------------------------------------------------------
            if verbose:
                print_heading(f" + Validation Loops complete- batch_idx:{batch_idx}  eval_iter: {eval_iter}", verbose = True)
                for t_id, _ in enumerate(self.tasks):
                    task_key = f"task{t_id+1}"
                    print_dbg(f"    task {t_id+1:3d}: sum(err)     : {task_loss_sum[task_key]:6.4f}    avg(err)=sum(err)/batch_id: {task_loss_sum[task_key]/batch_idx:.4f}   \n"
                            f"    task {t_id+1:3d}: sum(err_mean): {task_loss_sum_mean[task_key]:6.4f}   avg(err_mean): {task_loss_sum_mean[task_key]/batch_idx:.4f}\n"
                            f"    self.metrics[task_key][yc_wghts_sum] {self.metrics[task_key]['yc_wghts_sum']}  task_weights[task_key]: {task_class_weights[task_key]} \n", verbose = True)
                    # print_dbg(f"    yc_aggr_weight          : {data[task_key]['yc_aggr_weights']}", verbose = True)
                
                print_dbg(f"    all tasks: loss_sum     : {all_tasks_loss_sum:.4f}   avg(loss_sum)     : {all_tasks_loss_sum/eval_iter:.4f}      \n"
                        f"    all tasks: loss_sum_mean: {all_tasks_loss_mean:.4f}   avg(loss_sum_mean): {all_tasks_loss_mean / eval_iter:.4f} \n ", verbose = True)


            all_tasks_loss_sum /= batch_idx
            all_tasks_loss_mean /= batch_idx
    
            self.val_metrics["loss"]  = {"total": all_tasks_loss_sum.item()}
            self.val_metrics["loss_mean"] = {"total" : all_tasks_loss_mean.item()}
            task_classification_metrics = []
            task_aggregation_weights = [] 

            for t_id, task in enumerate(self.tasks):
                task_key = f"task{t_id+1}"
                self.val_metrics["loss"][task_key] = task_loss_sum[task_key].cpu().item() / batch_idx
                self.val_metrics["loss_mean"][task_key] = task_loss_sum_mean[task_key].cpu().item() / batch_idx 

                self.val_metrics[task_key] = {}
                yc_ind  = torch.cat(data[task_key]["yc_ind"] , dim=1).numpy()
                yc_data = torch.cat(data[task_key]["yc_data"], dim=0).numpy()
                yc_hat  = torch.cat(data[task_key]["yc_hat"] , dim=0).numpy()
                yc_aggr_weight = data[task_key]["yc_aggr_weights"]
                

                # out["classification"]     = compute_metrics(yc_ind[1], y_true=yc_data, y_score=yc_hat, num_tasks=num_class_tasks)
                self.val_metrics[task_key]["classification"] = compute_metrics(yc_ind[1], 
                                                                               y_true=yc_data, 
                                                                               y_score=yc_hat, 
                                                                               num_tasks=num_class_tasks[t_id])

                ## to_dict(): convert pandas series to dict to make it compatible with print_loss()
                # out["classification_agg"] = aggregate_results(out["classification"], weights=class_w)
                self.val_metrics[task_key]["classification_agg"] = aggregate_results(self.val_metrics[task_key]["classification"], 
                                                                                    weights = yc_aggr_weight).to_dict() 

                self.val_metrics[task_key]['classification_agg']['sc_loss'] = task_loss_sum[task_key].cpu().item() / eval_iter 
                self.val_metrics[task_key]["classification_agg"]["logloss"] = task_loss_sum[task_key].cpu().item() / task_class_weights[task_key].cpu().item()

                task_classification_metrics.append(self.val_metrics[task_key]['classification'])
                task_aggregation_weights.append(yc_aggr_weight)
            
                all_tasks_class_weights += task_class_weights[task_key].cpu().item()
                # self.val_metrics[task_key]["classification_agg"]["logloss"] = loss_class_sum[task_key].cpu().item() / loss_class_weights[task_key].cpu().item()    
                # self.val_metrics[task_key]["classification_agg"]["num_tasks_total"] = dataloader.dataset.class_output_size[t_id]
                # self.val_metrics[task_key]["classification_agg"]["num_tasks_agg"]   = (tasks_class.aggregation_weight > 0).sum()
                # self.val_metrics[task_key]['classification_agg']['yc_weights_sum'] = self.metrics[task_key]["yc_wghts_sum"].cpu().item()
    
            ## Calculate aggregated metrics across all task groups
            
            self.val_metrics['aggregated'] = aggregate_results( pd.concat(task_classification_metrics),
                                                                weights = np.concatenate(task_aggregation_weights)).to_dict()
            self.val_metrics['aggregated']['sc_loss'] = all_tasks_loss_sum / eval_iter 
            self.val_metrics['aggregated']["logloss"] = all_tasks_loss_sum / all_tasks_class_weights
        
            
            self.train()
            return


    def val2(self, is_policy, num_train_layers=None, hard_sampling=False, verbose = None):
        
        ## reset losses & get training hyper parms       
        self.losses = {}
        self.losses['parms'] = {}
        self.losses['parms']['gumbel_temp'] = self.temp
        curr_lrs = self.schedulers['weights'].get_last_lr()
        for i in range(len(curr_lrs)):
            self.losses['parms'][f'lr_{i}'] = curr_lrs[i]

        ## only difference between forward_eval and forward is the policy_sampling_mode, refactored these
        # if is_policy:
        #     self.forward_eval(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)
        # else:
        #     self.forward(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)

        if is_policy:
            policy_sampling = "eval"
        else:
            policy_sampling = None     ## originally set to "train" , but never used when is_policy == False
        
        self.forward(is_policy = is_policy, num_train_layers = num_train_layers, policy_sampling_mode = policy_sampling, hard_sampling = hard_sampling)
        # self.resize_results()
        
        self.compute_task_losses(verbose = verbose)
        # pp.pprint(self.losses)

        self.classification_metrics(verbose = False)
        # pp.pprint(self.metrics)
        return



    # def forward_eval(self, is_policy,  num_train_layers = None, hard_sampling = False,  verbose = None):
    #     '''
    #     Multi-task network called with mode == 'eval'
    #             mode:   specifies the policy sampling method 
    #             mode == 'eval'  AND  hard_sampling == False:  Uses softmax on task logits to generate sample policies 
    #     '''
    #     if verbose is None:
    #         verbose = self.verbose
    #
    #     if verbose:
    #         print_heading(f" {timestring()} - SparseChem network FORWARD_EVAL() start ")        
    #         print_dbg(" num_train_layers:{num_train_layers}   is_policy: {is_policy}    hard_sampling: {hard_sampling}", verbose = verbose)
    #
    #     outputs, policys, logits = self.networks['mtl-net'](img              = self.input, 
    #                                                         temperature      = self.temp, 
    #                                                         is_policy        = is_policy, 
    #                                                         num_train_layers = num_train_layers, 
    #                                                         hard_sampling    = hard_sampling,  
    #                                                         mode             = 'eval')  
    #
    #     for t_id, task in enumerate(self.tasks):
    #         setattr(self, 'task%d_pred' % (t_id + 1), outputs[t_id])
    #         setattr(self, 'policy%d' % (t_id + 1), policys[t_id])
    #         setattr(self, 'logit%d' % (t_id + 1), logits[t_id])
    #
    #         if verbose:
    #             print_underline(f" MTL-net output: task id {t_id+1}  task: {task}", verbose = True)
    #             print_dbg(f"    output[{t_id+1}]:  {outputs[t_id]}.shape", verbose = True)
    #             print_dbg(f"    policy{t_id+1}  :    {policys[t_id]}", verbose = True)
    #             print_dbg(f"    logits{t_id+1}  :    {logits[t_id]} ", verbose = True)
    #             print_dbg(f"    task{t_id+1}_logits:   {getattr(self.networks['mtl-net'], f'task{t_id+1}_logits')}   ", verbose = True)
    #
    #
    #     print_heading(f" {timestring()} - SparseChem network FORWARD_EVAL() end ", verbose = verbose)

