import os
import sys 
import pprint
# import pickle
# import time

import torch
import pandas as pd
import numpy  as np
from tqdm     import tqdm,trange
from scipy.special import softmax
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
from torch.autograd import Variable
# from models.base   import Bottleneck, BasicBlock
# from dev.MTL_Instance_Dev import MTL_Instance_Dev

from models            import MTL3
from envs.base_env     import BaseEnv
from utils             import (timestring, print_heading, print_dbg, print_underline, debug_on, debug_off, 
                                write_loss_csv_heading, is_notebook)
from utils             import censored_mse_loss, censored_mae_loss, aggregate_results, compute_metrics
from models            import SparseChemBlock
# if is_notebook():
#     from tqdm.notebook     import tqdm,trange
# else:
#     from tqdm     import tqdm,trange
pp = pprint.PrettyPrinter(indent=4)



class SparseChemEnv(BaseEnv):
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
        self.verbose = False if verbose is None else verbose
        print_heading(f"* {self.name}  Initializtion - verbose: {verbose}", verbose = verbose)
        
        self.init_neg_logits    = init_neg_logits
        self.gumbel_temperature = init_temperature
        self.temperature_decay  = temperature_decay
        self.num_tasks          = len(tasks_num_class)
        self.input_size         = opt['input_size']
        self.norm_loss          = opt['SC']['normalize_loss']
        self.task_lambdas       = opt['lambdas']
        self.weight_optimizer   = opt['train']['weight_optimizer']
        self.policy_optimizer   = opt['train']['policy_optimizer']
        
        super(SparseChemEnv, self).__init__(log_dir, checkpoint_dir, exp_name, tasks_num_class, device,
                                                is_train, opt, verbose)
        if self.verbose:
            print_underline(f"Input parms :", verbose = True)
            print( f" log_dir        :  {self.log_dir} \n"
                f" checkpoint_dir :  {self.checkpoint_dir} \n"
                f" exp_name       :  {exp_name} \n"
                f" tasks_num_class:  {self.tasks_num_clas} \n"
                f" device         :  {self.devic} \n"
                f" device id      :  {self.device_i} \n"
                f" dataset        :  {self.dataset} \n"
                f" tasks          :  {self.tasks} \n"
                f"                   \n"
                f" is_train       :  {is_train} \n"
                f" init_neg_logits:  {self.init_neg_logits} \n"
                f" init temp      :  {self.gumbel_temperature} \n"
                f" decay temp     :  {self.temperature_decay} \n"
                f" input_size     :  {self.input_size} \n"
                f" normalize loss :  {self.norm_loss} \n"
                f" num_tasks      :  {self.num_tasks} \n"
                f" policys        :  {self.networks['mtl-net'].policys}")
        
        write_loss_csv_heading(self.loss_csv_file, self.initialize_loss_metrics())

        print_heading(f"* {self.name} environment successfully created", verbose = True)


    def define_networks(self, verbose = False):
        print_heading(f" {self.name} Define Networks  ", verbose = verbose)

        if self.opt['backbone'] == 'SparseChem':
            # init_method = self.opt['train']['init_method']
            block = SparseChemBlock    ## This is used by blockdrop_env to indicate the number of resnet blocks in each layer. 

            if self.opt['policy_model'] == 'task-specific':
                self.networks['mtl-net'] = MTL3(conf = self.opt, 
                                                    block = block, 
                                                    layers = self.layers, 
                                                    num_classes_tasks = self.tasks_num_class, 
                                                    init_method = self.opt['train']['init_method'],
                                                    init_neg_logits = self.init_neg_logits, 
                                                    skip_layer = self.opt['skip_layer'], 
                                                    verbose = False)

            elif self.opt['policy_model'] == 'instance-specific':
                raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])
                # print(f'Create MTL_Instance with \n block: {block} \n layers: {self.layers} \n '
                #        'tasks_num_class: {tasks_num_class} \n init_method: {self.opt['train']['init_method']}')
                # self.networks['mtl-net'] = MTL_Instance_Dev(block, self.layers, tasks_num_class, init_method, self.init_neg_logits, self.opt['skip_layer'])

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
        
        if self.dataset == 'Chembl23_mini':
            self.loss_class     = torch.nn.BCEWithLogitsLoss(reduction="none")
            self.loss_class_sum = torch.nn.BCEWithLogitsLoss(reduction="sum")
            self.loss_class_mean = torch.nn.BCEWithLogitsLoss(reduction="mean")
            self.loss_regr  = censored_mse_loss

        else:
            raise NotImplementedError('Dataset %s is not implemented' % self.dataset)


    def define_optimizer(self, policy_learning=False, verbose = False):
        """"
        if we are in policy learning phase - use SGD
        otherwise, we use ADAM
        """
        # if verbose is None:
            # verbose = self.verbose

        task_specific_params = self.get_task_specific_parameters()
        arch_parameters      = self.get_arch_parameters()
        backbone_parameters  = self.get_backbone_parameters()

        # TODO: add policy learning to yaml

        print_heading(f" {self.name} Define Optimizers  - policy_learning: {policy_learning} ", verbose = verbose)

        #----------------------------------------
        # weight optimizers
        #----------------------------------------
        # Using optim.Adam with LR = 0.01 causes a severe jump in validation errors 

        # Between the swtich from warmup mode to training mode, the define_optimizer routine is called again whcih 
        # results in resetting the LR . By this change we basically keep the optimizer unchange during the transision 
        # from warmup to training 
        if policy_learning:
            self.optimizers['weights'] = optim.SGD([{'params': task_specific_params, 'lr': self.opt['train']['task_lr']/ 100},
                                                    {'params': backbone_parameters , 'lr': self.opt['train']['backbone_lr']/ 100}],
                                                    momentum=0.9, weight_decay=0.0001)
        else:
            if self.weight_optimizer == 'adam':
                self.optimizers['weights'] = optim.Adam([{'params': task_specific_params, 'lr': self.opt['train']['task_lr']},
                                                        {'params': backbone_parameters , 'lr': self.opt['train']['backbone_lr']}],
                                                        betas=(0.9, 0.999), weight_decay=0.0001)
            elif self.weight_optimizer == 'sgd':
                self.optimizers['weights'] = optim.SGD([{'params': task_specific_params, 'lr': self.opt['train']['task_lr'] },
                                                        {'params': backbone_parameters , 'lr': self.opt['train']['backbone_lr']}],
                                                        momentum=0.9, weight_decay=0.0001)
            else:
                raise NotImplementedError('Weight optimizer %s is not implemented' % self.weight_optimizer)

        print_dbg(f" define the weights optimizer - learning mode: {'policy' if policy_learning else 'non-policy'}", verbose = verbose)
        print_dbg(f" optimizers for weights : \n {self.optimizers['weights']}", verbose = verbose)

        #---------------------------------------
        # optimizers for alpha (logits??)
        #---------------------------------------
        # if self.opt['train']['init_method'] == 'all_chosen':
        if self.policy_optimizer == 'adam':
            self.optimizers['alphas'] = optim.Adam(arch_parameters, lr=self.opt['train']['policy_lr'], weight_decay=5*1e-4)
        elif self.policy_optimizer == 'sgd':
            self.optimizers['alphas'] = optim.SGD(arch_parameters, lr=self.opt['train']['policy_lr'], 
                                                 momentum=0.9, weight_decay=1.0e-4)
        else:
            raise NotImplementedError('Policy otimizer %s is not implemented' % self.weight_optimizer)

        print_dbg(f"\ndefine the logits optimizer (init_method: {self.opt['train']['init_method']})", verbose = verbose)
        print_dbg(f" optimizers for alphas : \n {self.optimizers['alphas']}", verbose = verbose)
            

    def define_scheduler(self, policy_learning=False, verbose = False):
        '''
        Adjust learning rate according to schedule

        if policy_decay_lr_* not defined, will revert to decay_lr_*
        '''
        # if verbose is None:
            # verbose = self.verbose

        # print_heading(f" {self.name} Define Scheduler  - policy_learning: {policy_learning} ", verbose = verbose)


        if  ('policy_decay_lr_freq' in self.opt['train'].keys())  and \
            ('policy_decay_lr_rate' in self.opt['train'].keys()) :
            # self.schedulers['alphas'] = scheduler.StepLR(self.optimizers['alphas'],
                                                            # step_size=self.opt['train']['policy_decay_lr_freq'],
                                                                # gamma=self.opt['train']['policy_decay_lr_rate'])

            self.schedulers['alphas'] = scheduler.ReduceLROnPlateau(self.optimizers['alphas'], 
                                                                     mode = 'min',
                                                                     factor=self.opt['train']['policy_decay_lr_rate'],
                                                                     patience=self.opt['train']['policy_decay_lr_freq'],   
                                                                     cooldown=self.opt['train']['policy_decay_lr_cooldown'],
                                                                     verbose = True)                                                                

        if ('decay_lr_freq' in self.opt['train'].keys()) and \
            ('decay_lr_rate' in self.opt['train'].keys()):
            # self.schedulers['weights'] = scheduler.StepLR(self.optimizers['weights'],
                                                            # step_size=self.opt['train']['decay_lr_freq'],
                                                            # gamma=self.opt['train']['decay_lr_rate'])

            self.schedulers['weights'] = scheduler.ReduceLROnPlateau(self.optimizers['weights'], 
                                                                     mode = 'min',
                                                                     factor=self.opt['train']['decay_lr_rate'],
                                                                     patience=self.opt['train']['decay_lr_freq'],   
                                                                     cooldown=self.opt['train']['decay_lr_cooldown'],
                                                                     verbose = True)

        # print_dbg(self.schedulers['weights'], verbose = verbose)


    def compute_task_losses(self,  instance=False, verbose = False):
        """
        Computes task classification losses for processed batch 
        """
        # print_heading(f" {timestring()} - SparseChem network compute task losses start  "
                    #   f" tasks_num_classes: {self.tasks_num_class} ", verbose = verbose)

        self.y_hat  = {}

        for t_id,  task in enumerate(self.tasks):
            task_key     = f"task{t_id+1}"
            task_pred    = task_key+"_pred"

            # ## build ground truth 
            yc_ind  = self.batch[task_key]['ind'].to(self.device, non_blocking=True)
            yc_data = self.batch[task_key]['data'].to(self.device, non_blocking=True)                
            trn_weights = self.batch[task_key]['trn_weights'].to(self.device, non_blocking=True)                 
            yc_w    = trn_weights[yc_ind[1]]

            ## yc_hat_all shape : [batch_size , # classes]         
            ## yc_hat shape : ( batch_size x # classes,) (one dimensional)
            yc_hat_all = getattr(self, task_pred)
            yc_hat  = yc_hat_all[yc_ind[0], yc_ind[1]]

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

            # print_dbg(f" loss normalizer is : {self.loss_normalizer})", verbose = True)

            # task_loss_none = self.loss_class(yc_hat, yc_data)
            loss_sum  = self.loss_class_sum(yc_hat, yc_data)
            loss_mean = self.loss_class_mean(yc_hat, yc_data)
            
            ## 'task_lambdas' are per-task weights 
            ## 'lambda_tasks' is a regularizer added that is used to counter lambda_sparsity and 
            ##                lambda_sharing if necessary
             
            tl_sum  = (self.task_lambdas[t_id] * loss_sum)  / self.loss_normalizer
            tl_mean = (self.task_lambdas[t_id] * loss_mean) / self.loss_normalizer 

            self.losses['task'][task_key]        = tl_sum    
            self.losses['task_mean'][task_key]   = tl_mean    
           
            self.losses['task']['total']        += tl_sum     
            self.losses['task_mean']['total']   += tl_mean    

        self.losses['task']['total']         *= self.opt['train']['lambda_tasks'] 
        self.losses['task_mean']['total']    *= self.opt['train']['lambda_tasks'] 
        
        # print_dbg(f"  task {t_id+1} "  
        #           f"  BCE Mean : {self.losses['task'][task_key]:.4f} = {loss_mean:.4f} * {self.task_lambdas[t_id]}" 
        #           f"  BCE Norm : {self.losses['task_mean'][task_key]:.4f} = {loss_sum:.4f}  * {self.task_lambdas[t_id]} / {norm} ", verbose = verbose)            
        # print_heading(f" {timestring()} - SparseChem network compute task losses end ", verbose = verbose)

        return


    def get_sparsity_loss(self, num_train_layers, verbose = None):
        '''
        Compute policy network Sparsity Loss
        Higher Sparsity loss means increased sharing among the different task groups
        
        Input Parms
            num_train_layers:  Number of layers policy network is training for

        Returns 
            self.losses['sparsity']
        '''
        if verbose is None:
            verbose = self.verbose

        # print_heading(f"{timestring()} -  get_sparsity_loss START   num_train_layers: {num_train_layers}  ", verbose = verbose)
        
        num_policy_layers = None

        # if self.opt['policy_model'] != 'task-specific':
        #     raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])
        # else:

        for t_id in range(self.num_tasks):
            task_key = f"task{t_id+1}"

            logits = self.get_task_logits(t_id)

            if num_policy_layers is None:
                num_policy_layers = logits.shape[0]
            else:
                assert (num_policy_layers == logits.shape[0])

            if num_train_layers is None:
                num_train_layers = num_policy_layers

            # print_dbg(f" sparsity_loss:  task {t_id+1} logits: {logits}", verbose= verbose)
            # print_underline(f" Compute sparsity error for task {t_id+1}", verbose = verbose)
            # print_dbg(f" num_train_layers: {num_train_layers}     num_policy_layers: {num_policy_layers}     logits shape:{logits.shape} \n", verbose = verbose)
            
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
                
                # print_dbg(f" loss_weights :  {loss_weights}", verbose = verbose)
                # print_dbg(f" cross_entropy:  {self.cross_entropy2(logits[-num_blocks:], gt)}   ", verbose = verbose)
                # print_dbg(f" loss[sparsity][{task_key}]: {self.losses['sparsity'][task_key] } ", verbose = verbose)
            
            else:
                # print_dbg(f" Compute CrossEntropyLoss between \n\t Logits   : {logits[-num_blocks:]} \n\t and gt: {gt} \n", verbose = verbose)
                self.losses['sparsity'][task_key]  = self.cross_entropy_sparsity(logits[-num_blocks:], gt)
                self.losses['sparsity'][task_key] *= self.opt['train']['lambda_sparsity'] 
            
            # print_dbg(f"\t loss[sparsity][{task_key}]: {self.losses['sparsity'][task_key]:.6f}  \n", verbose = verbose)
            self.losses['sparsity'][task_key] /= self.loss_normalizer
            self.losses['sparsity']['total'] += self.losses['sparsity'][task_key]

        # print_underline(f" loss[sparsity][total]: {self.losses['sparsity']['total'] }", verbose = verbose)
        
        # print_heading(f"{timestring()} -  get_sparsity_loss END   num_train_layers: {num_train_layers}  ", verbose = verbose)
        return


    def get_sharing_loss(self, num_policy_layers = None, verbose = None):
        """
        Used to induce sharing between different task policies
        Compute policy network sharing loss using Hamming distance

        Higher sharing loss means less sharing among the task groups

        """
        # if verbose is None:
        #     verbose = self.verbose

        total_sharing_loss = 0 
        
        ## Get logits for task I 
        for t_id in range(self.num_tasks):
            task_i_loss = 0.0
            logits_i = self.get_task_logits(t_id)

            # print_underline(f" task {t_id+1} Sharing (Hamming) loss: ",verbose=verbose)            
            # print_dbg(f" logits task : {logits_i}", verbose= verbose)

            ## Calculate (L - l)/L
            if num_policy_layers is None:
                num_policy_layers = logits_i.shape[0]
                if self.opt['diff_sparsity_weights']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                else:
                    loss_weights = (torch.ones((num_policy_layers)).float()).to(self.device)
            else:
                assert (num_policy_layers == logits_i.shape[0]) 

            ## Get logits for all other tasks  
            for t_j in range(t_id, self.num_tasks):

                logits_j = self.get_task_logits(t_j)

                if num_policy_layers is None:
                    num_policy_layers = logits_j.shape[0]
                else:
                    assert (num_policy_layers == logits_j.shape[0])

                task_i_j_loss = torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0]))
                task_i_loss += task_i_j_loss
                total_sharing_loss  +=  task_i_j_loss
                
                # print_underline(f" between tasks {t_id+1} and {t_j+1} : ",verbose=verbose)
                # print_dbg(f" task {t_id+1}: {logits_i[:,0]}  "
                #           f"\n task  {t_j+1}: {logits_j[:,0]}  "
                #           f"\n abs diff    :  {(torch.abs(logits_i[:, 0] - logits_j[:, 0]))} "
                #           f"\n loss_weights:  {loss_weights}"
                #           f"\n sum         :  {torch.sum(torch.abs(logits_i[:, 0] - logits_j[:, 0])):.5f} "
                #           f"\n weighted    :  {torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0])):.5f} ", verbose = verbose)

            # print_underline(f" Total Sharing loss for task {t_id} :  {task_i_loss} ", verbose=verbose)

        self.losses['sharing']['total']  = (total_sharing_loss   * self.opt['train']['lambda_sharing']) /self.loss_normalizer
        
        # print_underline(f" Total Unweighted Sharing loss for all tasks:  {total_sharing_loss:.5f} ", verbose=verbose)
        # print_underline(f" Total Weighted   Sharing loss for ALL TASKS:  {self.losses['sharing']['total']:.5f} ", verbose=verbose)
        return 


    def compute_losses(self, num_train_layers = None, verbose = False):
        ## Compute Task Losses
        self.compute_task_losses(verbose=verbose) 

        if self.opt['is_sharing']:
            self.get_sharing_loss()   

        if self.opt['is_sparse']:
            self.get_sparsity_loss(num_train_layers)   # places sparsity loss into self.losses[sparsity][total]
 
        ## losses have already been normalized in their respective functions
        ## Calc final losses 
        self.losses['total']['tasks']      =  self.losses['task']['total'] 
        self.losses['total']['policy']     =  self.losses['sparsity']['total'] + self.losses['sharing']['total'] 
        self.losses['total']['total']      =  self.losses['task']['total']   +  self.losses['sparsity']['total'] + self.losses['sharing']['total']
        self.losses['total']['total_mean'] =  self.losses['task_mean']['total']   +  self.losses['sparsity']['total'] + self.losses['sharing']['total']


    def optimize(self, task_lambdas, is_policy=False, flag='update_w', num_train_layers=None, 
                policy_sampling_mode = "train", 
                hard_sampling=False, verbose = None):
        """
        Train on one input batch 

        1) Make forward pass 
        2) compute losses based on tasks 
        3) run backward step (backward_weights or policy based on flag)
        """
        # if verbose:
        #     print_heading(f" {timestring()} - SparseChem network optimize() start ", verbose = verbose)
        #     print_dbg(f"\t flag: {flag}      num_train_layers: {num_train_layers}     is_policy: {is_policy}      hard_sampling: {hard_sampling}"
        #               f"    task lambdas:  {lambdas}", verbose = verbose)
        self.task_lambdas = task_lambdas

        ## reset losses & get loss_normalizer 
        self.set_loss_normalizer()
        self.losses = self.initialize_loss_metrics(num_train_layers)
     
        self.forward(is_policy = is_policy,
                     policy_sampling_mode = policy_sampling_mode,  
                     num_train_layers = num_train_layers, 
                     hard_sampling = hard_sampling, 
                     verbose = verbose)     

        self.compute_losses(num_train_layers)

        if flag == 'update_weights':
            self.backward_network()
        elif flag == 'update_alpha':
            self.backward_policy(num_train_layers)
        else:
            raise NotImplementedError('Training flag %s is not implemented' % flag)
        
        return 


    def optimize_fix_policy(self, task_lambdas, is_policy = True, num_train_layers=None, 
                            policy_sampling_mode = "fix_policy", 
                            hard_sampling = False, verbose = None):
        """
        num_train_layers = None   --> All layers are invovled in training 
        hard_sampling    = False  -->
        """
        self.task_lambdas = task_lambdas 
        
        ## reset losses & get loss_normalizer 
        self.set_loss_normalizer()
        self.losses = self.initialize_loss_metrics(num_train_layers)
     
        self.forward(is_policy = is_policy,                             ## Always True
                     policy_sampling_mode = policy_sampling_mode,       ## Always fix_policy
                     num_train_layers = num_train_layers,            
                     hard_sampling = hard_sampling,                     ## Always False (no effect)
                     verbose = verbose)     
                    
        self.compute_losses(num_train_layers)

        self.backward_network()

        for scheduler in self.schedulers.values():
            scheduler.step()         
        
        return 


    def validate(self, is_policy, num_train_layers=None,
                 policy_sampling_mode = 'eval',  
                 hard_sampling=False, verbose = None):
        """
        prev called  val2
        Validation on one input batch 

        1) Make forward pass 
        2) compute losses based on tasks 
        """
        ##  policy_sampling = None    Originally set to "train" when is_policy == False, but was never used 
        # policy_sampling = "eval" if is_policy else None 
 
        ## reset losses & get loss_normalizer 
        self.set_loss_normalizer()
        self.losses = self.initialize_loss_metrics(num_train_layers)
        
        self.forward(is_policy = is_policy, 
                    policy_sampling_mode = policy_sampling_mode, 
                    num_train_layers = num_train_layers, 
                    hard_sampling = hard_sampling,
                    verbose = verbose)

        
        self.compute_losses(num_train_layers)
        
        self.gather_batch_data(verbose = False)

        return


    def forward(self, is_policy, num_train_layers = None, policy_sampling_mode = "train" , hard_sampling = False, verbose = None):
        '''
        mode: Policy sampling mode (train, eval, fix_policy)
        '''
        # if verbose is None:
        #     verbose = self.verbose

        # if verbose: 
        #     print_heading(f" {timestring()} - SparseChem network FORWARD() start ", verbose = True)
        #     print_dbg(f"\t num_train_layers:{num_train_layers}   is_policy: {is_policy}   "
        #               f"policy_sampling_mode: {policy_sampling_mode}    hard_sampling: {hard_sampling}", verbose = True)

        outputs, policys, logits = self.networks['mtl-net'](input            = self.input, 
                                                            temperature      = self.gumbel_temperature, 
                                                            is_policy        = is_policy, 
                                                            num_train_layers = num_train_layers, 
                                                            hard_sampling    = hard_sampling,
                                                            policy_sampling_mode  = policy_sampling_mode)

        for t_id,  task in enumerate(self.tasks):
            # if verbose:
            #     print_dbg(f" set attributes: task id {t_id+1}  task: {task}", verbose = verbose)
            #     print_dbg(f"    output[{t_id+1}]:      {outputs[t_id].shape}", verbose = verbose)
            #     print_dbg(f"    policy{t_id+1}:        {policys[t_id]}", verbose = verbose)
            #     print_dbg(f"    logits{t_id+1}:        {logits[t_id]} ", verbose = verbose)
            #     print_dbg(f"    task{t_id+1}_logits:   {getattr(self.networks['mtl-net'], f'task{t_id+1}_logits')}", verbose = verbose)

            setattr(self, 'task%d_pred' % (t_id+1), outputs[t_id])
            setattr(self, 'policy%d' % (t_id+1), policys[t_id])
            setattr(self, 'logit%d' % (t_id+1), logits[t_id])
    
        # if verbose:
        #     print_heading(f" {timestring()} - SparseChem network FORWARD() end ", verbose = verbose)
        return 


    def backward_policy(self, num_train_layers, verbose = None):
        """
        Compute losses on policy and back-propagate
        """
        # if verbose:
            # print(f" B-P Task losses: {self.losses['task']['total']:8.4f}     mean: {self.losses['task_mean']['total']:8.4f}"
                #   f"     Sparsity: {self.losses['sparsity']['total']:.5e}     Sharing: { self.losses['sharing']['total']:.5e}"
                #   f"     Total: {self.losses['total']['total']:8.4f}     mean: {self.losses['total']['total_mean']:8.4f}")
        
        self.losses['total']['backprop']   =  (self.losses['task']['total']   +  self.losses['sparsity']['total'] + self.losses['sharing']['total']) 

        self.optimizers['alphas'].zero_grad()

        self.losses['total']['backprop'].backward()        
        self.optimizers['alphas'].step()

        # self.display_trained_logits(0)        
        return


    def backward_network(self, verbose = False):
        '''
        Aggregate losses and back-propagate
        '''
        # if verbose:
            # print(f" B-N Task losses: {self.losses['task']['total']:8.4f}     Mean: {self.losses['task_mean']['total']:8.4f}"
                #   f"     Sparsity: {self.losses['sparsity']['total']:.5e}     Sharing: {self.losses['sharing']['total']:.5e}"
                #   f"     Total: {self.losses['total']['total']:8.4f}     mean: {self.losses['total']['total_mean']:8.4f}")

        self.losses['total']['backprop']   =  self.losses['task']['total']    

        ## Here we only pass the Weights related loss through the backward step 
        ## Question - if the alpha parms are frozen (no grad), then is there any effect if we
        ## passed all loss (weight + policy) backwards???
 
        self.optimizers['weights'].zero_grad()

        self.losses['total']['backprop'].backward()
        self.optimizers['weights'].step()

        return 

    
    def initialize_loss_metrics(self, num_train_layers = None):

        loss_metrics = {}
        loss_metrics['parms'] = {}
        loss_metrics['parms']['gumbel_temp'] = self.gumbel_temperature
        loss_metrics['parms']['train_layers'] = 0 if num_train_layers is None else num_train_layers 
        
        if self.is_train:
            # curr_lrs = self.schedulers['weights'].get_last_lr()
            # for i, lr  in enumerate(curr_lrs):
                # loss_metrics['parms'][f'lr_{i}'] = lr
            for i, parm_grp in enumerate(self.optimizers['weights'].param_groups):
                loss_metrics['parms'][f'lr_{i}'] = parm_grp['lr']

            # for i, lr in enumerate(self.schedulers['alphas'].get_last_lr()):
            #     self.losses['parms'][f'policy_lr_{i}'] = lr
            # loss_metrics['parms'][f'policy_lr'] = self.schedulers['alphas'].get_last_lr()[0]
            loss_metrics['parms'][f'policy_lr'] = self.optimizers['alphas'].param_groups[0]['lr']

            loss_metrics['parms'][f'lambda_sparsity'] = self.opt['train']['lambda_sparsity'] 
            loss_metrics['parms'][f'lambda_sharing'] = self.opt['train']['lambda_sharing'] 
            loss_metrics['parms'][f'lambda_tasks'] = self.opt['train']['lambda_tasks'] 
        
        
        loss_metrics['task']       = {'total': 0.0}
        loss_metrics['task_mean']  = {'total': 0.0}
        loss_metrics['sparsity']   = {'total': 0.0}
        loss_metrics['sharing']    = {'total': 0.0}

        for t_id,  task in enumerate(self.tasks):
            task_key     = f"task{t_id+1}"
            loss_metrics['task'][task_key]      = 0.0     
            loss_metrics['task_mean'][task_key] = 0.0
            loss_metrics['sparsity'][task_key]    = 0.0

        loss_metrics['total'] = {'backprop'  : 0.0,
                                 'task'      : 0.0,
                                 'policy'    : 0.0,
                                 'total'     : 0.0,
                                 'total_mean': 0.0}
        return loss_metrics


    def gather_batch_data(self, verbose= None ):
        """
        Gather batch data required for calculation of validation metrics for current batch 

         yc_wghts_sum    :  batch_size x number of populated outputs in the yc batch , indiccated by yc_ind. 
         yc_trn_weights  :  weights of each indivudal class within a classification task
         yc_aggr_weights :  weights of each indivudal class within a regression task

        """
        if verbose is None:
            verbose = self.verbose

        # print_heading(f" {timestring()} - SparseChem classification_metrics start ", verbose = verbose)
        self.batch_data = {}

        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            yc_hat_all = getattr(self, f"{task_key}_pred")

            self.batch_data[task_key] =  {}        
            self.batch_data[task_key]['task_num_class']  = self.tasks_num_class[t_id]
            self.batch_data[task_key]['task_lambda']     = self.opt['lambdas'][t_id]
            self.batch_data[task_key]['yc_ind']          = self.batch[task_key]['ind'].to(self.device, non_blocking=True)
            self.batch_data[task_key]['yc_data']         = self.batch[task_key]['data'].to(self.device, non_blocking=True)    
            self.batch_data[task_key]['yc_hat']          = yc_hat_all[ self.batch_data[task_key]['yc_ind'][0], self.batch_data[task_key]['yc_ind'][1]]
            self.batch_data[task_key]['yc_wghts_sum']    = self.batch[task_key]['trn_weights'][self.batch_data[task_key]['yc_ind'][1]].sum()
            self.batch_data[task_key]['yc_trn_weights']  = self.batch[task_key]['trn_weights'] 
            self.batch_data[task_key]['yc_aggr_weights'] = self.batch[task_key]['aggr_weights']

        # print_heading(f" {timestring()} - SparseChem classification_metrics end ", verbose = verbose)

        return


    def evaluate(self, dataloader,  is_policy= False, num_train_layers=None, hard_sampling=False,
                policy_sampling_mode = 'eval', device = None, disable_tqdm=False, 
                eval_iters=-1, leave = False, verbose = False):

        self.val_metrics = self.initialize_loss_metrics()
        
        # task_sparsity_loss_sum  = {}☺
        task_loss_avg = {}
        task_class_weights  = {}

        all_tasks_class_weights = 0.0

        agg_data = {}
        self.val_data = {}
        num_class_tasks  = dataloader.dataset.class_output_size
        # num_regr_tasks   = dataloader.dataset.regr_output_size
        # class_w = tasks_class.aggregation_weight

        self.eval()

        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            agg_data[task_key] = { "yc_ind":  [],
                                   "yc_data": [],
                                   "yc_hat":  [] }
                
            self.val_data[task_key] = { "yc_ind":  [],
                                   "yc_data": [],
                                   "yc_hat":  [] }
                
            # task_sparsity_loss_sum[task_key] = 0.0
            task_loss_avg[task_key] = 0.0
            task_class_weights[task_key]  = 0.0

        if eval_iters == -1:
            eval_iters = len(dataloader)
            print('eval iters set to : ', eval_iters)

        with torch.no_grad():

            with trange(1,eval_iters+1, total=eval_iters, initial=0, leave=leave, file=sys.stdout,
                         disable=disable_tqdm, position=0, desc = "validation") as t_validation:
                
                for batch_idx in t_validation:
            
                    print_dbg(f"\n + Validation Loop Start - batch_idx:{batch_idx}  eval_iters: {eval_iters}"
                              f"  t_validation: {len(t_validation)} \n",verbose=verbose)

                    batch = next(dataloader)
                    self.set_inputs(batch, dataloader.dataset.input_size)  

                    self.validate(is_policy, 
                                  num_train_layers, 
                                  hard_sampling = hard_sampling, 
                                  policy_sampling_mode=policy_sampling_mode, verbose = verbose)            
                    
                    self.val_metrics['task']['total']      += self.losses['task']['total']
                    self.val_metrics['task_mean']['total'] += self.losses['task_mean']['total'] 
                    self.val_metrics['sharing']['total']   += self.losses['sharing']['total']
                    self.val_metrics['sparsity']['total']  += self.losses['sparsity']['total'] 
                    
                    for t_id, _ in enumerate(self.tasks):
                        task_key = f"task{t_id+1}"

                        self.val_metrics['task'][task_key]      += self.losses['task'][task_key]  
                        self.val_metrics['task_mean'][task_key] += self.losses['task_mean'][task_key] 
                        self.val_metrics['sparsity'][task_key]  += self.losses['sparsity'][task_key]

                        task_class_weights[task_key]     += self.batch_data[task_key]['yc_wghts_sum']
                        
                        ## 
                        agg_data[task_key]['yc_trn_weights']  = self.batch_data[task_key]['yc_trn_weights']
                        agg_data[task_key]['yc_aggr_weights'] = self.batch_data[task_key]['yc_aggr_weights']

                        agg_data[task_key]['yc_ind'].append(self.batch_data[task_key]["yc_ind"].cpu())            
                        agg_data[task_key]['yc_data'].append(self.batch_data[task_key]["yc_data"].cpu())            
                        agg_data[task_key]['yc_hat'].append(self.batch_data[task_key]["yc_hat"].cpu())            

                        ## storing Y data for EACH TASK (for metric computations)
                        # for key in ["yc_ind", "yc_data", "yc_hat"]:
                            # if (key in self.metrics[task_key]) and (self.metrics[task_key][key] is not None):
                                # data[task_key][key].append(self.metrics[task_key][key].cpu())            
                    
                    t_validation.set_postfix({'it': batch_idx, 
                                              'Lss': f"{self.losses['task']['total'].item():.4f}" , 
                                              'Spr': f"{self.losses['sparsity']['total'].item():.4e}" ,  
                                              'Shr': f"{self.losses['sharing']['total'].item():.4e}",
                                              'lyr': f"{num_train_layers}"})     
                    
                    #                 'loss'   : f"{all_tasks_loss_sum.item()/batch_idx:.4f}" ,
                    #                 'row_ids': f"{batch['row_id'][0]} - {batch['row_id'][-1]}" ,
                    #                 'task_wght': f"{self.metrics[task_key]['yc_wghts_sum']}"})

                    if verbose:                
                        for t_id, _ in enumerate(self.tasks):
                            task_key = f"task{t_id+1}"
                            
                            print_dbg(f"\n + Validation of one task complete - BATCH:{batch_idx} TASK: {t_id+1}  \n"
                                      f"    loss[task{t_id+1}]     : {self.losses['task'][task_key]:.4f}"
                                      f"    sum(err)               : {self.val_metrics['task'][task_key]:.4f}"
                                      f"    sum(err)/batch_id      : {self.val_metrics['task'][task_key]/batch_idx:.4f} \n"
                                      f"    loss_mean[task{t_id+1}]: {self.losses['task_mean'][task_key]:.4f}"
                                      f"    sum(err_mean)          : {self.val_metrics['task_mean'][task_key]:.4f}"
                                      f"    sum(err_mean)/batch_id : {self.val_metrics['task_mean'][task_key]/batch_idx:.4f} \n" 
                                      f"    task.[yc_wghts_sum]    : {self.batch_data[task_key]['yc_wghts_sum']:.4f}  "
                                      f"    task_weights[task_key] : {task_class_weights[task_key]}  ", verbose = True)
                        

            ##-----------------------------------------------------------------------
            ## All Validation batches have been feed to network - calcualte metrics
            ##-----------------------------------------------------------------------
            if verbose:
                print_heading(f" + Validation Loops complete- batch_idx:{batch_idx}  eval_iters: {eval_iters}", verbose = True)
                for t_id, _ in enumerate(self.tasks):
                    task_key = f"task{t_id+1}"
                    print_dbg(f"    task: {t_id+1:3d}     batch_idx       : {batch_idx} \n"     
                              f"    sum(err)                              : {self.val_metrics['task']['total']:8.4f} "
                              f"    sum(err)/batch_idx                    : {self.val_metrics['task']['total']/batch_idx:8.4f}\n"
                              f"    sum(err_mean)                         : {self.val_metrics['task_mean']['total']:8.4f}  "
                              f"    sum(err_mean)/batch_idx               : {self.val_metrics['task_mean']['total']/batch_idx:8.4f} \n"
                              f"    sum(sparsity loss)                    : {self.val_metrics['sparsity']['total']:.4e}\n "
                              f"    sum(sharing loss)                     : {self.val_metrics['sharing']['total']:.4e} \n"
                              f"    self.metrics[task_key][yc_wghts_sum]  : {self.batch_data[task_key]['yc_wghts_sum']}\n"
                              f"    task_weights[task_key]                : {task_class_weights[task_key]} \n"
                              f"    task yc_aggr_weight                   : {agg_data[task_key]['yc_aggr_weights']}\n" 
                              f"    task yc_trn_weight                    : {agg_data[task_key]['yc_trn_weights']}\n" 
                              f"    all_tasks_class_weights               : {all_tasks_class_weights}\n"
                              , verbose = True)
            

            assert batch_idx == eval_iters , f"Error - batch_idx {batch_idx} doesn't match eval_iters {eval_iters}"
    
            self.val_metrics["task"]['total']      = (self.val_metrics['task']['total'] / batch_idx).item()
            self.val_metrics["task_mean"]['total'] = (self.val_metrics['task_mean']['total']/ batch_idx).item()
            self.val_metrics["sparsity"]['total']  = (self.val_metrics['sparsity']['total']/ batch_idx).item()
            self.val_metrics["sharing"]['total']   = (self.val_metrics['sharing']['total']/ batch_idx).item()

            self.val_metrics["total"]      = {"total": self.val_metrics["task"]["total"]  +
                                                        self.val_metrics["sparsity"]["total"] +
                                                        self.val_metrics["sharing"]["total"]  ,

                                             "total_mean": self.val_metrics["task_mean"]["total"]  +   
                                                           self.val_metrics["sparsity"]["total"] +
                                                           self.val_metrics["sharing"]["total"],

                                             "task"      : self.val_metrics['task']['total'],

                                             "policy"    : self.val_metrics['sparsity']['total'] +
                                                           self.val_metrics['sharing']['total'] 
                                             }


            all_tasks_classification_metrics = []
            all_tasks_aggregation_weights    = [] 

            for t_id, task in enumerate(self.tasks):
                task_key = f"task{t_id+1}"

                self.val_metrics[task_key] = {}
                self.val_metrics["task"][task_key]      = (self.val_metrics["task"][task_key]      / batch_idx).item()
                self.val_metrics["task_mean"][task_key] = (self.val_metrics["task_mean"][task_key] / batch_idx ).item()
                self.val_metrics["sparsity"][task_key]  = (self.val_metrics["sparsity"][task_key]  / batch_idx).item()

                # print(f" batch_idx                        {type(batch_idx)}")
                # print_dbg(f" yc_ind shape:          {len(agg_data[task_key]['yc_ind'])  }", verbose = verbose)
                # print_dbg(f" yc_ind shape:          {agg_data[task_key]['yc_ind'][0].shape  }", verbose = verbose)
                # print_dbg(f" yc_hat shape:          {len(agg_data[task_key]['yc_data'])}", verbose = verbose)
                # print_dbg(f" yc_hat shape:          {agg_data[task_key]['yc_data'][0].shape}", verbose = verbose)
                # print_dbg(f" yc_aggr_weights:       {agg_data[task_key]['yc_aggr_weights'].shape}", verbose = verbose)


                
                self.val_data[task_key]['yc_ind']  = torch.cat(agg_data[task_key]['yc_ind'] , dim=1).numpy()
                self.val_data[task_key]['yc_data'] = torch.cat(agg_data[task_key]['yc_data'], dim=0).numpy()
                self.val_data[task_key]['yc_hat']  = torch.cat(agg_data[task_key]['yc_hat'] , dim=0).numpy()
                self.val_data[task_key]['yc_aggr_weights'] = agg_data[task_key]["yc_aggr_weights"]

                # print_dbg(f" after concatenation:", verbose = verbose)
                # print_dbg(f" yc_ind shape:          {self.val_data[task_key]['yc_ind'][:, :20]}", verbose = verbose)
                # print_dbg(f" yc_ind shape:          {self.val_data[task_key]['yc_ind'].shape}", verbose = verbose)
                # print_dbg(f" yc_true shape:         {self.val_data[task_key]['yc_data'][:20]}", verbose = verbose)
                # print_dbg(f" yc_true shape:         {self.val_data[task_key]['yc_data'].shape}  {self.val_data[task_key]['yc_data'].sum()}", verbose = verbose)
                # print_dbg(f" yc_hat shape:          {self.val_data[task_key]['yc_hat'][:20]}", verbose = verbose)
                # print_dbg(f" yc_hat shape:          {self.val_data[task_key]['yc_hat'].shape}   {self.val_data[task_key]['yc_hat'].sum()}", verbose = verbose)
                # print_dbg(f" yc_aggr_weights:       {self.val_data[task_key]['yc_aggr_weights'].shape}", verbose = verbose)

                self.val_metrics[task_key]["classification"] = compute_metrics(cols=self.val_data[task_key]['yc_ind'][1], 
                                                                               y_true=self.val_data[task_key]['yc_data'], 
                                                                               y_score=self.val_data[task_key]['yc_hat'] ,
                                                                               num_tasks=num_class_tasks[t_id])
                # print(self.val_metrics[task_key]["classification"])
                ## to_dict(): convert pandas series to dict to make it compatible with print_loss()
                self.val_metrics[task_key]["classification_agg"] = aggregate_results(self.val_metrics[task_key]["classification"], 
                                                                                    weights = self.val_data[task_key]['yc_aggr_weights']).to_dict() 
                # print(self.val_metrics[task_key]["classification_agg"])
                self.val_metrics[task_key]['classification_agg']['sc_loss'] = self.val_metrics["task"][task_key] / batch_idx
                self.val_metrics[task_key]["classification_agg"]["logloss"] = self.val_metrics["task"][task_key] / task_class_weights[task_key].cpu().item()

                # print(f" self.val_metrics['task'][{task_key}]                         :   {self.val_metrics['task'][task_key]} \n"
                #       f" self.val_metrics[{task_key}]['classification_agg']['logloss']:   {self.val_metrics[task_key]['classification_agg']['logloss']} \n"
                #       f" task_class_weights[{task_key}]                               :   {task_class_weights[task_key].cpu().item()} \n"
                #       f" self.val_metrics[{task_key}]['classification_agg']['sc_loss']:   {self.val_metrics[task_key]['classification_agg']['sc_loss']} \n"
                #       ) 

                all_tasks_classification_metrics.append(self.val_metrics[task_key]['classification'])
                all_tasks_aggregation_weights.append(self.val_data[task_key]['yc_aggr_weights'])
            
                all_tasks_class_weights += task_class_weights[task_key].cpu().item()

                # self.val_metrics[task_key]['classification_agg']['yc_weights_sum']  = self.metrics[task_key]["yc_wghts_sum"].cpu().item()
    
            ## Calculate aggregated metrics across all task groups
            
            self.val_metrics['aggregated'] = aggregate_results( pd.concat(all_tasks_classification_metrics),
                                                                weights = np.concatenate(all_tasks_aggregation_weights)).to_dict()
            self.val_metrics['aggregated']['sc_loss'] = (self.val_metrics['task']['total'] / eval_iters ) 
            self.val_metrics['aggregated']["logloss"] = (self.val_metrics['task']['total'] / all_tasks_class_weights)
    
            self.train()
            return self.val_metrics


    def decay_temperature(self, decay_ratio=None, verbose = False):
        tmp = self.gumbel_temperature
        if decay_ratio is None:
            self.gumbel_temperature *= self.temperature_decay
        else:
            self.gumbel_temperature *= decay_ratio
        # print_dbg(f"Change temperature from {tmp:.5f} to{self.gumbel_temperature:.5f}", verbose = verbose )


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
        policies = []

        ## Returns polcies_tensor as a list of tensors, logits as a list of ndarrays
        policies_tensor, logits  = self.networks['mtl-net'].test_sample_policy(hard_sampling)
        
        for policy in policies_tensor:
            policies.append(policy.cpu().numpy())
            
        return policies, logits
            

    # def sample_policy(self, hard_sampling):
    def set_sample_policy(self, hard_sampling, verbose = False):
        '''
        save list of policys as policy attributes 
        if hard_sampling == True  :    Network Task Logits --> Argmax
        if hard_sampling == False :    Network Task Logits --> Softmax --> random.choice( (1,0), P = softmax)
        '''
        ## Returns polcies_tensor as a list of tensors, logits as a list of ndarrays
        policies_tensor, logits  = self.networks['mtl-net'].test_sample_policy(hard_sampling)

        for t_id in range(len(policies_tensor)):
            setattr(self, 'policy%d' % (t_id+1), policies_tensor[t_id])         
            setattr(self, 'logit%d' % (t_id+1), logits[t_id])
            if verbose:
                print(f"\n task: {t_id+1}  logits              policys")
                # with np.printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan', precision=7, formatter={'float': lambda x: f"{x:12.5e}"}):                
                for l,p in zip(logits[t_id], policies_tensor[t_id].detach().cpu().numpy()):
                    print(f'   {l[0]:8.4f}  {l[1]:8.4f}       {p[0]:f}  {p[1]:f}')
        return policies_tensor,logits

    sample_policy = set_sample_policy
    
    def get_policy_prob(self, verbose = False):
        '''
        dervive Network policy from current task logits

        Network -> Logits -> Softmax
        '''

        if self.opt['policy_model'] == 'task-specific':
            logits = self.get_policy_logits()
            policies = softmax(logits, axis= -1)
            # for t_id in range(self.num_tasks):
            #     task_attr = f"task{t_id+1}_logits"
            #     if isinstance(self.networks['mtl-net'], nn.DataParallel):
            #         logits = getattr(self.networks['mtl-net'].module, task_attr).detach().cpu().numpy()
            #     else:
            #         logits = getattr(self.networks['mtl-net'], task_attr).detach().cpu().numpy()

            #     print_underline(f" task {t_id+1} logits: ",verbose = verbose)
            #     print_dbg(f"{logits}", verbose = verbose)
            #     distributions.append(softmax(logits, axis=-1))

        elif self.opt['policy_model'] == 'instance-specific':
            policies = []
            for t_id in range(self.num_tasks):
                task_attr = f"task{t_id+1}_logits"
                logit = getattr(self, task_attr).detach().cpu().numpy()
                policies.append(logit.mean(axis=0))
        else:
            raise ValueError('policy mode = %s is not supported' % self.opt['policy_model']  )

        return policies


    def get_policy_logits(self, verbose = False):
        '''
        Retrieve  Network task logits

        task_specific    : Network['mtl-net'] -> Logits -> Softmax
        instance_specific: Network['mtl-net'] -> task_logits --> mean()
        '''
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

        else:
            raise ValueError('policy mode = %s is not supported' % self.opt['policy_model']  )

        return logits_list


    def get_task_logits(self, task_id):

        task_key = f"task{task_id + 1}_logits" 
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            logits = getattr(self.networks['mtl-net'].module, task_key )
        else:
            logits = getattr(self.networks['mtl-net'], task_key)
        # print_dbg(f" networks[mtl-net] logits task {task_id+1} : {self.networks['mtl-net'].task1_logits}", verbose= True)
        # print_dbg(f" logits task {task_id+1} : {logits}", verbose= True)
        return logits


    def get_current_logits(self):
        logits = []
        for t_id in range(self.num_tasks):
            logit = getattr(self, 'logit%d' % (t_id + 1))
            # logit = logit.detach().cpu().numpy()
            logits.append(logit)

        return logits


    def get_current_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.detach().cpu().numpy()
            policys.append(policy)

        return policys


    def get_current_state(self, current_iter, current_epoch = 'unknown'):
        # print(f' sparsechem_env-dev get_current_state()   {current_iter}    {current_epoch}')
        current_state = super(SparseChemEnv, self).get_current_state(current_iter, current_epoch)
        current_state['temp'] = self.gumbel_temperature

        return current_state


    def load_snapshot(self, snapshot, verbose = False):
        super(SparseChemEnv, self).load_snapshot(snapshot, verbose)
        self.gumbel_temperature = snapshot['temp']

        return snapshot['iter'], snapshot['epoch']


    def fix_weights(self):
        # print_heading(f" Fix Weights - disable gradient flow through main computation graph     ")

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
        # print_heading(f" Free Weights - allow gradient flow through the main computation graph ")
        
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
        # print_heading(f" Fix Alpha - disable gradient flow through alpha computation graph  (policy network)")

        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = False


    def free_alpha(self):
        """Fix architecture parameters - allow gradient flow through alpha computation graph"""
        # print_heading(f" Free Alpha - allow gradient flow through alpha computation graph  (policy network)")

        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = True


    def cuda(self, gpu_ids):
        super(SparseChemEnv, self).cuda(gpu_ids)
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
        super(SparseChemEnv, self).cpu()
        print(f'sparsechem_env.cpu()')
        policys = []

        for t_id in range(self.num_tasks):
            if not hasattr(self, 'policy%d' % (t_id+1)):
                return
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.to(self.device)
            policys.append(policy)

        ## nn.DataParallel only applies to GPU configurations
        
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            setattr(self.networks['mtl-net'].module, 'policys', policys)
        else:
            setattr(self.networks['mtl-net'], 'policys', policys)
        
        print(f'environ cpu policy: {policys}')


    @property
    def name(self):
        return 'SparseChemEnv'

    def set_loss_normalizer(self):
        if self.norm_loss is None:
            self.loss_normalizer = self.batch['batch_size']
        else:
            self.loss_normalizer = self.norm_loss        

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


#    def gather_batch_data_old(self, verbose= None ):
#        """
#        Gather batch data required for calculation of validation metrics for current batch 
#        """
#        if verbose is None:
#            verbose = self.verbose
#
#        print_heading(f" {timestring()} - SparseChem classification_metrics start ", verbose = verbose)
#        self.batch_data = {}
#
#        for t_id, task in enumerate(self.tasks):
#            task_key = f"task{t_id+1}"
#            task_pred    = task_key+"_pred"
#            task_ind     = task_key+"_ind"
#            task_data    = task_key+"_data"
#            task_weights = task_key+"_weights"
# 
#            yc_hat_all = getattr(self, task_pred)
#
#            self.batch_data[task_key] =  {}        
#            self.batch_data[task_key]['task_num_class']  = self.tasks_num_class[t_id]
#            self.batch_data[task_key]['task_lambda']     = self.opt['lambdas'][t_id]
#            self.batch_data[task_key]['yc_ind']          = self.batch[task_ind].to(self.device, non_blocking=True)
#            self.batch_data[task_key]['yc_data']         = self.batch[task_data].to(self.device, non_blocking=True)    
#            self.batch_data[task_key]['yc_hat']          = yc_hat_all[self.batch_data[task_key]['yc_ind'][0], self.batch_data[task_key]['yc_ind'][1]]
#
#            ## batch_size x number of populated outputs in the yc batch , indiccated by yc_ind. 
#            self.batch_data[task_key]['yc_wghts_sum']    = self.batch[task_weights].training_weight[self.batch_data[task_key]['yc_ind'][1]].sum()
#            
#            ## weights of each indivudal class within a classification task
#            self.batch_data[task_key]['yc_trn_weights']  = ( self.batch[task_weights].training_weight).to(self.device, non_blocking=True)  
#            
#            ## weights of each indivudal class within a regression task
#            self.batch_data[task_key]['yc_aggr_weights'] = ( self.batch[task_weights].aggregation_weight)
#
#        print_heading(f" {timestring()} - SparseChem classification_metrics end ", verbose = verbose)
#
#        return
#


# def val_fix_policy(self, num_train_layers=None, policy_sampling_mode = 'fix_policy'):
#
#     self.forward(num_train_layers, policy_sampling_mode = policy_sampling_mode)
#     # self.forward_fix_policy(num_train_layers)
#     # self.resize_results()
#
#     # self.get_classification_loss()
#
#     return  
    

# def forward_fix_policy(self, is_policy = True, num_train_layers = None, hard_sampling = False, verbose = None):
#
#     if verbose is None:
#         verbose = self.verbose
#
#     print_heading(f"forward_fix_policy - policy model: {self.opt['policy_model']}  ", verbose = verbose)
#
#     outputs, _, _ = self.networks['mtl-net'](self.input,
#                                             self.gumbel_temperature, 
#                                             is_policy,  
#                                             num_train_layers=num_train_layers, 
#                                             hard_sampling = hard_sampling , 
#                                             policy_sampling_mode='fix_policy')
#
#
#     for t_id, task in enumerate(self.tasks):
#         print_dbg(f" set attributes: task id {t_id+1}  task: {task}", verbose = verbose)
#         print_dbg(f"    output[{t_id+1}]:      {outputs[t_id].shape}", verbose = verbose)
#         # print_dbg(f"    policy{t_id+1}:        {policys[t_id]}", verbose = verbose)
#         # print_dbg(f"    logits{t_id+1}:        {logits[t_id]} ", verbose = verbose)
#         print_dbg(f"    task{t_id+1}_logits:   {getattr(self.networks['mtl-net'], f'task{t_id+1}_logits')}", verbose = verbose)
#
#         # setattr(self, '%s_pred' % task, outputs[t_id])
#         setattr(self, 'task%d_pred' % (t_id+1), outputs[t_id])

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
#     outputs, policys, logits = self.networks['mtl-net'](input            = self.input, 
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

#-----------------------------------------------------------------------------------------------------------------------
#  Save Best Checkpoint Code (saved below and in sparsechem_env_dev.py)
#-----------------------------------------------------------------------------------------------------------------------
#
#            #----------------------------------------------------------------------------------------------
#            # if number of iterations completed after the warm up phase is greater than the number of 
#            # (weight/policy alternations) x (cirriculum speed) x (number of layers to be policy trained)
#            #
#            # check metrics for improvement, and issue a checkpoint if necessary
#            #----------------------------------------------------------------------------------------------
# 
#             if current_iter - opt['train']['warm_up_iters'] >= num_blocks * opt['curriculum_speed'] * \
#                     (opt['train']['weight_iter_alternate'] + opt['train']['alpha_iter_alternate']):
#                 new_value = 0
#                 print(f"  {current_iter - opt['train']['warm_up_iters']} IS GREATER THAN "
#                        f" {num_blocks * opt['curriculum_speed'] * (opt['train']['weight_iter_alternate'] + opt['train']['alpha_iter_alternate'])} -- "
#                        f"  evaluate progress and make checkpoint if necessary." )            
# 
#                 ## compare validation metrics against reference metrics.
#                 
#                 for k in refer_metrics.keys():
#                     if k in val_metrics.keys():
#                         for kk in val_metrics[k].keys():
#                             if not kk in refer_metrics[k].keys():
#                                 continue
#                             if (k == 'sn' and kk in ['Angle Mean', 'Angle Median']) or (
#                                     k == 'depth' and not kk.startswith('sigma')) or (kk == 'err'):
#                                 value = refer_metrics[k][kk] / val_metrics[k][kk]
#                             else:
#                                 value = val_metrics[k][kk] / refer_metrics[k][kk]
#                             value = value / len(list(set(val_metrics[k].keys()) & set(refer_metrics[k].keys())))
#                             new_value += value
# 
#                 print('Best Value %.4f  New value: %.4f' % new_value)
# 
#                 ## if results have improved, save these results and issue a checkpoint
# 
#                 if (new_value > best_value):
#                     print('Previous best iter: %d, best_value: %.4f' % (best_iter, best_value), best_metrics)
#                     best_value = new_value
#                     best_metrics = val_metrics
#                     best_iter = current_iter
#                     environ.save_checkpoint('best', current_iter)
#                     print('New      best iter: %d, best_value: %.4f' % (best_iter, best_value), best_metrics)                         
#                     print('Best Value %.4f  New value: %.4f' % new_value)
#
#-----------------------------------------------------------------------------------------------------------------------