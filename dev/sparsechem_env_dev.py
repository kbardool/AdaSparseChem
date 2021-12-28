import os
import pickle
from scipy.special import softmax
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from models.base import Bottleneck, BasicBlock
from dev.MTL3_Dev import MTL3_Dev
from dev.MTL_Instance_Dev import MTL_Instance_Dev
from dev.sc_model_dev import censored_mse_loss, censored_mae_loss
from dev.base_env_dev import BaseEnv
from dev.sparsechem_models import SparseChemBlock
from utils.util import timestring, print_heading, print_dbg, print_underline, debug_on, debug_off
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
            layers = [1, 1, 1, 1]   ## This is used by blockdrop_env to indicate the number of resnet blocks in each layer. 

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
        self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        
        if self.dataset == 'Chembl_23_mini':
            self.loss_class = torch.nn.BCEWithLogitsLoss(reduction="none")
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
                                                   momentum=0.9, weight_decay=1e-4)
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
                ## NOT select the layer. Meaning for logits [p1 , p2] where p1: layer is selected,
                ## and p2: layer NOT being selected, the Ground Truth gt is [1] 
                ##---------------------------------------------------------------------------------------
                gt = torch.ones((num_blocks)).long().to(self.device)

                if self.opt['diff_sparsity_weights'] and not self.opt['is_sharing']:

                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                    self.losses['sparsity'][task_key] = 2 * (loss_weights[-num_blocks:] * self.cross_entropy2(logits[-num_blocks:], gt)).mean()
                   
                    print_dbg(f" loss_weights :  {loss_weights}", verbose = verbose)
                    print_dbg(f" cross_entropy:  {self.cross_entropy2(logits[-num_blocks:], gt)}   ", verbose = verbose)
                    print_dbg(f" loss[sparsity][{task_key}]: {self.losses['sparsity'][task_key] } ", verbose = verbose)
               
                else:
                    print_dbg(f" Compute CrossEntropyLoss between \n\t Logits   : {logits[-num_blocks:]} \n\t and gt: {gt} \n", verbose = verbose)
                    self.losses['sparsity'][task_key] = self.cross_entropy_sparsity(logits[-num_blocks:], gt)
                
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
        
        self.losses['hamming'] = {}
        self.losses['hamming']['total'] = 0

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
            print_underline(f" {task_i_attr} Hamming loss: ",verbose=verbose)

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
                self.losses['hamming']['total'] +=  task_i_j_loss
                
                print_underline(f" between {task_i_attr} and {task_j_attr} : ",verbose=verbose)
                print_dbg(f" {task_i_attr:12s}: {logits_i[:,0]}  "
                          f"\n {task_j_attr:12s}: {logits_j[:,0]}  "
                          f"\n abs diff    : {(torch.abs(logits_i[:, 0] - logits_j[:, 0]))} "
                          f"\n loss_weights: {loss_weights}"
                          f"\n\t sum       : {torch.sum(torch.abs(logits_i[:, 0] - logits_j[:, 0])):.5f} "
                          f"\n\t weighted  : {torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0])):.5f} ", verbose = verbose)

            print_underline(f" Total hamming loss for {task_i_attr} :  {task_i_loss} ", verbose=verbose)


        print_underline(f" Total hamming loss for ALL TASKS  :  {self.losses['hamming']['total']:.5f} ", verbose=verbose)
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
        
        self.losses = {}
        self.task_lambdas = lambdas

        # self.losses['total']['total']    = torch.tensor(0.0, device  = self.device, dtype=torch.float64)
        # self.losses['sparsity']['total'] = torch.tensor(0.0, device  = self.device, dtype=torch.float64)
        # self.losses['hamming']['total']  = torch.tensor(0.0, device  = self.device, dtype=torch.float64)
        # self.losses['tasks']['total']    = torch.tensor(0.0, device  = self.device, dtype=torch.float64)        
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



    # def val2_old(self, is_policy, num_train_layers=None, hard_sampling=False):
        # metrics = {}
       
        # if is_policy:
        #     self.forward_eval(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)
        # else:
        #     self.forward(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)
        # self.resize_results()
       
        # if 'seg' in self.tasks:
        #     metrics['seg'] = {}
        #     seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
        #     pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
        #     metrics['seg']['pred'] = pred
        #     metrics['seg']['gt'] = gt
        #     metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
        #     metrics['seg']['err'] = err                            
        # return metrics


    def val2(self, is_policy, num_train_layers=None, hard_sampling=False, verbose = None):
        
        self.losses = {}

        if is_policy:
            self.forward_eval(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)
        else:
            self.forward(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)
        # self.resize_results()
        
        self.compute_task_losses()
        # pp.pprint(self.losses)

        self.classification_metrics()
        # pp.pprint(self.metrics)
        return


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

 
    def forward(self, is_policy, num_train_layers = None, hard_sampling = False, verbose = None):
        if verbose is None:
            verbose = self.verbose

        if verbose: 
            print_heading(f" {timestring()} - SparseChem network FORWARD() start ", verbose = True)
            print_dbg(f"\t num_train_layers:{num_train_layers}   is_policy: {is_policy}    hard_sampling: {hard_sampling}", verbose = True)

        outputs, policys, logits = self.networks['mtl-net'](img              = self.input, 
                                                            temperature      = self.temp, 
                                                            is_policy        = is_policy, 
                                                            num_train_layers = num_train_layers, 
                                                            hard_sampling    = hard_sampling,
                                                            mode             = 'train')
        # import pdb
        # pdb.set_trace()
        
        for t_id,  task in enumerate(self.tasks):
            print_dbg(f" set attributes: task id {t_id+1}  task: {task}", verbose = verbose)
            print_dbg(f"    output[{t_id+1}]:  {outputs[t_id].shape}", verbose = verbose)
            print_dbg(f"    policy{t_id+1}:    {policys[t_id]}", verbose = verbose)
            print_dbg(f"    logits{t_id+1}:    {logits[t_id]} ", verbose = verbose)
            print_dbg(f"    task{t_id+1}_logits:   {getattr(self.networks['mtl-net'], f'task{t_id+1}_logits')}", verbose = verbose)

            setattr(self, 'task%d_pred' % (t_id+1), outputs[t_id])
            setattr(self, 'policy%d' % (t_id+1), policys[t_id])
            setattr(self, 'logit%d' % (t_id+1), logits[t_id])
    
        print_heading(f" {timestring()} - SparseChem network FORWARD() end ", verbose = verbose)
        return 


    def forward_eval(self, is_policy,  num_train_layers = None, hard_sampling = False, verbose = None):
        '''
        Multi-task network called with mode == 'eval'
                mode:   specifies the policy sampling method 
                mode == 'eval'  AND  hard_sampling == False:  Uses softmax on task logits to generate sample policies 
        '''
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print_heading(f" {timestring()} - SparseChem network FORWARD_EVAL() start ")        
            print_dbg(" num_train_layers:{num_train_layers}   is_policy: {is_policy}    hard_sampling: {hard_sampling}", verbose = verbose)

        outputs, policys, logits = self.networks['mtl-net'](img              = self.input, 
                                                            temperature      = self.temp, 
                                                            is_policy        = is_policy, 
                                                            num_train_layers = num_train_layers, 
                                                            hard_sampling    = hard_sampling,  
                                                            mode             = 'eval')  

        for t_id, task in enumerate(self.tasks):
            setattr(self, 'task%d_pred' % (t_id + 1), outputs[t_id])
            setattr(self, 'policy%d' % (t_id + 1), policys[t_id])
            setattr(self, 'logit%d' % (t_id + 1), logits[t_id])

            if verbose:
                print_underline(f" MTL-net output: task id {t_id+1}  task: {task}", verbose = True)
                print_dbg(f"    output[{t_id+1}]:  {outputs[t_id]}.shape", verbose = True)
                print_dbg(f"    policy{t_id+1}  :    {policys[t_id]}", verbose = True)
                print_dbg(f"    logits{t_id+1}  :    {logits[t_id]} ", verbose = True)
                print_dbg(f"    task{t_id+1}_logits:   {getattr(self.networks['mtl-net'], f'task{t_id+1}_logits')}   ", verbose = True)


        print_heading(f" {timestring()} - SparseChem network FORWARD_EVAL() end ", verbose = verbose)


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
        self.losses['tasks'] = {}
        self.losses['total'] = {}

        total_loss = 0
        total_tasks_loss =  0 

        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            # task_loss = self.task_lambdas[t_id] * self.losses[task_key]['total']
            # print_dbg(f"\t backward_policy - {task_key}   loss: {self.task_lambdas[t_id]} * {self.losses[task_key]['total']:.4f}"
                #   f" = {task_loss:.4f}" , verbose = verbose )            
            # self.losses['tasks'][task_key] = task_loss
            self.losses['tasks'][task_key] = self.losses[task_key]['total']
            total_tasks_loss += self.losses[task_key]['total']

        self.losses['tasks']['total'] = total_tasks_loss        
        self.losses['total']['total'] = total_tasks_loss

        # print_dbg(f"\t backward Policy - Sum of weighted task losses :  {self.losses['tasks']['total']:.4f} ", verbose=True )

        # for t_id, task in enumerate(self.tasks):
        #     task_key = f"task{t_id+1}"
        #     task_loss = (self.task_lambdas[t_id] * self.losses[task_key]['total']).detach()
        #     print_dbg(f"\t backward_policy - {task_key}   loss: {self.task_lambdas[t_id]} * {self.losses[task_key]['total']:.4f}"
        #           f" = {task_loss:.4f}" , verbose = verbose )            
        #     self.losses['tasks'][task_key] = task_loss
        #     total_tasks_loss += task_loss

        # self.losses['tasks']['total'] = total_tasks_loss

        if self.opt['is_sharing']:
            self.get_hamming_loss()
            sharing_loss = self.opt['train']['Lambda_sharing'] * self.losses['hamming']['total']
            self.losses['hamming']['total']  = sharing_loss

        if self.opt['is_sparse']:
            self.get_sparsity_loss(num_train_layers)
            sparsity_loss = self.opt['train']['Lambda_sparsity'] * self.losses['sparsity']['total']
            self.losses['sparsity']['total'] = sparsity_loss
        
        total_loss = total_tasks_loss + sparsity_loss + sharing_loss
        
        if verbose:
            print(f"\n Total Task losses                           : {total_tasks_loss:.4f}")
            print(f" Sharing loss (Lambda_sharing *  hamming Loss) : {self.opt['train']['Lambda_sharing']} * {self.losses['hamming']['total']:.4f}"
                  f" = {sharing_loss:.5e}")
            print(f" Sparsity loss (Lambda_sharing *  CE Loss)     : {self.opt['train']['Lambda_sparsity']} * {self.losses['sparsity']['total']:.4f}"
                  f" = {sparsity_loss:.5e}")
            print(f"        Total Loss (Task + Sharing + Sparsity) : { total_loss:.4f}")


        self.losses['total']['total']    = total_loss
        self.losses['total']['total'].backward()

        self.optimizers['alphas'].step()
        
        print_heading(f" {timestring()} - SparseChem backward Policy end  " , verbose = verbose )
        return


    def backward_network(self, verbose = None):
        if verbose is None:
            verbose = self.verbose
        
        print_heading(f" {timestring()} - SparseChem backward Network start" , verbose = verbose)
        
        self.optimizers['weights'].zero_grad()

        self.losses['tasks'] = {}
        self.losses['total'] = {}
        
        total_loss = 0
        total_tasks_loss =  0 
        
        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            # task_loss = self.task_lambdas[t_id] * self.losses[task_key]['total']
            # print_dbg(f"\t backward_policy - {task_key}   loss: {self.task_lambdas[t_id]} * {self.losses[task_key]['total']:.4f}"
                #   f" = {task_loss:.4f}" , verbose = verbose )            
            # self.losses['tasks'][task_key] = task_loss
            self.losses['tasks'][task_key] = self.losses[task_key]['total']
            total_tasks_loss += self.losses[task_key]['total']

        self.losses['tasks']['total'] = total_tasks_loss        
        self.losses['total']['total'] = total_tasks_loss
        
        print_dbg(f"\t backward Network - Sum of weighted task losses :  {self.losses['tasks']['total']:.4f} ", verbose=verbose)

        print_dbg(f"\t {timestring()} - backward pass & optimize step begin  ", verbose=verbose )
        
        self.losses['total']['total'].backward()
        self.optimizers['weights'].step()

        print_dbg(f"\t {timestring()} - backward pass & optimize step end  ", verbose=verbose )
        
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


    def sample_policy(self, hard_sampling):
        # dist1, dist2 = self.get_policy_prob()
        # print(np.concatenate((dist1, dist2), axis=-1))
        policys = self.networks['mtl-net'].test_sample_policy(hard_sampling)
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


    def get_current_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            print(f'get policy{t_id+1}')
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.detach().cpu().numpy()
            policys.append(policy)

        return policys

    # ##################### change the state of each module ####################################

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


    def fix_w(self):
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


    def free_w(self, fix_BN):
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

    # ##################### change the state of each module ####################################
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
        print(f'environ.cpu policy: {policys}')


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
        # if torch.cuda.is_available():
        #     self.input = self.input.to(self.device)

        # if 'depth' in self.tasks:
        #     self.depth = batch['depth']
        #     if torch.cuda.is_available():
        #         self.depth = self.depth.to(self.device)
        #     if 'depth_mask' in batch.keys():
        #         self.depth_mask = batch['depth_mask']
        #         if torch.cuda.is_available():
        #             self.depth_mask = self.depth_mask.to(self.device)
        #     if 'depth_policy' in batch.keys():
        #         self.depth_policy = batch['depth_policy']
        #         if torch.cuda.is_available():
        #             self.depth_policy = self.depth_policy.to(self.device)


    def resize_results(self):
        pass


    def compute_task_losses(self,  instance=False, verbose = None):
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
            # self.losses[task_key][task_ind]  = yc_ind
            # self.losses[task_key][task_data] = yc_data
            
            yc_hat_all = getattr(self, task_pred)
            yc_hat  = yc_hat_all[yc_ind[0], yc_ind[1]]

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

            if self.norm_loss is None:
                norm = self.batch['batch_size']
            else:
                norm = self.norm_loss

            task_loss  = (self.loss_class(yc_hat, yc_data) ).sum()
            task_loss /= norm

            self.losses[task_key]['total'] = self.task_lambdas[t_id] * task_loss
            
            # print_dbg(f"\t compute task losses - {task_key}   loss: {self.task_lambdas[t_id]} * {task_loss:.4f}"
            #       f" = {self.losses[task_key]['total'] :.4f}" , verbose = True )   



            # print(f"\t Loss for task id {t_id+1} before normalization: {self.losses[task_key]['total']}    norm: {norm}")            
            # self.losses['tasks']['total']+= self.losses[task_key]['total'] 

            # print(f"\t Loss for task id {t_id+1} after  normalization: {self.losses[task_key]['total']} ")            
            # print('\n\n')
            
            # loss_wo_reduction = torch.nn.BCEWithLogitsLoss(reduction="none")
            # loss_sum_reduction = torch.nn.BCEWithLogitsLoss(reduction="sum")
            # loss_mean_reduction = torch.nn.BCEWithLogitsLoss(reduction="mean")
            # print(f"\t Loss wo reduction   {t_id+1} : {loss_wo_reduction(yc_hat, yc_data)} ")            
            # print(f"\t Loss sum reduction  {t_id+1} : {loss_sum_reduction(yc_hat, yc_data)} ")            
            # print(f"\t Loss mean reduction {t_id+1} : {loss_mean_reduction(yc_hat, yc_data)} ")            
            
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
            self.metrics[task_key]['err_lambda']      = self.opt['lambdas'][t_id]
            self.metrics[task_key]['err']             = self.losses[task_key]['total'] 
            self.metrics[task_key]['yc_ind']          = self.batch[task_ind].to(self.device, non_blocking=True)
            self.metrics[task_key]['yc_data']         = self.batch[task_data].to(self.device, non_blocking=True)    
            self.metrics[task_key]['yc_wghts_sum']    = self.batch[task_weights].training_weight[self.metrics[task_key]['yc_ind'][1]].sum()
            self.metrics[task_key]['yc_trn_weights']  = ( self.batch[task_weights].training_weight).to(self.device, non_blocking=True)  
            self.metrics[task_key]['yc_aggr_weights'] = ( self.batch[task_weights].aggregation_weight)
            self.metrics[task_key]['yc_hat']          = yc_hat_all[self.metrics[task_key]['yc_ind'][0], self.metrics[task_key]['yc_ind'][1]]

        print_heading(f" {timestring()} - SparseChem classification_metrics end ", verbose = verbose)

        return

    # def seg_error(self, seg_num_class):
    #     """
    #     Returns:
        
    #     predictions: Predictions (y_hat)
    #     gt:          Ground truth values
    #     pixelAcc:    Pixel accuracy 
    #     err:         Cross Entropy loss
    #     """
    #     gt = self.seg.view(-1)
    #     labels = gt < seg_num_class
    #     gt = gt[labels].int()

    #     logits = self.seg_output.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
    #     logits = logits[labels]
    #     err = self.cross_entropy(logits, gt.long())

    #     prediction = torch.argmax(self.seg_output, dim=1)
    #     prediction = prediction.unsqueeze(1)

    #     # pixel acc
    #     prediction = prediction.view(-1)
    #     prediction = prediction[labels].int()
    #     pixelAcc = (gt == prediction).float().mean()

    #     return prediction.cpu().numpy(), gt.cpu().numpy(), pixelAcc, err.cpu().numpy()