import os
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from torch.utitensorboardX import SummaryWriter
from utils  import print_heading, print_dbg, print_underline, print_loss, timestring
# from data_utils.image_decoder import inv_preprocess, decode_labels2


class BaseEnv():
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_name, tasks_num_class, device=0, is_train=True, opt=None, verbose = None):
        """
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        self.verbose = False if verbose is None else verbose

        print_heading(f"{self.name}.super() init()  Start - verbose: {verbose}", verbose = self.verbose)
        
        self.opt = opt
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.is_train = is_train
        self.tasks_num_class = tasks_num_class
        # self.device = device
        # self.device = 'cpu' if self.opt['cpu'] else 'gpu'
        self.dataset = self.opt['dataload']['dataset']
        self.tasks = self.opt['tasks']
        self.loss_csv_file = os.path.join(self.log_dir, 'loss_seed_%04d.csv'%self.opt['random_seed'])
        self.log_txt_file = os.path.join(self.log_dir, 'log_seed_%04d.txt'%self.opt['random_seed'])
        self.log_file = open(self.log_txt_file, 'a') 
        ## First entry in the hidden layers is the output of the Input layer
        ## so we dont count it in the hidden layer count
        self.layers = [1 for _ in self.opt['hidden_sizes']] 
        self.num_layers = len(self.opt['hidden_sizes'])
        
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            self.device = torch.device("cuda:%d" % device)
        else:
            self.device = "cpu"

        print(' device is ', self.device)

        self.networks   = {}
        self.losses     = {}
        self.optimizers = {}
        self.schedulers = {}
        self.metrics    = {}

        self.define_networks()
        self.define_loss()

        
        if is_train:
            # define optimizer MUST BE DONE AFTER MODEL LOAD
            # self.define_optimizer()
            # self.define_scheduler()
            # define summary writer
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.write_run_info()

        print_heading(f"{self.name}.super() init()  end", verbose = self.verbose)

        return


    # ##################### define networks / optimizers / losses ####################################

    def define_loss(self):
        pass

    def define_networks(self, tasks_num_class):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    def set_inputs(self, batch):
        pass

    def extract_features(self):
        pass

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

    def print_configuration(self, verbose = False):
                #  f"---------------------------------------- \n" \
        config = f" \n\n" \
                 f"{self.name}  Configuration       \n" \
                 f"---------------------------------------- \n" \
                 f"\n"                     \
                 f"----------------\n"     \
                 f"networks       :\n"     \
                 f"----------------\n"     \
                 f" {self.networks}\n\n"   \
                 f"----------------\n"     \
                 f"optimizers     :\n"     \
                 f"----------------\n"     \
                 f" {self.optimizers}\n\n" \
                 f"----------------\n"     \
                 f"schedulers     :\n"     \
                 f"----------------\n"    

        for name, sch in self.schedulers.items():
            for key,val in sch.__dict__.items(): 
                config +=f"{key:30s}: {val} \n"
                #  f"----------------\n"     \
                #  f"losses         :\n"     \
                #  f"----------------\n"     \
                #  f" {self.losses}  \n\n"   \
        return config


    def get_loss_dict(self, verbose = False):
        print_dbg(f"get loss dict from self.losses", verbose)
        loss = {}
        for key in self.losses.keys():
            loss[key] = {}
            for subkey, v in self.losses[key].items():
                print_dbg(f"  key:  {key}   subkey: {subkey}  value: {v:.4f}", verbose)
                if isinstance(v, torch.Tensor):
                    loss[key][subkey] = v.data
                else:
                    loss[key][subkey] = v
        return loss

    def disp_for_excel(self):
        ln = f"""
    folder: {self.opt['exp_folder']}
    layers: {len(self.opt['hidden_sizes'])} {self.opt['hidden_sizes']} 
    
    first dropout          : {self.opt['first_dropout']}
    middle dropout         : {self.opt['middle_dropout']}
    last dropout           : {self.opt['last_dropout']}
    diff_sparsity_weights  : {self.opt['diff_sparsity_weights']}
    skip_layer             : {self.opt['skip_layer']}
    is_curriculum          : {self.opt['is_curriculum']}
    curriculum_speed       : {self.opt['curriculum_speed']}
    
    task_lr                : {self.opt['train']['task_lr']}
    backbone_lr            : {self.opt['train']['backbone_lr']}
    decay_lr_rate          : {self.opt['train']['decay_lr_rate']}
    decay_lr_freq          : {self.opt['train']['decay_lr_freq']}
    
    policy_lr              : {self.opt['train']['policy_lr']}
    policy_decay_lr_rate   : {self.opt['train']['policy_decay_lr_rate']}
    policy_decay_lr_freq   : {self.opt['train']['policy_decay_lr_freq']}
    lambda_sparsity        : {self.opt['train']['lambda_sparsity']}
    lambda_sharing         : {self.opt['train']['lambda_sharing']}
    lambda_tasks           : {self.opt['train']['lambda_tasks']}
    
    Gumbel init_temp       : {self.opt['train']['init_temp']}
    Gumbel decay_temp      : {self.opt['train']['decay_temp']}
    Gumbel decay_temp_freq : {self.opt['train']['decay_temp_freq']}
    Logit init_method      : {self.opt['train']['init_method']}
    Logit init_neg_logits  : {self.opt['train']['init_neg_logits']}
    Logit hard_sampling    : {self.opt['train']['hard_sampling']}
    Warm-up epochs         : {self.opt['train']['warmup_epochs']}
    training epochs        : {self.opt['train']['training_epochs']}
    Data split ratios      : {self.opt['dataload']['x_split_ratios']}
"""
        return ln   



    def write_run_info(self):
        split_rto = self.opt['dataload']['x_split_ratios']
        md = f"""
### Run Information

    Project Name    :    {self.opt['project_name']}  
    Experiment name :    {self.opt['exp_name']}

### Description:


** {self.opt['exp_description']} **
        """
    # Batch Size          :   {self.opt['train']['batch_size']}             
    # Data split ratios   :   {self.opt['dataload']['x_split_ratios']}
    # Hidden layers       :   {len(self.opt['hidden_sizes'])} - {self.opt['hidden_sizes']} 
    # starting task_lr    :   {self.opt['train']['task_lr']} 
    # starting backbone_lr:   {self.opt['train']['backbone_lr']} 
    # starting policy_lr  :   {self.opt['train']['policy_lr']} 
    # LR Decay freq       :   {self.opt['train']['decay_lr_freq']} |
    # LR Decay rate       :   {self.opt['train']['decay_lr_rate']} |
        md += self.disp_for_excel()
        md += f"""

**Hyperparameters**

| param | values |
| ----- | ---------- |
| batch_size | {self.opt['train']['batch_size']} |
| # of hidden layers | {len(self.opt['hidden_sizes'])} |
| layer sizes   | {self.opt['hidden_sizes']} |
| LR tasks:     | {self.opt['train']['task_lr']} |
| LR backbone   | {self.opt['train']['backbone_lr']} |
| LR Decay freq | {self.opt['train']['decay_lr_freq']} |
| LR Decay rate | {self.opt['train']['decay_lr_rate']} |
| Policy LR  | {self.opt['train']['policy_lr']} |
| Policy LR Decay freq | {self.opt['train']['policy_decay_lr_freq']} |
| Policy LR Decay rate | {self.opt['train']['policy_decay_lr_rate']} |
| Lambda Sparsity   | {self.opt['train']['lambda_sparsity']} |
| Lambda Sharing    | {self.opt['train']['lambda_sharing']} |
| Lambda Tasks      | {self.opt['train']['lambda_tasks']} |
| Gumbel Temp initial    | {self.opt['train']['init_temp']} |
| Gumbel Temp decay rate | {self.opt['train']['decay_temp']} |
| Gumbel Temp decay freq | {self.opt['train']['decay_temp_freq']} |

"""
        self.writer.add_text('_General Info_', md, 0)
        return md

    def display_trained_logits(self, epoch=0, out = None):
        if not isinstance(out, list):
            out = [out]

        logits = self.get_policy_logits()
        logits_argmaxs = 1-np.argmax(logits, axis = -1)
        ln = "\n"
        ln += f" ep: {epoch:4d}   "
        # "logits      s         logits       s\n"
        # "----------------  -    ----------------  - \n"

        hdr2 = f" ----- "
        for t_id, _ in enumerate(self.tasks):
            ln   += "logits       s         " 
            hdr2 += "----------------- -    "
        ln = ln + '\n' + hdr2 + '\n'
        for lyr in range(self.num_layers):
            ln += f"{lyr:3d}"
            for tsk in range(self.num_tasks):
                # ln += f"  task {policy_softmaxs[tsk][lyr]} info"
                ln += f"  {logits[tsk][lyr][0]:8.4f}  {logits[tsk][lyr][1]:8.4f}  {logits_argmaxs[tsk][lyr]:1d}"
            ln += '\n'
                     
        
        # for idx, (l1,l2,l3,  p1,p2,p3) in enumerate(zip(logits[0], logits[1], logits[2], 
                                                        # logits_argmaxs[0], logits_argmaxs[1], logits_argmaxs[2]),1):
            # ln += f"{idx:4d}  {l1[0]:8.4f}  {l1[1]:8.4f}  {p1:1d}  {l2[0]:8.4f}  {l2[1]:8.4f}  {p2:1d}  {l3[0]:8.4f}  {l3[1]:8.4f}  {p3:1d}\n"
        # ln += '\n'

        for file in out:
            print(ln, file = file)


    def display_trained_policy(self, epoch=0, out = None):
        if not isinstance(out, list):
            out = [out]

        policy_softmaxs = self.get_policy_prob()
        policy_argmaxs = 1-np.argmax(policy_softmaxs, axis = -1)
        ln = "\n"
        ln += f" ep: {epoch:4d}   "
        # "softmax       s        softmax       s\n"
        ## ----------------- -    ----------------- - \n"

        hdr2 = f" ----- "
        for t_id, _ in enumerate(self.tasks):
            ln   +=  "softmax      s        "
            hdr2 += "----------------- -    " 
        ln = ln + '\n' + hdr2 + '\n'

        for lyr in range(self.num_layers):
            ln += f"{lyr:3d}"
            for tsk in range(self.num_tasks):
                # ln += f"  task {policy_softmaxs[tsk][lyr]} info"
                ln += f"  {policy_softmaxs[tsk][lyr][0]:8.4f}  {policy_softmaxs[tsk][lyr][1]:8.4f}  {policy_argmaxs[tsk][lyr]:1d}"
            # for idx, (l1,l2,l3,  p1,p2,p3) in enumerate(zip(policy_softmaxs[0], policy_softmaxs[1], policy_softmaxs[2], 
                                                        # policy_argmaxs[0], policy_argmaxs[1], policy_argmaxs[2]),1):
            # "  {l2[0]:8.4f}  {l2[1]:8.4f}  {p2:1d}  {l3[0]:8.4f}  {l3[1]:8.4f}  {p3:1d}\n"
            ln += '\n'
        for file in out:
            print(ln, file = file)


    def display_test_sample_policy(self, epoch=0, hard_sampling = False, out = None):
        if not isinstance(out, list):
            out = [out]

        policies, logits = self.get_sample_policy(hard_sampling)
        # policies = 1-np.argmax(logits, axis = -1)
        ln =  f" Sample Policy (Testing mode - hard_sampling: {hard_sampling}) "
        ln += "\n"
        ln += f" epch: {epoch:3d}   logits         sel        logits         sel         logits         sel \n"
        ln += f" -----   ----------------  -----    ----------------  -----    ---------------   ----- \n"
        for idx, (l1,l2,l3,  p1,p2,p3) in enumerate(zip(logits[0], logits[1], logits[2], 
                                                        policies[0], policies[1], policies[2]),1):
            # ln += f"   {idx}      {l1[0]}   {l1[1]}   {p1}   {l2[0]}   {l2[1]}  {p2}   {l3[0]}   {l3[1]}  {p3}\n"
            ln += f"   {idx}    {l1[0]:7.4f}   {l1[1]:7.4f}  {p1}   {l2[0]:7.4f}   {l2[1]:7.4f}  {p2}   {l3[0]:7.4f}   {l3[1]:7.4f}  {p3}\n"
        ln += '\n'
        for file in out:
            print(ln, file = file)
            

    def display_train_sample_policy(self, epoch=0, temp = None, hard_sampling = False, out = None):
        if not isinstance(out, list):
            out = [out]
        if temp is None:
            temp = self.gumbel_temperature

        policies_tensors, logits = self.networks['mtl-net'].train_sample_policy(temp, hard_sampling)
        policies = [p.detach().cpu().numpy() for p in policies_tensors]
        ln = f" Sample Policy (Training mode - hard_sampling: {hard_sampling}) "
        ln += "\n"
        ln += f" epch: {epoch:3d}   logits          gumbel                logits           gumbel               logits             gumbel \n"
        ln += f" -----   ----------------------------------     ----------------------------------     ------------------------------------ \n"
        for idx, (l1,l2,l3,  p1,p2,p3) in enumerate(zip(logits[0], logits[1], logits[2], 
                                                        policies[0], policies[1], policies[2]),1):
            # ln += f"   {idx}      {l1[0]}   {l1[1]}   {p1[0]}   {p1[1]}      {l2[0]}   {l2[1]}  {p2}   {l3[0]}   {l3[1]}  {p3}\n"
            ln += f"   {idx}    {l1[0]:7.4f}  {l1[1]:7.4f}   {p1[0]:7.4f}  {p1[1]:7.4f}    {l2[0]:7.4f}  {l2[1]:7.4f}   {p2[0]:7.4f}  {p2[1]:7.4f}" \
                  f"    {l3[0]:7.4f}   {l3[1]:7.4f}    {p3[0]:7.4f}  {p3[1]:7.4f}\n"
        ln += '\n'
        for file in out:
            print(ln, file = file)

    def display_logit_grads(self, title):
        for name, param in self.networks['mtl-net'].named_parameters():
            if 'task' in name and 'logits' in name:
                grad_sum = param.grad.sum() if param.grad is not None else None
                print(f" {title} -  {name:30s}   ")
                print(f"              grad_sum: {grad_sum:.7f}                     param_sum: {param.sum():.7f}  ")
                for p, g in zip(param, param.grad):
                    print(f"          {g.detach().cpu().numpy()}            {p.detach().cpu().numpy()}")
                print(f"------------------------------------------\n")
        
    # def display_sampled_policy(self, epoch=0, hard_sampling = False, out = None):
    #     if not isinstance(out, list):
    #         out = [out]
    #     policy_softmaxs = self.get_sample_policy(hard_sampling = hard_sampling)
    #     policy_argmaxs = 1-np.argmax(policy_softmaxs, axis = -1)
    #     ln = "\n"
    #     ln += f" {epoch:3d} epochs  softmax        sel        softmax       sel        softmax       sel \n"
    #     ln += f" -----    ---------------   ---     ---------------  ---     ---------------  --- \n"
    #     for idx, (l1,l2,l3,  p1,p2,p3) in enumerate(zip(policy_softmaxs[0], policy_softmaxs[1], policy_softmaxs[2], 
    #                                                     policy_argmaxs[0], policy_argmaxs[1], policy_argmaxs[2]),1):
    #         ln += f"   {idx}      {l1[0]:.4f}   {l1[1]:.4f}   {p1:2d}   {l2[0]:9.4f}   {l2[1]:.4f}  {p2:2d}   {l3[0]:9.4f}   {l3[1]:.4f}  {p3:2d}\n"
    #     ln += '\n'
    #     for file in out:
    #         print(ln, file = file)

                      
    def print_trn_metrics(self, epoch, iter, start_time, loss=None, title='Iteration', 
                    to_tb      = True,  
                    to_csv     = True, 
                    to_display = False, 
                    to_text    = False):
        elapsed_time = time.time() - start_time
        title = f"{title} ep:{epoch}    it:{iter}"
        if to_display:
            print_loss(iter, title, self.loss)
        
        if to_csv:
            self.write_metrics_csv(self.loss_csv_file, epoch, iter, elapsed_time, self.losses )
        
        if to_text:
            self.write_metrics_txt(self.log_file, epoch, iter, elapsed_time, self.losses)

        if to_tb:
            for key in ['parms']:
                if key not in self.losses:
                    continue
                # print(key + ':')
                if isinstance(self.losses[key], dict):
                    for subkey in self.losses[key].keys():
                        self.writer.add_scalar('trn_%s/%s'%(key, subkey), self.losses[key][subkey], iter)
                elif (isinstance(self.losses[key], float)):
                    self.writer.add_scalar('trn_%s'%(key), self.losses[key], iter)

            for key in [ 'task', 'task_mean', 'total', 'sharing', 'sparsity']:
                if key not in self.losses:
                    continue
                # print(key + ':')
                if isinstance(self.losses[key], dict):
                    for subkey in self.losses[key].keys():
                        self.writer.add_scalar('trn_loss_%s/%s'%(key, subkey), self.losses[key][subkey], iter)
                elif (isinstance(self.losses[key], float)):
                    self.writer.add_scalar('trn_loss_%s'%(key), self.losses[key], iter)


    def print_val_metrics(self, epoch, iter, start_time, metrics=None, title='Iteration', verbose = False):
        """ write metrics to tensorboard and optionally to sysout """
        if metrics is None:
            metrics = self.val_metrics

        title = f"{title} ep:{epoch}    it:{iter}"

        ## Following items will be written as val_loss[key]:[subkey] to Tensorboard
        for key in ['task', 'task_mean', 'sharing', 'sparsity' , 'total']:
            if key not in metrics:
                continue
            # print(key + ':')
            if isinstance(metrics[key], dict):
                for subkey, metric_value in metrics[key].items():
                    self.writer.add_scalar('val_loss_%s/%s'%(key, subkey), metric_value, iter)
                    # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, metrics[key], time.time() - start_time)
            elif (isinstance(metrics[key], float)):
                self.writer.add_scalar('val_loss_%s'%(key), metrics[key], iter)
                # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, metrics[key], time.time() - start_time)

        ## Write aggregated metrics for each group (i.e, group of tasks)
        ## Following items will be written as val_metrics:[key]/[subkey] to Tensorboard
        for t_id, _ in enumerate(self.tasks):
            key = f"task{t_id+1}"
            print_heading(f"{title}  {iter}  {key} : {metrics[key]['classification_agg']}", verbose = verbose)

            for subkey, metric_value in metrics[key]['classification_agg'].items():
                self.writer.add_scalar(f"val_metrics:{key:s}/{subkey:s}", metric_value, iter)
                 # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, loss[key], time.time() - start_time)

        ## Write aggregated metrics (aggregated accross all groups/tasks)
        ## Following items will be written as val_metrics:[key]/[subkey] to Tensorboard
        key = "aggregated"
        print_heading(f"{title}  {iter}  {key} : {metrics[key]}", verbose = verbose)
        for subkey, metric_value  in metrics[key].items():
            self.writer.add_scalar(f"val_metrics:{key:s}/{subkey:s}", metric_value, iter)
            # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, loss[key], time.time() - start_time)


    def write_metrics_csv_heading(self, csv_file = None, losses = None):
        message = ' epoch, iteration, timestamp,elapsed,' 
        sorted_keys = sorted(['task', 'task_mean', 'parms', 'sharing', 'sparsity', 'total'])
        
        losses = self.initialize_loss_metrics() if losses is None else losses
        csv_file = self.loss_csv_file if csv_file is None else csv_file

        for key in sorted_keys:
            if key not in losses:
                continue

            if isinstance(losses[key], dict):
                for subkey in sorted(losses[key].keys()):
                    message += f"{key:s}.{subkey:s}," 
            elif (isinstance(losses[key], float)):
                    message += f"{key:s}.{key},"
        
        with open(csv_file, 'a') as f:
            f.write('%s \n' % message.rstrip(" ,"))


    def write_metrics_csv(self, log_name, epoch, iteration, elapsed, losses):
        message = '%4d,%4d,%26s,%6.3f,' % (epoch, iteration, timestring(), elapsed)
        sorted_keys = sorted([ 'task', 'task_mean', 'parms', 'sharing', 'sparsity', 'total'])
        
        for key in sorted_keys:
            if key not in losses:
                continue

            if isinstance(losses[key], dict):
                for subkey in sorted(losses[key].keys()):
                    message += f"{losses[key][subkey]}," 
            elif (isinstance(losses[key], float)):
                    message += f"{losses[key]},"
        
        with open(log_name, 'a') as log_file:
            log_file.write('%s \n' % message.rstrip(" ,"))


    def write_metrics_txt(self, log_name, epoch, iteration, elapsed, losses):
        sorted_keys = sorted( [ 'task', 'task_mean','parms', 'sharing', 'sparsity', 'total'])
        
        for key in sorted_keys:
            if key not in losses:
                continue
            message = 'epoch: %4d   iter: %4d, timestamp: %s wall clock time: %7.3f  %12s :' % (epoch, iteration, timestring(), elapsed, key)
            if isinstance(losses[key], dict):
                for subkey, value  in losses[key].values():
                    message += ' %s: %s ' % (subkey, str(value))
            elif (isinstance(losses[key], float)):
                message += ' %s: %.3f ' % (key, losses[key])

        with open(log_name, 'a') as log_file:
            log_file.write('%s \n' % message)


    def get_current_state(self, current_iter, current_epoch= -1):
        """ 
        original doc: change the state of each module  
        get the state_dict for all networks and optimizers
        return in current_state
        """
        current_state = {}
        for k, v in self.networks.items():
            if isinstance(v, nn.DataParallel):
                current_state[k] = v.module.state_dict()
            else:
                current_state[k] = v.state_dict()
        for k, v in self.optimizers.items():
            current_state[k] = v.state_dict()
        current_state['iter'] = current_iter
        current_state['epoch'] = current_epoch
        return current_state


    def load_snapshot(self, snapshot, verbose = False):
        """
        load snapshot
        """
        for k, v in self.networks.items():
            print_dbg(f'  networks -  network:  {k}', verbose = verbose)
            if k in snapshot.keys():
                print_dbg(f'  load snapshot - network:  {k}', verbose = verbose)
                # loading values for the existed keys
                model_dict = v.state_dict()
                pretrained_dict = {}
                for kk, vv in snapshot[k].items():
                    print_dbg(f'    network {k} - item {kk}', verbose = verbose)
                    if kk in model_dict.keys() and model_dict[kk].shape == vv.shape:
                        pretrained_dict[kk] = vv
                    else:
                        print_dbg('skipping %s' % kk, verbose = verbose)
                model_dict.update(pretrained_dict)
                self.networks[k].load_state_dict(model_dict)
                # self.networks[k].load_state_dict(snapshot[k])

        if self.is_train:
            for k, v in self.optimizers.items():
                print_dbg(f'  optimizers - optimizer:  {k}', verbose = verbose)
                if k in snapshot.keys():
                    print_dbg(f'    load snapshot - optimizer: {k} ', verbose = verbose)
                    self.optimizers[k].load_state_dict(snapshot[k])
        return


    def save_checkpoint(self, label, current_iter, current_epoch = 'unknown', verbose = False):
        """
        Save the current checkpoint
        :param label: str, the label for the loading checkpoint
        :param current_iter: int, the current iteration
        """
        save_filename = '%s_model.pth.tar' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        current_state = self.get_current_state(current_iter, current_epoch)
        torch.save(current_state, save_path)
        print_heading(f" Saved checkpoint to {save_path} iteration: {current_iter}", verbose = verbose)


    def load_checkpoint(self, label, path=None, verbose = False):
        """
        load the checkpoint
        :param label: str, the label for the loading checkpoint
        :param path: str, specify if knowing the checkpoint path
        """
        save_filename = '%s_model.pth.tar' % label
        if path is None:
            save_path = os.path.join(self.checkpoint_dir, save_filename)
        else:
            save_path = os.path.join(path,save_filename)
            # save_path = path

        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            print('=> loading snapshot to   {}'.format(self.device))
            if self.device == 'cpu':
                print(f'   Loading to CPU')
                snapshot = torch.load(save_path, map_location='cpu')
            else:
                print(f'   Loading to GPU {self.device}')
                # snapshot = torch.load(save_path, map_location='cuda:%d' % self.device_id)
                snapshot = torch.load(save_path, map_location=self.device)
            data = self.load_snapshot(snapshot, verbose = verbose)
            print(' data is : ', data)
            return data 
        else:
            raise ValueError('snapshot %s does not exist' % save_path)

    def save_policy(self, label, path = None, verbose = False):
        path = self.checkpoint_dir if path is None else path
        policy = {}
        for t_id in range(self.num_tasks):
            tmp = getattr(self, 'policy%d' % (t_id + 1))
            policy['task%d_policy' % (t_id + 1)] = tmp.cpu().data
            print_dbg(f' policy {t_id+1} \n {tmp}', verbose = verbose)
        save_filename = 'policy_%s.pickle' % str(label)
        save_path = os.path.join(path, save_filename)
        print_dbg(f" save_policy(): load policies to {save_path}", verbose = verbose)
        with open(save_path, 'wb') as handle:
            pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_policy(self, label, path = None, verbose = False):
        path = self.checkpoint_dir if path is None else path
        save_filename = 'policy_%s.pickle' % str(label)
        save_path = os.path.join(path, save_filename)
        print_dbg(f" load_policy(): load policies from {save_path}", verbose = verbose)

        with open(save_path, 'rb') as handle:
            policy = pickle.load(handle)

        for t_id in range(self.num_tasks):
            print_dbg(f"setting policy{t_id+1} attribute ....", verbose = verbose)
            setattr(self, 'policy%d' % (t_id + 1), policy['task%d_policy' % (t_id+1)])
            print_dbg(getattr(self, 'policy%d' % (t_id + 1)), verbose = verbose)


    def check_exist_policy(self, label, path = None, verbose = False):
        path = self.checkpoint_dir if path is None else path

        save_filename = 'policy_%s.pickle' % str(label)
        save_path = os.path.join(path, save_filename)
        print_dbg(f" check_exist_policy() : check for policy file  {save_path}", verbose = verbose)
        return os.path.exists(save_path)


    def visualize(self):
        pass
        # ##################### visualize #######################
        #     # TODO: implement the visualization of depth
        #     save_results = {}
        #     if 'seg' in self.tasks:
        #         num_seg_class = self.tasks_num_class[self.tasks.index('seg')]
        #         self.save_seg = decode_labels2(torch.argmax(self.seg_output, dim=1).unsqueeze(dim=1), num_seg_class, 'seg', self.seg)
        #         self.save_gt_seg = decode_labels2(self.seg, num_seg_class, 'seg', self.seg)
        #         save_results['save_seg'] = self.save_seg
        #         save_results['save_gt_seg'] = self.save_gt_seg
        #     if 'sn' in self.tasks:
        #         self.save_normal = decode_labels2(F.normalize(self.sn_output) * 255, None, 'normal', F.normalize(self.normal.float()) * 255)
        #         self.save_gt_normal = decode_labels2(F.normalize(self.normal.float()) * 255, None, 'normal', F.normalize(self.normal.float()) * 255,)
        #         save_results['save_sn'] = self.save_normal
        #         save_results['save_gt_sn'] = self.save_gt_normal
        #     if 'depth' in self.tasks:
        #         self.save_depth = decode_labels2(self.depth_output, None, 'depth', self.depth.float())
        #         self.save_gt_depth = decode_labels2(self.depth.float(), None, 'depth', self.depth.float())
        #         save_results['save_depth'] = self.save_depth
        #         save_results['save_gt_depth'] = self.save_gt_depth
        #     self.save_img = inv_preprocess(self.img)
        #     save_results['save_img'] = self.save_img
        #     return save_results
        # #######################################################


    def train(self):
        # ##################### change the state of each module ####################################
        """
        Change to the training mode
        """
        for k, v in self.networks.items():
            v.train()


    def eval(self):
        """
        Change to the eval mode
        """
        for k, v in self.networks.items():
            v.eval()


    def cuda(self, gpu_ids):
        """
        Move network items to assigned GPU device (self.device)
        """
        if len(gpu_ids) == 1:
            for k, v in self.networks.items():
                v.to(self.device)
        else:
            for k, v in self.networks.items():
                self.networks[k] = nn.DataParallel(v, device_ids=gpu_ids)
                self.networks[k].to(self.device)

    def cpu(self):
        """
        Move network items to  CPU device
        """
        print(f'base_env.cpu()')
        for k, v in self.networks.items():
            print(f' Network item {k} moved to cpu')
            v.cpu()

    def name(self):
        return 'BaseEnv'