#!/usr/bin/env python
# coding: utf-8
import os 
import sys
sys.path.insert(0, '.')
import time
import argparse
import yaml
import types
import copy, pprint
from time import sleep
from datetime import datetime
import numpy  as np
from pynvml import *
import pandas as pd

from utils import (initialize, init_dataloaders, init_environment, init_wandb, training_initializations, model_initializations, 
                   check_for_resume_training, disp_dataloader_info, disp_info_1, warmup_phase, weight_policy_training, 
                   display_gpu_info, init_dataloaders_by_fold_id, print_separator, print_heading, 
                   timestring, print_loss, get_command_line_args, load_from_pickle)   

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
pd.options.display.width = 132
# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
# disp_gpu_info() 


# ********************************************************************
# ************************  Initialization ***************************
# ********************************************************************  
opt, ns = initialize(build_folders = True)

# ********************************************************************
# ************ Dataloaders and Envronment Initialization *************
# ********************************************************************  
dldrs = init_dataloaders_by_fold_id(opt, verbose = False)
disp_dataloader_info(dldrs)


# ********************************************************************
# **************** define optimizer and schedulers *******************
# ********************************************************************  
environ = init_environment(ns, opt, is_train = True, policy_learning = False, display_cfg = True)
                              

print(f" Current LR: {environ.optimizers['alphas'].param_groups[0]['lr'] }")
print(f" Current LR: {environ.optimizers['weights'].param_groups[0]['lr']}")
print(f" Current LR: {environ.optimizers['weights'].param_groups[1]['lr']}")

# check_for_resume_training(ns, opt, environ)
print_separator('Initiate Training ')

training_prep(ns, opt, environ, dldrs )
disp_info_1(ns, opt, environ)
print(environ.disp_for_excel())


# ********************************************************************
# ************************ warmup training  **************************
# ********************************************************************        

print_separator('TRAINING')
print_heading(f" Last Epoch: {ns.current_epoch}   # of warm-up epochs to do:  {ns.warmup_epochs}"
              f" - Run epochs {ns.current_epoch+1} to {ns.current_epoch + ns.warmup_epochs}", verbose = True)

# warmup_phase(ns,opt, environ, dldrs, epochs = 25)
warmup_phase(ns,opt, environ, dldrs)         

# Post warmup training  

# print( f" Backbone Learning Rate      : {environ.opt['train']['backbone_lr']}\n"
#        f" Tasks    Learning Rate      : {environ.opt['train']['task_lr']}\n"
#        f" Policy   Learning Rate      : {environ.opt['train']['policy_lr']}\n")
# print( f" Sparsity regularization     : {environ.opt['train']['lambda_sparsity']}\n"
#        f" Sharing  regularization     : {environ.opt['train']['lambda_sharing']} \n\n"
#        f" Tasks    regularization     : {environ.opt['train']['lambda_tasks']}   \n"
#        f" Gumbel Temp                 : {environ.gumbel_temperature:.4f}         \n" #
#        f" Gumbel Temp decay           : {environ.opt['train']['decay_temp_freq']}") #
print(' current lr: ', environ.optimizers['alphas'].param_groups[0]['lr'],)
print(' current lr: ', environ.optimizers['weights'].param_groups[0]['lr'])
print(' current lr: ', environ.optimizers['weights'].param_groups[1]['lr'])


print(f"ns.current_epoch           : {ns.current_epoch}") 
print(f"ns.training_epochs         : {ns.training_epochs} \n") 
print(f"ns.current_iters           : {ns.current_iter}")  
print(f"Batches in weight epoch    : {ns.stop_iter_w}")
print(f"Batches in policy epoch    : {ns.stop_iter_a}")
print(f"num_train_layers           : {ns.num_train_layers}")
print()
print_loss(environ.val_metrics, title = f"[e] Last ep:{ns.current_epoch}  it:{ns.current_iter}") 

print(f"Best Epoch :       {ns.best_epoch}\n"
      f"Best Iteration :   {ns.best_iter} \n"
      f"Best Precision :   {ns.best_value}\n")

ns.wandb_run.finish()