# %load_ext autoreload
# %autoreload 2
import os 
import sys
# sys.path.insert(0, '/home/kbardool/kusanagi/AdaSparseChem/src')
sys.path.insert(0, '.')
# for pth in sys.path:
#     print(pth)
# print(sys.path)
import time
import argparse
import yaml
import types
import copy, pprint
from time import sleep
from datetime import datetime
import numpy  as np
import torch  
import wandb
import pandas as pd
from utils.notebook_modules import initialize, init_dataloaders, init_environment, init_wandb, \
                                   training_prep, disp_dataloader_info,disp_info_1, \
                                   warmup_phase, weight_policy_training, disp_gpu_info

from utils.util import (print_separator, print_heading, timestring, print_loss, load_from_pickle) 
#  print_underline, load_from_pickle, print_dbg, get_command_line_args ) 

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
torch.set_printoptions(precision=6, linewidth=132)
pd.options.display.width = 132
# disp_gpu_info() 
# os.environ["WANDB_NOTEBOOK_NAME"] = "Adashare_Training.ipynb"

opt, ns = initialize(build_folders = True)

# ********************************************************************
# ************ Dataloaders and Envronment Initialization *************
# ********************************************************************  

dldrs = init_dataloaders(opt)

disp_dataloader_info(dldrs)

environ = init_environment(ns, opt, is_train = True, policy_learning = False, display_cfg = True)


# ********************************************************************
# ********************** WandB Initialization ************************
# ********************************************************************  
                              
init_wandb(ns, opt, environment = environ)

print(f" PROJECT NAME: {ns.wandb_run.project}\n"
      f" RUN ID      : {ns.wandb_run.id} \n"
      f" RUN NAME    : {ns.wandb_run.name}") 


# ********************************************************************
# **************** define optimizer and schedulers *******************
# ********************************************************************  
                              
environ.define_optimizer(policy_learning=False)
environ.define_scheduler(policy_learning=False)

print(f" Current LR: {environ.optimizers['alphas'].param_groups[0]['lr'] }")
print(f" Current LR: {environ.optimizers['weights'].param_groups[0]['lr']}")
print(f" Current LR: {environ.optimizers['weights'].param_groups[1]['lr']}")

if opt['train']['resume']:
    RESUME_MODEL_CKPT = ""
    RESUME_METRICS_CKPT = ""
    print(opt['train']['which_iter'])
    print(opt['paths']['checkpoint_dir'])
    print(RESUME_MODEL_CKPT)
    # opt['train']['resume'] = True
    # opt['train']['which_iter'] = 'warmup_ep_40_seed_0088'
    print_separator('Resume training')
    loaded_iter, loaded_epoch = environ.load_checkpoint(RESUME_MODEL_CKPT, path = opt['paths']['checkpoint_dir'], verbose = True)
    print(loaded_iter, loaded_epoch)    
#     current_iter = environ.load_checkpoint(opt['train']['which_iter'])
    environ.networks['mtl-net'].reset_logits()
    val_metrics = load_from_pickle(opt['paths']['checkpoint_dir'], RESUME_METRICS_CKPT)
    # training_prep(ns, opt, environ, dldrs, epoch = loaded_epoch, iter = loaded_iter )

else:
    print_separator('Initiate Training ')

training_prep(ns, opt, environ, dldrs )
disp_info_1(ns, opt, environ)
print(environ.disp_for_excel())


# ********************************************************************
# ************************ warmup training  **************************
# ********************************************************************        

print_separator('WARMUP TRAINING')
print(ns.warmup_epochs, ns.current_epoch)
print_heading(f" Last Epoch: {ns.current_epoch}   # of warm-up epochs to do:  {ns.warmup_epochs}"
              f" - Run epochs {ns.current_epoch+1} to {ns.current_epoch + ns.warmup_epochs}", verbose = True)

# warmup_phase(ns,opt, environ, dldrs, epochs = 25)
warmup_phase(ns,opt, environ, dldrs)         

# Post warmup training  

print( f" Backbone Learning Rate      : {environ.opt['train']['backbone_lr']}\n"
       f" Tasks    Learning Rate      : {environ.opt['train']['task_lr']}\n"
       f" Policy   Learning Rate      : {environ.opt['train']['policy_lr']}\n")
print( f" Sparsity regularization     : {environ.opt['train']['lambda_sparsity']}\n"
       f" Sharing  regularization     : {environ.opt['train']['lambda_sharing']} \n\n"
       f" Tasks    regularization     : {environ.opt['train']['lambda_tasks']}   \n"
       f" Gumbel Temp                 : {environ.gumbel_temperature:.4f}         \n" #
       f" Gumbel Temp decay           : {environ.opt['train']['decay_temp_freq']}") #
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
 