#!/usr/bin/env python
# coding: utf-8

# ## Initialization  

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import os 
import sys
sys.path.insert(0, '../src')
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
from pynvml import *
import pandas as pd
from utils.notebook_modules import (initialize, init_dataloaders, init_environment, init_wandb, 
                                   training_prep, disp_dataloader_info,disp_info_1, 
                                   warmup_phase, weight_policy_training, display_gpu_info,
                                init_dataloaders_by_fold_id)
                                    

from utils import (print_separator, print_heading, timestring, print_loss, load_from_pickle) #, print_underline, 
#                       print_dbg, get_command_line_args ) 

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
torch.set_printoptions(precision=6, linewidth=132)
pd.options.display.width = 132
# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
# sys.path.insert(0, '/home/kbardool/kusanagi/AdaSparseChem/src')
# print(sys.path)
# disp_gpu_info() 
os.environ["WANDB_NOTEBOOK_NAME"] = "Adashare_Training-Chembl_mini.ipynb"


# display_gpu_info()

# ## Create Environment

# ### Parse Input Args  - Read YAML config file - wandb initialization

# synthetic_1task_config = "../yamls/chembl_synt_train_1task.yaml"
# synthetic_3task_config = "../yamls/chembl_synt_train_3task.yaml"
# synthetic_5task_config = "../yamls/chembl_synt_train_5task.yaml"
# synthetic_config = "../yamls/chembl_synt_train.yaml"
# mini_config      = "../yamls/chembl_mini_train.yaml"
##  For Initiating 
##
# input_args = f" --config  {mini_config} "              
#               " --exp_name       0410_1947 "              
#               " --exp_desc     weight 105 bch/ep policy 105 bch/ep "              
#               " --warmup_epochs       100 "               
#               " --hidden_size       40 40 "               
#               " --tail_hidden_size     40 "               
#               " --first_dropout       0.0 "               
#               " --middle_dropout      0.0 "               
#               " --last_dropout        0.0 "               
#               " --seed_idx              0 "               
#               " --batch_size          128 "               
#               " --task_lr            0.01 "               
#               " --backbone_lr        0.01 "               
#               " --decay_lr_rate       0.3 "               
#               " --decay_lr_freq        10 "               
#               " --policy_lr         0.001 "               
#               " --lambda_sparsity    0.02 "               
#               " --lambda_sharing     0.01 "               
#               " --folder_sfx       no_resid"               
#               " --cpu "              
#               " --no_residual "

#              " --hidden_size   100 100 100 100 100 100" \
#              " --tail_hidden_size  100 " \
#              " --decay_lr_rate      0.75"  \
#              " --decay_lr_freq       20"  \


opt, ns = initialize(None, build_folders = True)


# ### Setup Dataloader and Model  
# dldrs = init_dataloaders(opt, verbose = False)
dldrs = init_dataloaders_by_fold_id(opt, verbose = False)
disp_dataloader_info(dldrs)

environ = init_environment(ns, opt, is_train = True, policy_learning = False, display_cfg = True)
# environ.define_optimizer(policy_learning=False)
# environ.define_scheduler(policy_learning=False)


# environ.optimizers['weights'].param_groups[0]
# print(environ.print_configuration())


# ### Initiate / Resume Training Prep
# ns.wandb_run.finish()
# check_for_resume_training(ns, opt, environ)

# ### Training Preparation
training_prep(ns, opt, environ, dldrs)
# print('-'*80)
# disp_info_1(ns, opt, environ)
print('-'*80)
print(environ.disp_for_excel())


# ## Warmup Training

# environ.display_trained_policy(ns.current_epoch,out=sys.stdout)
# ns.check_for_improvment_wait = 0
# ns.warmup_epochs = 100
print_heading(f" Last Epoch: {ns.current_epoch}   # of warm-up epochs to do:  {ns.warmup_epochs} - Run epochs {ns.current_epoch+1} to {ns.current_epoch + ns.warmup_epochs}", verbose = True)


warmup_phase(ns,opt, environ, dldrs)

print(f"Best Epoch :       {ns.best_epoch}\n"
      f"Best Iteration :   {ns.best_iter} \n"
      f"Best Precision :   {ns.best_value:.5f}\n")
print()
pp.pprint(environ.val_metrics['aggregated'])


# warmup_phase(ns,opt, environ, dldrs, epochs = 25)
ns.wandb_run.finish()

