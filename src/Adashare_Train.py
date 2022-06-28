#!/usr/bin/env python
# coding: utf-8

# ## Initialization  
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
# import wandb
from   pynvml import *
import pandas as pd
from utils.notebook_modules import (initialize, init_dataloaders, init_environment, init_wandb, 
                                   training_prep, disp_dataloader_info,disp_info_1, 
                                   warmup_phase, weight_policy_training, display_gpu_info,
                                   init_dataloaders_by_fold_id)                                    

from utils import (print_separator, print_heading, timestring, print_loss, load_from_pickle,
                   print_underline, print_dbg, get_command_line_args ) 

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
pd.options.display.width = 132

ns = types.SimpleNamespace()
    
# input_args = input_args.split() if input_args is not None else input_args

ns.args = get_command_line_args(None, display = True)

print(f" cuda_devices : {ns.args.cuda_devices}")
os.environ["CUDA_VISIBLE_DEVICES"]=ns.args.cuda_devices

import torch  
torch.set_printoptions(precision=6, linewidth=132)


#-----------------------------------------------------------------
# ### Parse Input Args, Read YAML config file, wandb initialization
#-----------------------------------------------------------------
opt = initialize(ns, build_folders = True)


#-----------------------------------------------------------------
# ### Setup Dataloader 
#-----------------------------------------------------------------
# dldrs = init_dataloaders(opt, verbose = False)
dldrs = init_dataloaders_by_fold_id(opt, verbose = False)
disp_dataloader_info(dldrs)

#-----------------------------------------------------------------
# ### Setup Model Environment  
#-----------------------------------------------------------------
environ = init_environment(ns, opt, is_train = True, policy_learning = False, display_cfg = True)
# environ.define_optimizer(policy_learning=False)
# environ.define_scheduler(policy_learning=False)

#-----------------------------------------------------------------
# ### Initiate / Resume Training Prep
#-----------------------------------------------------------------
# check_for_resume_training(ns, opt, environ)

#-----------------------------------------------------------------
# ### Training Preparation
#-----------------------------------------------------------------
training_prep(ns, opt, environ, dldrs)
# print('-'*80)
# disp_info_1(ns, opt, environ)
print('-'*80)
print(environ.disp_for_excel())


#-----------------------------------------------------------------
# ## Warmup Training
#-----------------------------------------------------------------
# environ.display_trained_policy(ns.current_epoch,out=sys.stdout)
print_heading(f" Last Epoch: {ns.current_epoch}   # of warm-up epochs to do:  {ns.warmup_epochs} - "\
              f"Run epochs {ns.current_epoch+1} to {ns.current_epoch + ns.warmup_epochs}", verbose = True)


warmup_phase(ns,opt, environ, dldrs, write_checkpoint=False)

print(f"Best Epoch :       {ns.best_epoch}\n"
      f"Best Iteration :   {ns.best_iter} \n"
      f"Best Accuracy  :   {ns.best_accuracy:.5f}\n"
      f"Best ROC AUC   :   {ns.best_roc_auc:.5f}\n")

print()
pp.pprint(environ.val_metrics['aggregated'])
print()
df = environ.val_metrics['task1']['classification']
print(df[pd.notna(df.roc_auc_score)])

ns.wandb_run.finish()

