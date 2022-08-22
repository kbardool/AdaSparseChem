#!/usr/bin/env python
# coding: utf-8
import os 
import sys
sys.path.insert(0, '../src')
import time
import argparse
import yaml
import types
import copy
import pprint
from   time import sleep
from   datetime import datetime
import numpy  as np
from   pynvml import *
import pandas as pd

from utils import (initialize, init_dataloaders, init_environment, init_wandb, training_initializations, model_initializations, 
                   check_for_resume_training, disp_dataloader_info, disp_info_1, warmup_phase, weight_policy_training, 
                   display_gpu_info, init_dataloaders_by_fold_id, print_separator, print_heading, 
                   timestring, print_loss, get_command_line_args, load_from_pickle)                    

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
pd.options.display.width = 132

ns = types.SimpleNamespace()
# input_args = input_args.split() if input_args is not None else input_args
ns.args = get_command_line_args(None, display = True)

print(f" cuda_devices : {ns.args.cuda_devices}")
os.environ["CUDA_VISIBLE_DEVICES"]=ns.args.cuda_devices

# import torch  
# torch.set_printoptions(precision=6, linewidth=132)


#-----------------------------------------------------------------
# ### Parse Input Args, Read YAML config file, wandb initialization
#-----------------------------------------------------------------
opt = initialize(ns, build_folders = True)


#-----------------------------------------------------------------
# ### Setup Dataloader 
#-----------------------------------------------------------------
dldrs = init_dataloaders_by_fold_id(opt, verbose = False)
disp_dataloader_info(dldrs)

#-----------------------------------------------------------------
# ### Setup Model Environment  
#-----------------------------------------------------------------
environ = init_environment(ns, opt, is_train = True, policy_learning = False, display_cfg = True)

#-----------------------------------------------------------------
# ### Initiate / Resume Training Prep
#-----------------------------------------------------------------
check_for_resume_training(ns, opt, environ, epoch = 0 , iter = 0)

#-----------------------------------------------------------------
# ### Training Preparation
#-----------------------------------------------------------------
model_initializations(ns, opt, environ, phase='update_weights', policy_learning = False)
training_initializations(ns, opt, environ, dldrs, phase='update_weights', warmup = True)
# print('-'*80)
# disp_info_1(ns, opt, environ)
print('-'*80)
print(environ.disp_for_excel())


#-----------------------------------------------------------------
# ### Warmup Training
#-----------------------------------------------------------------
# environ.display_trained_policy(ns.current_epoch,out=sys.stdout)
print_heading(f" Last Epoch: {ns.current_epoch}   # of warm-up epochs to do:  {ns.warmup_epochs} - "\
              f"Run epochs {ns.current_epoch+1} to {ns.current_epoch + ns.warmup_epochs}", verbose = True)


# warmup_phase(ns,opt, environ, dldrs, write_checkpoint=False)
warmup_phase(ns,opt, environ, dldrs, verbose = False, disable_tqdm = True)

 

print(f"Best Epoch :       {ns.best_epoch}\n"
      f"Best Iteration :   {ns.best_iter} \n"
      f"Best ROC AUC   :   {ns.best_roc_auc:.5f}\n"
      f"Best Precision :   {ns.best_accuracy:.5f}\n")
print()
for key in environ.val_metrics['aggregated']:
    print(f"{key:20s}    {environ.val_metrics['aggregated'][key]:0.4f}")
print()
df = environ.val_metrics['task1']['classification']
print(df[pd.notna(df.roc_auc_score)])


#-----------------------------------------------------------------
# ### Weight/Policy  Training Preparation
#-----------------------------------------------------------------

ns.flag = 'update_weights'
ns.flag_warmup = False
model_initializations(ns, opt, environ, phase = ns.flag, policy_learning = True)
training_initializations(ns, opt, environ, dldrs, phase='update_weights', weight_iterations = 10, policy_iterations = 10, warmup = False)

print_heading( f"** {timestring()} \n"
               f"** Training epoch: {ns.current_epoch} iter: {ns.current_iter}   flag: {ns.flag} \n"
               f"** Set optimizer and scheduler to policy_learning = True (Switch weight optimizer from ADAM to SGD)\n"
               f"** Switch from Warm Up training to Alternate training Weights & Policy \n"
               f"** Take checkpoint and block gradient flow through Policy net", verbose=True)

print( f" ns.flag_warmup                 :      {ns.flag_warmup}")
print( f" ns.flag                        :      {ns.flag}")
print( f" num_train_layers               :      {ns.num_train_layers}")
print( f" environ.opt['is_curriculum']   :      {environ.opt['is_curriculum']}")
print( f" environ.opt['curriculum_speed']:      {environ.opt['curriculum_speed']}\n")
print( f" Backbone Initial LR            :      {environ.opt['train']['backbone_lr']:4f}      current LR : {environ.optimizers['alphas'].param_groups[0]['lr']} \n"
       f" Tasks    Initial LR            :      {environ.opt['train']['task_lr']:4f}      current LR : {environ.optimizers['weights'].param_groups[0]['lr']}    \n"
       f" Policy   Initial LR            :      {environ.opt['train']['policy_lr']:4f}      current LR : {environ.optimizers['weights'].param_groups[1]['lr']}  \n")
print( f" Sparsity regularization        :      {environ.opt['train']['lambda_sparsity']}\n"
       f" Sharing  regularization        :      {environ.opt['train']['lambda_sharing']} \n\n"
       f" Tasks    regularization        :      {environ.opt['train']['lambda_tasks']}   \n"
       f" Gumbel Temp                    :      {environ.gumbel_temperature:.4f}         \n" #
       f" Gumbel Temp decay              :      {environ.opt['train']['decay_temp_freq']} \n") #

print( f" ns.current_epoch               :      {ns.current_epoch}")
print( f" ns.training_epochs             :      {ns.training_epochs} \n") 
print( f" ns.current_iters               :      {ns.current_iter}")  
print( f" Batches in weight epoch        :      {ns.trn_iters_weights}")
print( f" Batches in policy epoch        :      {ns.trn_iters_policy}")
print( f" num_train_layers               :      {ns.num_train_layers} \n")

print_loss(ns.val_metrics, title = f"[e] Last ep:{ns.current_epoch}  it:{ns.current_iter} ")
print()

environ.display_trained_policy(ns.current_epoch)
environ.display_trained_logits(ns.current_epoch)

#-----------------------------------------------------------------
# ### Weight/Policy Training
#-----------------------------------------------------------------
weight_policy_training(ns, opt, environ, dldrs, disable_tqdm = True)

#-----------------------------------------------------------------
# ### Post Training
#-----------------------------------------------------------------

environ.display_trained_policy(ns.current_epoch)
environ.display_trained_logits(ns.current_epoch)

print(f"Best Epoch :       {ns.best_epoch}\n"
      f"Best Iteration :   {ns.best_iter} \n"
      f"Best ROC AUC   :   {ns.best_roc_auc:.5f}\n"
      f"Best Precision :   {ns.best_accuracy:.5f}\n")
print()
for key in environ.val_metrics['aggregated']:
    print(f"{key:20s}    {environ.val_metrics['aggregated'][key]:0.4f}")
print()
df = environ.val_metrics['task1']['classification']
print(df[pd.notna(df.roc_auc_score)])



ns.wandb_run.finish()