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
from pynvml import *
import pandas as pd
from utils.notebook_modules import (initialize, init_dataloaders, init_environment, init_wandb, 
                                   training_prep, disp_dataloader_info,disp_info_1, 
                                   warmup_phase, weight_policy_training, disp_gpu_info,
                                    init_dataloaders_by_fold_id)
                                    

from utils.util import (print_separator, print_heading, timestring, print_loss, load_from_pickle) 

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
torch.set_printoptions(precision=6, linewidth=132)
pd.options.display.width = 132
# disp_gpu_info() 

# ********************************************************************
# ******************** Display GPU information ***********************
# ********************************************************************  
disp_gpu_info()
if torch.cuda.is_available():
    print('cuda is available')
    nvmlInit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# torch.cuda package supports CUDA tensor types but works with GPU computations. Hence, if GPU is used, it is common to use CUDA. 
torch.cuda.current_device()
torch.cuda.device_count()
torch.cuda.get_device_name(0)

torch_gpu_id = torch.cuda.current_device()
print(torch_gpu_id)
if "CUDA_VISIBLE_DEVICES" in os.environ:
  ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
  print(' ids : ', ids)
  nvml_gpu_id = ids[torch_gpu_id] # remap
else:
  nvml_gpu_id = torch_gpu_id
print('nvml_gpu_id: ', nvml_gpu_id)
nvml_handle = nvmlDeviceGetHandleByIndex(nvml_gpu_id)
print(nvml_handle)

info = nvmlDeviceGetMemoryInfo(nvml_handle)
print(info) 


# ********************************************************************
# ************************  Initialization ***************************
# ********************************************************************  
opt, ns = initialize(build_folders = True)

# ********************************************************************
# ************ Dataloaders and Envronment Initialization *************
# ********************************************************************  

# dldrs = init_dataloaders(opt)
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