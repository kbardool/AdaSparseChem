# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:98% !important; }</style>"))
# %load_ext autoreload
# %autoreload 2

import os 
import sys
print(sys.path)
sys.path.insert(0, './src')
import time
import argparse
import yaml
# from tqdm import tqdm, tqdm_notebook, trange
# import tqdm.notebook.trange as tnrange
import copy, pprint
import numpy  as np
import torch  
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import scipy.sparse
from time import sleep
from scipy.special import softmax
 
from datetime import datetime
from GPUtil import showUtilization as gpu_usage
from tqdm.notebook import trange, tqdm

from envs         import SparseChemEnv
from dataloaders  import ClassRegrSparseDataset_v3, ClassRegrSparseDataset, InfiniteDataLoader
from utils.util   import ( makedir, print_separator, create_path, print_yaml, print_yaml2, print_loss, should, 
                         fix_random_seed, read_yaml_from_input, timestring, print_heading, print_dbg, 
                         print_underline, write_parms_report, get_command_line_args, is_notebook, print_metrics_cr)


if is_notebook():
    from tqdm.notebook     import tqdm,trange
else:
    from tqdm     import tqdm,trange

print(' Cuda is available  : ', torch.cuda.is_available())
print(' CUDA device count  : ', torch.cuda.device_count())
print(' CUDA current device: ', torch.cuda.current_device())
print(' GPU Processes      : \n', torch.cuda.list_gpu_processes())
print()

for i in range(torch.cuda.device_count()):
    print(f" Device : cuda:{i}")
    print('   name:       ', torch.cuda.get_device_name())
    print('   capability: ', torch.cuda.get_device_capability())
    print('   properties: ', torch.cuda.get_device_properties(i))
    ## current GPU memory usage by tensors in bytes for a given device
    print('   Allocated : ', torch.cuda.memory_allocated(i) ) 
    ## current GPU memory managed by caching allocator in bytes for a given device, in previous PyTorch versions the command was torch.cuda.memory_cached
    print('   Reserved  : ', torch.cuda.memory_reserved(i) )   
    print()

gpu_usage()                             

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
torch.set_printoptions(precision=6, linewidth=132)
# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
pd.options.display.width = 132

# input_args = " --config yamls/adashare/chembl_2task.yaml --cpu --batch_size 09999".split()
# get command line arguments
args = get_command_line_args()
print(args)
print()

if args.exp_name is None:
    args.exp_name = datetime.now().strftime("%m%d_%H%M")
    
print(args.exp_name)

print_separator('READ YAML')

opt, gpu_ids, _ = read_yaml_from_input(args)

fix_random_seed(opt["seed"][1])

opt['exp_description'] = f"No Alternating Weight/Policy - training all done with both weights and policy"
# folder_name=  f"{opt['exp_name']}_bs{opt['batch_size']:03d}_{opt['train']['decay_lr_rate']:3.2f}_{opt['train']['decay_lr_freq']}"

print_heading(f" Project name          : {opt['exp_name']} \n"
              f" experiment name       : {opt['exp_name']} \n"
              f" folder_name           : {opt['exp_folder']} \n"
              f" experiment description: {opt['exp_description']}\n"
              f" log folder            : {opt['exp_log_dir']}\n"
              f" checkpoint folder     : {opt['exp_checkpoint_dir']}", verbose = True)

# for line in lines: 
create_path(opt)    

# # print yaml on the screen
# for line in print_yaml2(opt):
#     print(line)

write_parms_report(opt)    

##---------------------------------------------------------------------
##  Create Dataloaders
##---------------------------------------------------------------------

trainset  = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 0, verbose = False)
trainset1 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 1)
trainset2 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 2)
valset    = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 3)

train_loader  = InfiniteDataLoader(trainset , batch_size=opt['batch_size'], num_workers = 2, pin_memory=True, collate_fn=trainset.collate, shuffle=False)
val_loader    = InfiniteDataLoader(valset   , batch_size=opt['batch_size'], num_workers = 1, pin_memory=True, collate_fn=valset.collate  , shuffle=False)
train1_loader = InfiniteDataLoader(trainset , batch_size=opt['batch_size'], num_workers = 2, pin_memory=True, collate_fn=trainset.collate, shuffle=False)
train2_loader = InfiniteDataLoader(trainset , batch_size=opt['batch_size'], num_workers = 2, pin_memory=True, collate_fn=trainset.collate, shuffle=False)

# train1_loader = InfiniteDataLoader(trainset1, batch_size=opt['batch_size'], num_workers = 2, pin_memory=True, collate_fn=trainset1.collate, shuffle=False)
# train2_loader = InfiniteDataLoader(trainset2, batch_size=opt['batch_size'], num_workers = 2, pin_memory=True, collate_fn=trainset2.collate, shuffle=False)



##---------------------------------------------------------------------
##  Create Model
##---------------------------------------------------------------------
environ = SparseChemEnv(log_dir          = opt['exp_log_dir'], 
                        checkpoint_dir   = opt['exp_checkpoint_dir'], 
                        exp_name         = opt['exp_name'],
                        tasks_num_class  = opt['tasks_num_class'], 
                        init_neg_logits  = opt['train']['init_neg_logits'], 
                        device           = gpu_ids[0],
                        init_temperature = opt['train']['init_temp'], 
                        temperature_decay= opt['train']['decay_temp'], 
                        is_train         = True,
                        opt              = opt, 
                        verbose          = False)

cfg = environ.print_configuration()
write_parms_report(opt, cfg, mode = 'a')

# print(environ.networks['mtl-net'])


##---------------------------------------------------------------------
##  Training Prep
##---------------------------------------------------------------------
environ.define_optimizer(policy_learning=False)
environ.define_scheduler(policy_learning=False)
# Fix Alpha - 
environ.fix_alpha()
environ.free_weights(opt['fix_BN'])

if opt['train']['resume']:
    print_separator('Resume training')
    current_iter = environ.load(opt['train']['which_iter'])
    environ.networks['mtl-net'].reset_logits()
else:
    print_separator('Initiate Training ')

if torch.cuda.is_available():
    print(' cuda available', gpu_ids)   
    environ.cuda(gpu_ids)
else:
    print(' cuda not available')
    environ.cpu()

if opt['train']['print_freq'] == -1:
    print(f" set print_freq to length of train loader: {len(train_loader)}")
    opt['train']['print_freq']    = len(train_loader)

if opt['train']['val_iters'] == -1:
    print(f" set eval_iters to length of val loader  : {len(val_loader)}")
    eval_iters    = len(val_loader)    
else:
    eval_iters    = opt['train']['val_iters']

opt['train']['weight_iter_alternate'] = opt['train'].get('weight_iter_alternate' , len(train1_loader))
opt['train']['alpha_iter_alternate']  = opt['train'].get('alpha_iter_alternate'  , len(train2_loader))
# opt['train']['weight_iter_alternate'] = len(train_loader)
# opt['train']['alpha_iter_alternate']  = len(train_loader)
stop_iter_w = opt['train']['weight_iter_alternate']
stop_iter_a = opt['train']['alpha_iter_alternate'] 
    
    
flag           = 'update_w'
current_epoch  = 0
current_iter   = 0
current_iter_w = 0 
current_iter_a = 0
best_value     = 0 
best_iter      = 0
p_epoch        = 0
w_epoch        = 0

best_metrics   = None
flag_warmup    = True
num_prints     = 0
num_blocks     = sum(environ.networks['mtl-net'].layers)

warm_up_epochs     = opt['train']['warm_up_epochs']
train_total_epochs = opt['train']['training_epochs']
curriculum_speed   = opt['curriculum_speed'] 

stop_epoch_warmup  = current_epoch + warm_up_epochs
 
##---------------------------------------------------------------------
##  Display Some Parms 
##---------------------------------------------------------------------
print(f" trainset.y_class                       :  {[ i.shape  for i in trainset.y_class_list]}")
print(f" trainset1.y_class                      :  {[ i.shape  for i in trainset1.y_class_list]}")
print(f" trainset2.y_class                      :  {[ i.shape  for i in trainset2.y_class_list]}")
print(f" valset.y_class                         :  {[ i.shape  for i in valset.y_class_list  ]} ")
print()
print(f' size of training set 0 (warm up)       :  {len(trainset)}')
print(f' size of training set 1 (network parms) :  {len(trainset1)}')
print(f' size of training set 2 (policy weights):  {len(trainset2)}')
print(f' size of validation set                 :  {len(valset)}')
print(f'                               Total    :  {len(trainset)+len(trainset1)+len(trainset2)+len(valset)}')
print()
print(f" batch size                             :  {opt['batch_size']}")
print()
print(f" # batches training 0 (warm up)         :  {len(train_loader)}")
print(f" # batches training 1 (network parms)   :  {len(train1_loader)}")
print(f" # batches training 2 (policy weights)  :  {len(train2_loader)}")
print(f" # batches validation dataset           :  {len(val_loader)}")
print()


print(f"\n experiment name           : {opt['exp_name']}",
      f"\n experiment description    : {opt['exp_description']}",
      f"                                \n"
      f"\n Network[mtl_net].layers   : {environ.networks['mtl-net'].layers}",
      f"\n Num_blocks                : {sum(environ.networks['mtl-net'].layers)}"    
      f"                                \n"
      f"\n batch size                : {opt['batch_size']}",    
      f"\n Total iterations          : {opt['train']['total_iters']}",
      f"\n Warm-up iterations        : {opt['train']['warm_up_iters']}",
      f"\n Warm-up epochs            : {opt['train']['warm_up_epochs']}",
      f"\n Warm-up stop              : {stop_epoch_warmup}",
      f"\n train_total_epochs        : {train_total_epochs}",
      f"                                \n"
      f"\n Print Frequency           : {opt['train']['print_freq']}",
      f"\n Validation Frequency      : {opt['train']['val_freq']}",
      f"\n Validation Iterations     : {opt['train']['val_iters']}",
      f"\n eval_iters                : {eval_iters}",
      f"\n which_iter                : {opt['train']['which_iter']}",
      f"\n train_resume              : {opt['train']['resume']}",
      f"                                \n",                     
      f"\n Length train_loader       : {len(train_loader)}",
      f"\n Length val_loader         : {len(val_loader)}",
      f"\n stop_iter_w               : {stop_iter_w}",
      f"                                \n",
      f"\n fix BN parms              : {opt['fix_BN']}",    
      f"\n Backbone LR               : {opt['train']['backbone_lr']}",
      f"\n Backbone LR               : {opt['train']['task_lr']   }",     
      f"                                \n"
      f"\n Sharing  regularization   : {opt['train']['lambda_sharing']}",    
      f"\n Sparsity regularization   : {opt['train']['lambda_sparsity']}",  
      f"\n Task     regularization   : {opt['train']['lambda_tasks']}",
      f"\n Last Epoch                : {current_epoch} ",
      f"\n # of warm-up epochs to do : {warm_up_epochs}")


print(f"\n folder: {opt['exp_folder']}",
      f"\n layers: {opt['hidden_sizes']}",    
      f"                               \n",
      f"\n diff_sparsity_weights  : {opt['diff_sparsity_weights']}",
      f"\n skip_layer             : {opt['skip_layer']}",
      f"\n is_curriculum          : {opt['is_curriculum']}",
      f"\n curriculum_speed       : {opt['curriculum_speed']}",
      f"                              \n",    
      f"\n decay_lr_rate          : {opt['train']['decay_lr_rate']}",      
      f"\n decay_lr_freq          : {opt['train']['decay_lr_freq']}",     
      f"\n policy_decay_lr_rate   : {opt['train']['policy_decay_lr_rate']}",      
      f"\n policy_decay_lr_freq   : {opt['train']['policy_decay_lr_freq']}", 
      f"                              \n",    
      f"\n policy_lr              : {opt['train']['policy_lr']}", 
      f"\n lambda_sparsity        : {opt['train']['lambda_sparsity']}",      
      f"\n lambda_sharing         : {opt['train']['lambda_sharing']}", 
      f"                              \n",    
      f"\n lambda_tasks           : {opt['train']['lambda_tasks']}",  
      f"\n init_temp              : {opt['train']['init_temp']}",
      f"\n decay_temp             : {opt['train']['decay_temp']}",    
      f"\n decay_temp_freq        : {opt['train']['decay_temp_freq']}",   
      f"\n init_method            : {opt['train']['init_method']}", 
      f"\n init_neg_logits        : {opt['train']['init_neg_logits']}",    
      f"\n hard_sampling          : {opt['train']['hard_sampling']}",
      f"\n Warm-up epochs         : {opt['train']['warm_up_epochs']}",
      f"\n training epochs        : {opt['train']['training_epochs']}")

##---------------------------------------------------------------------
##  Training - WarmUp Phase
##---------------------------------------------------------------------
 
print(f" Last Epoch: {current_epoch}   # of warm-up epochs to do:  {warm_up_epochs} - Run epochs {current_epoch+1} to {stop_epoch_warmup}")

line_count = 0
t = tqdm(initial = current_epoch, total=stop_epoch_warmup, desc=f" Warmup training")

while current_epoch < stop_epoch_warmup:
    start_time = time.time()
    current_epoch+=1
    t.update(1)
    current_iter_w  = 0     
    #-----------------------------------------
    # Train & Update the network weights
    #-----------------------------------------   
    with trange(+1, stop_iter_w+1 , initial = 0 , total = stop_iter_w, 
                     position=0, leave= False, desc=f" Warmup Epoch {current_epoch}") as t_warmup :
        for current_iter_w in t_warmup:
            current_iter += 1            

            environ.train()    
            batch = next(train_loader)            
            environ.set_inputs(batch, train_loader.dataset.input_size)
            environ.optimize(opt['lambdas'], 
                             is_policy=False, 
                             flag='update_w', 
                             verbose = False)
        
            t_warmup.set_postfix({'curr_iter':current_iter, 
                                  'Loss': f"{environ.losses['total']['total'].item():.4f}"})
#                                   'row_ids':f"{batch['row_id'][0]}-{batch['row_id'][-1]}"})
        ##--------------------------------------------------------------- 
        ## validation
        ##--------------------------------------------------------------- 
#         if should(current_iter_w, stop_iter_w):
        trn_losses = environ.losses
        environ.print_trn_metrics(current_iter, start_time, title = f"[e] Warmup epoch:{current_epoch}    iter:")

        val_metrics = environ.evaluate(val_loader, opt['tasks'], 
                               is_policy       = False, 
                               num_train_layers= None,
                               eval_iters      = eval_iters, 
                               progress        = True,
                               leave           = False,
                               verbose         = False)

        environ.print_val_metrics(current_iter, start_time, title = f"[v] Warmup epoch:{current_epoch}    iter:")
    
        print_metrics_cr(current_epoch, time.time() - start_time, trn_losses, environ.val_metrics, line_count, out=[sys.stdout, environ.log_file]) 
        line_count += 1


t.close()
environ.save_checkpoint('warmup', current_iter)   
print_loss(current_iter, environ.val_metrics, f"[e] Warmup epoch:{current_epoch}    iter:")
environ.display_trained_policy(current_epoch,out=[sys.stdout, environ.log_file])
environ.log_file.flush()

##---------------------------------------------------------------------
##  Training - Weight / Policy Training Preparation
##---------------------------------------------------------------------
if flag_warmup:
    print_heading( f"** {timestring()} - Training iteration {current_iter}   flag: {flag} \n"
                   f"** Set optimizer and scheduler to policy_learning = True (Switch weight optimizer from ADAM to SGD)\n"
                   f"** Switch from Warm Up training to Alternate training Weights & Policy \n"
                   f"** Take checkpoint and block gradient flow through Policy net", verbose=True)
    environ.define_optimizer(policy_learning=True)
    environ.define_scheduler(policy_learning=True)
    flag_warmup = False
    environ.save_checkpoint('warmup', current_iter)
    environ.fix_alpha()
    flag = 'update_w'

train_total_epochs = 50
stop_epoch_training = current_epoch + train_total_epochs

print(f"Batches in weight epoch (stop_iter_w): {stop_iter_w}")
print(f"Batches in policy epoch (stop_iter_a): {stop_iter_a}")
print()
print(f"current_epoch          : {current_epoch}") 
print(f"current_iters          : {current_iter}")  
print(f"train_total_epochs     : {train_total_epochs}") 
print(f"stop_epoch_training    : {stop_epoch_training}")


##---------------------------------------------------------------------
##  Training - Weight / Policy Training Preparation
##---------------------------------------------------------------------
print_loss(current_iter, environ.val_metrics, title = f"[e] Last epoch:{current_epoch}    iter:")
environ.display_trained_policy(current_epoch,out=[sys.stdout, environ.log_file])

print_heading(f" Last Epoch Completed: {current_epoch}   "
              f"\n # of epochs to do :  {train_total_epochs} - Run epochs {current_epoch+1} to {stop_epoch_training}"
              f"\n policy_lr         : {opt['train']['policy_lr']}"
              f"\n lambda_sparsity   : {opt['train']['lambda_sparsity']}"
              f"\n lambda_sharing    : {opt['train']['lambda_sharing']}", verbose = True)

leave      = False
verbose    = False
line_count = 0
t = tqdm(initial = current_epoch, total=train_total_epochs, position =0, 
         leave = leave, desc=f" Alternate Weight/Policy training")

while current_epoch < stop_epoch_training:
    current_epoch+=1
    t.update(1)
    #-----------------------------------------------------------------------------------------------------------
    # Set number of layers to train based on cirriculum_speed and p_epoch (number of epochs of policy training)
    # e.g., When curriculum_speed == 3, num_train_layers is incremented  after every 3 policy training epochs
    #-----------------------------------------------------------------------------------------------------------
    num_train_layers = (p_epoch // opt['curriculum_speed']) + 1  if opt['is_curriculum'] else None

    #-----------------------------------------
    # Train & Update the network weights
    #-----------------------------------------
    if flag == 'update_w':
        start_time = time.time()

        with trange(+1, stop_iter_w+1 , initial = 0, total = stop_iter_w, position=0,
                     leave= leave, desc=f"Epoch {current_epoch} weight training") as t_weights :
            
            for current_iter_w in t_weights:    
                current_iter += 1
                environ.train()
                batch = next(train_loader)

                environ.set_inputs(batch, train1_loader.dataset.input_size)
 
                environ.optimize(opt['lambdas'], 
                                 is_policy=opt['policy'], 
                                 flag=flag, 
                                 num_train_layers=num_train_layers,
                                 hard_sampling=opt['train']['hard_sampling'],
                                 verbose = False)

                t_weights.set_postfix({'iter': current_iter, 
                                       'Loss': f"{environ.losses['losses']['total'].item():.4f}" , 
                                       'Spar': f"{environ.losses['sparsity']['total'].item():.4e}",  
                                       'Shar': f"{environ.losses['sharing']['total'].item():.4e}"})  

        #--------------------------------------------------------------------
        # validation process (here current_iter_w and stop_iter_w are equal)
        #--------------------------------------------------------------------
        trn_losses = environ.losses
        environ.print_trn_metrics(current_iter, start_time, title = f"[e] Weight trn epoch:{current_epoch}    iter:", to_display = False)

        val_metrics = environ.evaluate(val_loader, opt['tasks'], is_policy=opt['policy'],
                                       num_train_layers=num_train_layers, hard_sampling=opt['train']['hard_sampling'],
                                       eval_iters = eval_iters, progress = True, leave = leave, verbose = False)  

        environ.print_val_metrics(current_iter, start_time, title = f"[v] Weight training epoch:{current_epoch}    iter:", verbose = False)
        print_metrics_cr(current_epoch, time.time() - start_time, trn_losses, environ.val_metrics, line_count, out=[sys.stdout, environ.log_file]) 
        line_count +=1
        # Take check point:     environ.save_checkpoint('latest_weights', current_iter)
        #------------------------------------------------------------------------ 
        #  Save Best Checkpoint Code (saved below and in sparsechem_env_dev.py)
        #----------------------------------------------------------------------- 

        #-----------------------------------------------------------------------
        # END validation process 
        #-----------------------------------------------------------------------
        flag = 'update_alpha'
        environ.fix_weights()
        environ.free_alpha()
        
    #-----------------------------------------
    # Policy Training  
    #-----------------------------------------
    if flag == 'update_alpha':
        start_time = time.time()        

        with trange( +1, stop_iter_a+1 , initial = 0, total = stop_iter_a,  position=0,
                     leave= leave, desc=f"Epoch {current_epoch} policy training") as t_policy :
            for current_iter_a in t_policy:    
                current_iter += 1
                batch = next(train_loader)

                environ.set_inputs(batch, train2_loader.dataset.input_size)

                environ.optimize(opt['lambdas'], is_policy=opt['policy'], 
                                 flag=flag, num_train_layers=num_train_layers,
                                 hard_sampling=opt['train']['hard_sampling'], verbose = False)
                
                t_policy.set_postfix({'iter': current_iter, 
                                      'Loss': f"{environ.losses['losses']['total'].item():.4f}" , 
                                      'Spar': f"{environ.losses['sparsity']['total'].item():.4e}" ,  
                                      'Shar': f"{environ.losses['sharing']['total'].item():.4e}",
                                      'lyrs': f"{num_train_layers}"})    
#                                       'row_ids':f"{batch['row_id'][0]}-{batch['row_id'][-1]}"})
        #---------------------------------------------------------------------
        # print loss results (here current_iter_w and stop_iter_w are equal)
        #---------------------------------------------------------------------
        trn_losses = environ.losses
        environ.print_trn_metrics(current_iter, start_time, title = f"[e] Policy trn epoch:{current_epoch}    iter:")

        val_metrics = environ.evaluate(val_loader, opt['tasks'], is_policy=opt['policy'],
                                       num_train_layers=num_train_layers, hard_sampling=opt['train']['hard_sampling'],
                                       eval_iters = eval_iters, progress = True, leave = False, verbose = False)  

        environ.print_val_metrics(current_iter, start_time, title = f"[v] Policy training epoch:{current_epoch}    iter:", verbose = False)
        print_metrics_cr(current_epoch, time.time() - start_time, trn_losses, environ.val_metrics, line_count, out=[sys.stdout, environ.log_file])      
        line_count +=1
        p_epoch += 1        
        if should(p_epoch, opt['train']['decay_temp_freq']):
            environ.decay_temperature()
            print(f" decay gumbel softmax to {environ.gumbel_temperature}")
        
        flag = 'update_w'
        environ.fix_alpha()
        environ.free_weights(opt['fix_BN'])
        
#         print_loss(current_iter, environ.val_metrics, title = f"[e] Policy training epoch:{current_epoch}    iter:")
#         environ.display_trained_policy(current_epoch,out=[sys.stdout, environ.log_file])
    
    #-----------------------------------------
    # End Policy Training  
    #----------------------------------------- 
 
    if should(current_epoch, 5):
        environ.save_checkpoint('latest_weights_policy', current_iter)        
        print_loss(current_iter, environ.val_metrics, title = f"\n[e] Policy training epoch:{current_epoch}    iter:")
        environ.display_trained_policy(current_epoch,out=[sys.stdout, environ.log_file])
        environ.log_file.flush()
        line_count = 0

t.close()
environ.save_checkpoint('final_weights_policy', current_iter)   
print_loss(current_iter,environ.val_metrics, title = f"[e] Final epoch:{current_epoch}    iter:")
environ.display_trained_policy(current_epoch,out=[sys.stdout, environ.log_file])
environ.log_file.flush()