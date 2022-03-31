import sys
import time 
import types

import torch
import wandb 
from tqdm     import trange 

from envs.sparsechem_env import SparseChemEnv_Dev
from dataloaders.chembl_dataloader import ClassRegrSparseDataset_v3,   InfiniteDataLoader
from utils.sparsechem_utils import print_metrics_cr
from utils.util import ( makedir, print_separator, create_path, print_yaml, print_yaml2, print_loss, should, print_to,
                         fix_random_seed, read_yaml, timestring, print_heading, print_dbg, save_to_pickle, load_from_pickle,
                         print_underline, write_config_report, display_config, get_command_line_args, is_notebook) 

# DISABLE_TQDM = True

def initialize(input_args = None, build_folders = True):
    ns = types.SimpleNamespace()
    
    input_args = input_args.split() if input_args is not None else input_args

    ns.args = get_command_line_args(input_args, display = True)
         
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    print_separator('READ YAML')

    opt = read_yaml(ns.args)

    fix_random_seed(opt["random_seed"])
        
    if build_folders:
        create_path(opt)    
     
    print_heading(f" experiment name       : {opt['exp_name']} \n"
                f" experiment id         : {opt['exp_id']} \n"
                f" folder_name           : {opt['exp_folder']} \n"
                f" experiment description: {opt['exp_description']}\n"
                f" Random seeds          : {opt['seed_list']}\n"
                f" Random  seed used     : {opt['random_seed']} \n"
                f" log folder            : {opt['paths']['log_dir']}\n"
                f" checkpoint folder     : {opt['paths']['checkpoint_dir']}\n"
                f" Gpu ids               : {opt['gpu_ids']}\n"
                f" Seed index            : {ns.args.seed_idx}\n"
                f" policy_iter           : {opt['train']['policy_iter']}\n"
                f" Data Split ratios     : {opt['dataload']['x_split_ratios']}", verbose = True)

    ns.config_filename = 'run_config_seed_%04d.txt' % (opt['random_seed'])
    write_config_report(opt, filename = ns.config_filename)    
    display_config(opt)

    return opt, ns

def init_dataloaders(opt):
    ns = types.SimpleNamespace()
    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('CREATE DATALOADERS')

    ns.trainset0 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 0, verbose = False)
    ns.trainset1 = ns.trainset0
    ns.trainset2 = ns.trainset0
    # trainset1 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 1)
    # trainset2 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 2)
    ns.valset    = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 1)
    ns.testset   = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 2)

    ns.warmup_trn_loader = InfiniteDataLoader(ns.trainset0 , batch_size=opt['train']['batch_size'], num_workers = 2, pin_memory=True, collate_fn=ns.trainset0.collate, shuffle=True)
    ns.weight_trn_loader = InfiniteDataLoader(ns.trainset1 , batch_size=opt['train']['batch_size'], num_workers = 2, pin_memory=True, collate_fn=ns.trainset1.collate, shuffle=True)
    ns.policy_trn_loader = InfiniteDataLoader(ns.trainset2 , batch_size=opt['train']['batch_size'], num_workers = 2, pin_memory=True, collate_fn=ns.trainset2.collate, shuffle=True)
    ns.val_loader        = InfiniteDataLoader(ns.valset    , batch_size=opt['train']['batch_size'], num_workers = 1, pin_memory=True, collate_fn=ns.valset.collate  , shuffle=True)
    ns.test_loader       = InfiniteDataLoader(ns.testset   , batch_size=32                        , num_workers = 1, pin_memory=True, collate_fn=ns.testset.collate  , shuffle=True)

    opt['train']['weight_iter_alternate'] = opt['train'].get('weight_iter_alternate' , len(ns.weight_trn_loader))
    opt['train']['alpha_iter_alternate']  = opt['train'].get('alpha_iter_alternate'  , len(ns.policy_trn_loader))    

    return ns

def init_environment(ns, opt, is_train = True, policy_learning = False, display_cfg = False, verbose = False):
    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************
    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = SparseChemEnv_Dev(log_dir          = opt['paths']['log_dir'], 
                                checkpoint_dir   = opt['paths']['checkpoint_dir'], 
                                exp_name         = opt['exp_name'],
                                tasks_num_class  = opt['tasks_num_class'], 
                                init_neg_logits  = opt['train']['init_neg_logits'], 
                                device           = opt['gpu_ids'][0],
                                init_temperature = opt['train']['init_temp'], 
                                temperature_decay= opt['train']['decay_temp'], 
                                is_train         = is_train,
                                opt              = opt, 
                                verbose          = False)


    cfg = environ.print_configuration()
    write_config_report(opt, cfg, filename = ns.config_filename, mode = 'a')

    if display_cfg:
        print(cfg)        
    
    return environ



def init_wandb(namespace, opt, environment, resume = "allow" , log_freq  = 10):

    # wandb_run_name = opt['exp_instance']
    # opt['exp_id'] = wandb.util.generate_id()
    print(opt['exp_id'], opt['exp_name'], opt['project_name']) # , opt['exp_instance'])
    namespace.wandb_run = wandb.init(project=opt['project_name'], entity="kbardool", resume=resume, id = opt['exp_id'], name = opt['exp_name'])
    wandb.config = opt.copy()

    # wandb.watch(environ.networks['mtl-net'], log='all', log_freq=10)
    wandb.watch(environment.networks['mtl-net'], log='all', log_freq= log_freq)     ###  Weights and Biases Initialization 

    # assert wandb.run is None, "Run is still running"
    print(f" PROJECT NAME: {namespace.wandb_run.project}\n"
          f" RUN ID      : {namespace.wandb_run.id} \n"
          f" RUN NAME    : {namespace.wandb_run.name}")     
    return 



def training_prep(ns, opt, environ, dldrs, phase = 'update_w', epoch = 0, iter = 0, verbose = False):

    if torch.cuda.is_available():
        print_dbg(f" cuda available {opt['gpu_ids']}", True)
        environ.cuda(opt['gpu_ids'])
    else:
        print_dbg(f" cuda not available", verbose = True)
        environ.cpu()

    if opt['train']['print_freq'] == -1:
        print(f" set print_freq to length of train loader: {len(dldrs.warmup_trn_loader)}")
        ns.print_freq = len(dldrs.warmup_trn_loader)
    else:
        print(f" set print_freq to opt[train][print_freq]: {opt['train']['print_freq']}")
        ns.print_freq = opt['train']['print_freq']     

    if opt['train']['val_iters'] == -1:
        print(f" set eval_iters to length of val loader  : {len(dldrs.val_loader)}")
        ns.eval_iters    = len(dldrs.val_loader)    
    else:
        ns.eval_iters    = opt['train']['val_iters']
    
    ns.stop_iter_w = opt['train']['weight_iter_alternate']
    ns.stop_iter_a = opt['train']['alpha_iter_alternate'] 
        
    # Fix Alpha -     
    if phase == 'update_w':
        ns.flag = phase
        environ.fix_alpha()
        environ.free_weights(opt['fix_BN'])
    elif phase == 'update_alpha':
        ns.flag = phase
        environ.fix_weights()
        environ.free_alpha() 
    else: 
        raise ValueError('training mode/phase %s  is not valid' % phase)

    ns.current_epoch  = epoch
    ns.current_iter   = iter
 
    ns.best_results   = {}
    ns.best_metrics   = None
    ns.best_value     = 0 
    ns.best_iter      = 0
    ns.best_epoch     = 0

    ns.p_epoch        = 0
    ns.w_epoch        = 0

    ns.num_train_layers = None     
    ns.leave            = False
    ns.flag_warmup      = True

    ns.num_prints         = 0
    ns.num_blocks         = sum(environ.networks['mtl-net'].layers)
    ns.warmup_epochs      = opt['train']['warmup_epochs']
    ns.training_epochs    = opt['train']['training_epochs']
    ns.curriculum_speed   = opt['curriculum_speed'] 
    ns.curriculum_epochs  =  0
    ns.check_for_improvment_wait  = 0

    return


def retrain_prep(ns, opt, environ, dldrs, phase = 'update_w', epoch = 0, iter = 0, verbose = False):
    
    if torch.cuda.is_available():
        print_dbg(f" cuda available {opt['gpu_ids']}", verbose = verbose)
        environ.cuda(opt['gpu_ids'])
    else:
        print_dbg(f" cuda not available", verbose = verbose)
        environ.cpu()

    if opt['train']['val_iters'] == -1:
        print(f" set eval_iters to length of val loader  : {len(dldrs.val_loader)}")
        ns.eval_iters    = len(dldrs.val_loader)    
    else:
        ns.eval_iters    = opt['train']['val_iters']

        
    ns.stop_iter_w =  len(dldrs.weight_trn_loader) 

    # Fix Alpha -     
    if phase == 'update_w':
        ns.flag = phase
        environ.fix_alpha()
        environ.free_weights(opt['fix_BN'])
    elif phase == 'update_alpha':
        ns.flag = phase
        environ.fix_weights()
        environ.free_alpha() 
    else: 
        raise ValueError('training mode/phase %s  is not valid' % phase)

    # environ.define_optimizer(policy_learning=False)   
    # environ.define_scheduler(policy_learning=False)   
    ns.current_epoch  = epoch
    ns.current_iter   = iter
    ns.check_for_improvment_wait  = 0
    ns.best_results   = {}
    ns.best_metrics   = None
    ns.best_value     = 0 
    ns.best_iter      = 0
    ns.best_epoch     = 0 

    opt['train']['retrain_total_iters'] = opt['train'].get('retrain_total_iters', opt['train']['total_iters'])
    print(f"opt['train']['retrain_total_iters']:   {opt['train']['retrain_total_iters']}")
    # refer_metrics = get_reference_metrics(opt)



def warmup_phase(ns,opt, environ, dldrs, disable_tqdm = True, epochs = None):
    ns.phase = 'warmup'
    ns.flag  = 'update_weights'
    if epochs is not None:
        ns.warmup_epochs = epochs 

    ns.stop_epoch_warmup = ns.current_epoch + ns.warmup_epochs

    print_heading(f" Last Epoch: {ns.current_epoch}   # of warm-up epochs to do:  {ns.warmup_epochs} -"
                  f" Run epochs {ns.current_epoch+1} to {ns.stop_epoch_warmup}", verbose = True, out=[sys.stdout, environ.log_file])    
    if ns.current_epoch >= ns.stop_epoch_warmup:
        return

    line_count = 0
    input_size = dldrs.warmup_trn_loader.dataset.input_size

    while ns.current_epoch < ns.stop_epoch_warmup:
        start_time = time.time()
        ns.current_epoch+=1
        environ.train()    
        #-----------------------------------------
        # Train & Update the network weights
        #-----------------------------------------   
        with trange(+1, ns.stop_iter_w+1 , initial = 0 , total = ns.stop_iter_w, position=0, file=sys.stdout,
                    leave= False, disable = disable_tqdm, desc=f" Warmup Epoch {ns.current_epoch}/{ns.stop_epoch_warmup}") as t_warmup :
            for _ in t_warmup:
                ns.current_iter += 1            

                batch = next(dldrs.warmup_trn_loader)            
                environ.set_inputs(batch, input_size)

                environ.optimize(opt['lambdas'], 
                                is_policy=False, 
                                flag='update_weights', 
                                verbose = False)
            
                t_warmup.set_postfix({'curr_iter':ns.current_iter, 
                                    'Loss': f"{environ.losses['total']['total'].item():.4f}"})

        trn_losses = environ.losses
        environ.print_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Warmup Trn]")
        wandb.log(environ.losses)

        ##--------------------------------------------------------------- 
        ## validation
        ##--------------------------------------------------------------- 
        val_metrics = environ.evaluate(dldrs.val_loader,
                                        is_policy       = False, 
                                        num_train_layers= None,
                                        eval_iters      = ns.eval_iters, 
                                        disable_tqdm    = disable_tqdm,
                                        leave           = False,
                                        verbose         = False)

        environ.print_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Warmup Val]")    
        print_metrics_cr(ns.current_epoch,  time.time() - start_time, trn_losses, environ.val_metrics, line_count, 
                        out=[sys.stdout, environ.log_file], to_tqdm = True) 
        line_count += 1

        environ.schedulers['weights'].step(val_metrics['total']['total'])
        environ.schedulers['alphas'].step(val_metrics['total']['total'])            
        wandb.log(environ.val_metrics)
        
        # Checkpoint on best results
        check_for_improvement(ns,opt,environ)    
        
    wrapup_phase(ns, opt, environ)
    return 



def weight_policy_training(ns, opt, environ, dldrs, disable_tqdm = True, epochs = None, display_policy = False, verbose = False):

    ns.phase = 'train'
    if epochs is not None:
        ns.training_epochs = epochs

    ns.stop_epoch_training = ns.current_epoch + ns.training_epochs

    if opt['is_curriculum']:
        ns.curriculum_epochs = (environ.num_layers * opt['curriculum_speed'])
 

    print_heading(f" Last Epoch Completed : {ns.current_epoch}   # of epochs to run:  {ns.training_epochs}"
                  f" -->  epochs {ns.current_epoch+1} to {ns.stop_epoch_training}    \n"
                  f" policy_learning rate : {environ.opt['train']['policy_lr']}      \n"
                  f" lambda_sparsity      : {environ.opt['train']['lambda_sparsity']}\n"
                  f" lambda_sharing       : {environ.opt['train']['lambda_sharing']} \n"
                  f" curriculum training  : {opt['is_curriculum']}     cirriculum speed:"
                  f" {opt['curriculum_speed']}     num_training_layers : {ns.num_train_layers}", 
              verbose = True, out=[sys.stdout, environ.log_file])    
     
    if  ns.current_epoch >=  ns.stop_epoch_training:
        return 

    line_count = 0
    weight_input_size = dldrs.weight_trn_loader.dataset.input_size
    policy_input_size = dldrs.policy_trn_loader.dataset.input_size

    while ns.current_epoch < ns.stop_epoch_training:
        ns.current_epoch+=1

        #-----------------------------------------------------------------------------------------------------------
        # Set number of layers to train based on cirriculum_speed and p_epoch (number of epochs of policy training)
        # e.g., When curriculum_speed == 3, num_train_layers is incremented  after every 3 policy training epochs
        #-----------------------------------------------------------------------------------------------------------
        if opt['is_curriculum']:
            ns.num_train_layers =  min( (ns.p_epoch // opt['curriculum_speed']) + 1  ,  environ.num_layers)
        else:
            ns.num_train_layers =  environ.num_layers

        #-----------------------------------------
        # Train & Update the network weights
        #-----------------------------------------
        if ns.flag == 'update_weights':
            start_time = time.time()
            environ.train()
            
            with trange(+1, ns.stop_iter_w+1 , initial = 0, total = ns.stop_iter_w,  file=sys.stdout,
                        position=0, ncols = 132, leave= False, disable = disable_tqdm,
                        desc=f"Ep: {ns.current_epoch} [weights]") as t_weights :
                
                for _ in t_weights:    
                    ns.current_iter += 1
                    batch = next(dldrs.weight_trn_loader)
                    environ.set_inputs(batch , weight_input_size)

                    environ.optimize(opt['lambdas'], 
                                    is_policy=opt['policy'], 
                                    flag=ns.flag, 
                                    num_train_layers=ns.num_train_layers,
                                    hard_sampling=opt['train']['hard_sampling'],
                                    verbose = False)

                    t_weights.set_postfix({'it' : ns.current_iter, 
                                           'Lss': f"{environ.losses['task']['total'].item():.4f}" , 
                                           'Spr': f"{environ.losses['sparsity']['total'].item():.4e}",  
                                           'Shr': f"{environ.losses['sharing']['total'].item():.4e}",
                                           'lyr': f"{ns.num_train_layers}"})    
    
            trn_losses = environ.losses
            environ.print_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Trn]", to_display = False)
            wandb.log(environ.losses)
                        
            #--------------------------------------------------------------------
            # validation process (here current_iter_w and stop_iter_w are equal)
            #--------------------------------------------------------------------
            val_metrics = environ.evaluate(dldrs.val_loader,  
                                           is_policy        = opt['policy'],
                                           num_train_layers = ns.num_train_layers,
                                           hard_sampling    = opt['train']['hard_sampling'],
                                           eval_iters       = ns.eval_iters, 
                                           disable_tqdm     = disable_tqdm, 
                                           leave = False, verbose = False)  

            environ.print_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Val]", verbose = False)
            print_metrics_cr(ns.current_epoch, time.time() - start_time, trn_losses, environ.val_metrics, line_count, out=[sys.stdout, environ.log_file]) 
            line_count +=1

            environ.schedulers['weights'].step(val_metrics['task']['total'])
            
            wandb.log(environ.val_metrics)

            # Checkpoint on best results
            check_for_improvement(ns,opt,environ)                                 

            #-----------------------------------------------------------------------
            # END validation process 
            #-----------------------------------------------------------------------
            ns.flag = 'update_alpha'
            environ.fix_weights()
            environ.free_alpha()
            
        #-----------------------------------------
        # Policy Training  
        #-----------------------------------------
        if ns.flag == 'update_alpha':
            start_time = time.time()        
            environ.train()
            
            with trange( +1, ns.stop_iter_a+1 , initial = 0, total = ns.stop_iter_a,   file=sys.stdout,
                        position=0, dynamic_ncols = True, leave= False,  disable = disable_tqdm, 
                        desc=f"Ep:{ns.current_epoch} [policy] ") as t_policy :
                for _ in t_policy:    
                    ns.current_iter += 1
                    batch = next(dldrs.policy_trn_loader)
                    environ.set_inputs(batch, policy_input_size)

                    environ.optimize(opt['lambdas'], 
                                     is_policy        = opt['policy'],  
                                     flag             = ns.flag, 
                                     num_train_layers = ns.num_train_layers,
                                     hard_sampling    = opt['train']['hard_sampling'], 
                                     verbose          = False)
                    
                    t_policy.set_postfix({'it' : ns.current_iter,
                                        'Lss': f"{environ.losses['task']['total'].item():.4f}",
                                        'Spr': f"{environ.losses['sparsity']['total'].item():.4e}",
                                        'Shr': f"{environ.losses['sharing']['total'].item():.4e}",
                                        'lyr': f"{ns.num_train_layers}"})    
                                        # ,'row_ids':f"{batch['row_id'][0]}-{batch['row_id'][-1]}"})

            # print loss results - here current_iter_w and stop_iter_w are equal
            trn_losses = environ.losses
            environ.print_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Policy Trn]")
            wandb.log(environ.losses)
            
            #--------------------------------------------------------------------
            # validation process (here current_iter_a and stop_iter_a are equal)
            #--------------------------------------------------------------------        
            val_metrics = environ.evaluate(dldrs.val_loader, 
                                           is_policy        = opt['policy'],
                                           num_train_layers = ns.num_train_layers, 
                                           hard_sampling    = opt['train']['hard_sampling'],
                                           eval_iters       = ns.eval_iters, 
                                           disable_tqdm     = disable_tqdm, 
                                           leave = False, verbose = False)  

            environ.print_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Policy Val]", verbose = False)
            print_metrics_cr(ns.current_epoch, time.time() - start_time, trn_losses, environ.val_metrics, 
                             line_count, out=[sys.stdout, environ.log_file])      
            line_count +=1

            environ.schedulers['alphas'].step(val_metrics['total']['total'])
            wandb.log(environ.val_metrics)

            # Checkpoint on best results
            check_for_improvement(ns,opt,environ)    
                                         
            #-----------------------------------------------------------------------
            # END validation process 
            #-----------------------------------------------------------------------    
            
            ns.p_epoch += 1        
            if should(ns.p_epoch, opt['train']['decay_temp_freq']):
                environ.decay_temperature()
                print(f" decay gumbel softmax to {environ.gumbel_temperature}")
            
            ns.flag = 'update_weights'
            environ.fix_alpha()
            environ.free_weights(opt['fix_BN'])
            
    #         environ.display_trained_logits(current_epoch)        
    #         print_loss(current_epoc, current_iter, environ.val_metrics, title = f"[Policy trn]  ep:{current_epoch}   it:{current_iter}")
        
        #-----------------------------------------
        # End Policy Training  
        #----------------------------------------- 
        if should(ns.current_epoch, 5):
            environ.save_checkpoint('model_latest_weights_policy', ns.current_iter, ns.current_epoch)        
            print_loss(environ.val_metrics, title = f"\n[e] Policy training epoch:{ns.current_epoch}  it:{ns.current_iter}",
                      out=[sys.stdout, environ.log_file])
            environ.display_trained_policy(ns.current_epoch,out=[sys.stdout, environ.log_file])
            environ.log_file.flush()
            line_count = 0
        else:
            if display_policy:
                environ.display_trained_policy(ns.current_epoch,out=[sys.stdout, environ.log_file])
            
    wrapup_phase(ns, opt, environ)
    return



def retrain_phase(ns, opt, environ, dldrs, epochs = None, disable_tqdm = True,
                  display_policy = False, verbose = False):
    ns.phase = 'retrain'
    if epochs is not None:
        ns.training_epochs = epochs

    ns.stop_epoch_training = ns.current_epoch + ns.training_epochs

    print_heading(f" Last Epoch Completed: {ns.current_epoch}   # of epochs to do:  {ns.training_epochs} -  epochs {ns.current_epoch+1} to {ns.stop_epoch_training}"
                f"\n stop_iter_w         : {ns.stop_iter_w}"
                f"\n policy_lr           : {opt['train']['policy_lr']}"
                f"\n lambda_sparsity     : {opt['train']['lambda_sparsity']}"
                f"\n lambda_sharing      : {opt['train']['lambda_sharing']}", verbose = True)

    if  ns.current_epoch >=  ns.stop_epoch_training:
        return 
    line_count = 0
    weight_input_size = dldrs.weight_trn_loader.dataset.input_size
    # policy_input_size = dldrs.policy_trn_loader.dataset.input_size

    while (ns.current_epoch < ns.stop_epoch_training):
        ns.current_epoch+=1    
        start_time = time.time()

        with trange(+1, ns.stop_iter_w+1 , initial = 0, total = ns.stop_iter_w, position=0,  file=sys.stdout,
                        ncols = 132, leave= False, disable = disable_tqdm, 
                        desc=f"Epoch {ns.current_epoch} weight training") as t_weights :
            for _ in t_weights:    
                ns.current_iter += 1
                environ.train()
                batch = next(dldrs.weight_trn_loader)
                environ.set_inputs(batch, weight_input_size )
        
                environ.optimize_fix_policy(opt['lambdas']) 
    #           is_policy=opt['policy'], flag=flag, num_train_layers=num_train_layers,
    #            hard_sampling=opt['train']['hard_sampling'], verbose = False)

                t_weights.set_postfix({'it' : ns.current_iter, 
                                    'Lss': f"{environ.losses['task']['total'].item():.4f}" , 
                                    'Spr': f"{environ.losses['sparsity']['total'].item():.4e}",  
                                    'Shr': f"{environ.losses['sharing']['total'].item():.4e}"})  

            trn_losses = environ.losses
            environ.print_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Trn]", to_display = False)
            wandb.log(environ.losses)

        # validation   
        # val_metrics = eval_fix_policy(environ = environ, dataloader = val_loader, tasks = opt['tasks'], num_seg_cls = num_seg_class)    
        val_metrics = environ.evaluate(dldrs.val_loader, 
                                        is_policy        = True,
                                        policy_sampling_mode = 'fix_policy',
                                        hard_sampling    = opt['train']['hard_sampling'],
                                        eval_iters       = ns.eval_iters, 
                                        disable_tqdm     = disable_tqdm, 
                                        leave = False, verbose = False)      

    #     print_loss(environ.val_metrics, title = f"[Retrain] ep:{current_epoch}  it:{current_iter}")
        environ.print_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Val]", verbose = False)
        print_metrics_cr(ns.current_epoch, time.time() - start_time, trn_losses, environ.val_metrics, line_count, out=[sys.stdout, environ.log_file]) 
        line_count +=1        
        wandb.log(environ.val_metrics)
        
    wrapup_phase(ns, opt, environ)
    return 


def wrapup_phase(ns, opt, environ, label = None):
    label = ns.phase if label is None else label

    # ns.model_label   = 'model_%s_ep_%d_seed_%04d'  % (label, ns.current_epoch, opt['random_seed'])
    # ns.metrics_label = 'metrics_%s_ep_%d_seed_%04d.pickle' % (label,ns.current_epoch, opt['random_seed'])
    ns.model_label   = 'model_%s_ep_%d'  % (label, ns.current_epoch)
    ns.metrics_label = 'metrics_%s_ep_%d.pickle' % (label,ns.current_epoch)
    environ.save_checkpoint(ns.model_label, ns.current_iter, ns.current_epoch) 
    save_to_pickle(environ.val_metrics, environ.opt['paths']['checkpoint_dir'], ns.metrics_label)
    print_loss(environ.val_metrics, title = f"[Final] ep:{ns.current_epoch}  it:{ns.current_iter}",)
    environ.display_trained_policy(ns.current_epoch,out=[sys.stdout, environ.log_file])
    environ.display_trained_logits(ns.current_epoch,out=[sys.stdout, environ.log_file])
    print_to(f" save {label} val_metrics to :  {ns.model_label}", out=[sys.stdout, environ.log_file])
    print_to(f" save {label} checkpoint  to :  {ns.model_label}", out=[sys.stdout, environ.log_file])    
    environ.log_file.flush()
    return 



def check_for_improvement(ns,opt,environ):
    #------------------------------------------------------------------------ 
    #  Save Best Checkpoint Code (saved below and in sparsechem_env_dev.py)
    #----------------------------------------------------------------------- 
    ## ns.curriculum_epochs = (environ.num_layers * opt['curriculum_speed']) 

    # if (ns.current_epoch - opt['train']['warmup_epochs']) >= ns.curriculum_epochs:    
    if (ns.current_epoch - ns.check_for_improvment_wait) >= ns.curriculum_epochs:    

        if environ.val_metrics['aggregated']['avg_prec_score'] > ns.best_value:
            print('Previous best_epoch: %5d   best iter: %5d,   best_value: %.5f' % (ns.best_epoch, ns.best_iter, ns.best_value))        
            ns.best_value   = environ.val_metrics['aggregated']['avg_prec_score']
            ns.best_metrics = environ.val_metrics
            ns.best_iter    = ns.current_iter
            ns.best_epoch   = ns.current_epoch
            model_label     = 'model_best_seed_%04d' % (opt['random_seed'])
            environ.save_checkpoint(model_label, ns.current_iter, ns.current_epoch) 
            print('New      best_epoch: %5d   best iter: %5d,   best_value: %.5f' % (ns.best_epoch, ns.best_iter, ns.best_value))        
            metrics_label = 'metrics_best_seed_%04d.pickle' % (opt['random_seed'])
            save_to_pickle(environ.val_metrics, environ.opt['paths']['checkpoint_dir'], metrics_label)    
    return


def disp_dataloader_info(dldrs):
    """ display dataloader information"""
    print(f"\n trainset.y_class                                   :  {[ i.shape  for i in dldrs.trainset0.y_class_list]}",
          f"\n trainset1.y_class                                  :  {[ i.shape  for i in dldrs.trainset1.y_class_list]}",
          f"\n trainset2.y_class                                  :  {[ i.shape  for i in dldrs.trainset2.y_class_list]}",
          f"\n valset.y_class                                     :  {[ i.shape  for i in dldrs.valset.y_class_list  ]} ",
          f"\n testset.y_class                                    :  {[ i.shape  for i in dldrs.testset.y_class_list  ]} ",
          f"\n                                ",
          f'\n size of training set 0 (warm up)                   :  {len(dldrs.trainset0)}',
          f'\n size of training set 1 (network parms)             :  {len(dldrs.trainset1)}',
          f'\n size of training set 2 (policy weights)            :  {len(dldrs.trainset2)}',
          f'\n size of validation set                             :  {len(dldrs.valset)}',
          f'\n size of test set                                   :  {len(dldrs.testset)}',
          f'\n                               Total                :  {len(dldrs.trainset0)+len(dldrs.trainset1)+len(dldrs.trainset2)+len(dldrs.valset)+ len(dldrs.testset)}',
          f"\n                                ",
          f"\n lenght (# batches) in training 0 (warm up)         :  {len(dldrs.warmup_trn_loader)}",
          f"\n lenght (# batches) in training 1 (network parms)   :  {len(dldrs.weight_trn_loader)}",
          f"\n lenght (# batches) in training 2 (policy weights)  :  {len(dldrs.policy_trn_loader)}",
          f"\n lenght (# batches) in validation dataset           :  {len(dldrs.val_loader)}",
          f"\n lenght (# batches) in test dataset                 :  {len(dldrs.test_loader)}",
          f"\n                                ")
                
def disp_info_1(ns, opt, environ):
    print(
            f"\n Num_blocks                : {sum(environ.networks['mtl-net'].layers)}"    
            f"                                \n"
            f"\n batch size                : {opt['train']['batch_size']}",    
            f"\n batches/ Weight trn epoch : {ns.stop_iter_w}",
            f"\n batches/ Policy trn epoch : {ns.stop_iter_a}",
            # f"\n Total iterations          : {opt['train']['total_iters']}",
            # f"\n Warm-up iterations        : {opt['train']['warm_up_iters']}",
            f"                                \n"
            f"\n Print Frequency           : {opt['train']['print_freq']}",
            f"\n Config Val Frequency      : {opt['train']['val_freq']}",
            f"\n Config Val Iterations     : {opt['train']['val_iters']}",
            f"\n Val iterations            : {ns.eval_iters}",
            f"\n which_iter                : {opt['train']['which_iter']}",
            f"\n train_resume              : {opt['train']['resume']}",
            f"                                \n",                     
            f"\n fix BN parms              : {opt['fix_BN']}",    
            f"\n Task LR                   : {opt['train']['task_lr']   }",     
            f"\n Backbone LR               : {opt['train']['backbone_lr']}",
            f"                                \n"
            f"\n Sharing  regularization   : {opt['train']['lambda_sharing']}",    
            f"\n Sparsity regularization   : {opt['train']['lambda_sparsity']}",  
            f"\n Task     regularization   : {opt['train']['lambda_tasks']}",
            f"                                \n"
            f"\n Current epoch             : {ns.current_epoch} ",
            f"\n Warm-up epochs            : {opt['train']['warmup_epochs']}",
            f"\n Training epochs           : {opt['train']['training_epochs']}",
            # f"                                \n",
            # f"\n # of warm-up epochs to do : {ns.warmup_epochs}",
            # f"\n Warm-up stop              : {ns.stop_epoch_warmup}",
            # f"\n training_epochs           : {ns.training_epochs}",
            # f"\n stop_iter_w               : {ns.stop_iter_w}",
        )

def disp_gpu_info():
    from GPUtil import showUtilization as gpu_usage
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
        ## current GPU memory managed by caching allocator in bytes for a given device, 
        ## in previous PyTorch versions the command was torch.cuda.memory_cached
        print('   Reserved  : ', torch.cuda.memory_reserved(i) )   
        print()

    gpu_usage()                             