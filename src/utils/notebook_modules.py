import os 
import sys
import time 
import types
import numpy as np
import torch
import wandb 
from tqdm        import trange 
from envs        import SparseChemEnv
from dataloaders import ClassRegrSparseDataset_v3,   InfiniteDataLoader
from utils       import ( makedir, print_separator, create_path, print_yaml, print_yaml2, print_loss, should, print_to,
                         fix_random_seed, read_yaml, timestring, print_heading, print_dbg, save_to_pickle, load_from_pickle,
                         print_underline, write_config_report, display_config, get_command_line_args, is_notebook, 
                         load_sparse, print_metrics_cr) 

# DISABLE_TQDM = True

def initialize(ns, build_folders = True, start_wandb = True):
         
    print_separator('READ YAML')
    opt = read_yaml(ns.args)
    fix_random_seed(opt["random_seed"])

    # print(f" cuda_devices : {ns.args.cuda_devices}")
    # os.environ["CUDA_VISIBLE_DEVICES"]=ns.args.cuda_devices

    # torch.set_device(opt['pytorch_threads'])
    print(f" Pytorch thread count: {torch.get_num_threads()}")
    print(f" Set Pytorch thread count to : {opt['pytorch_threads']}")
    torch.set_num_threads(opt['pytorch_threads'])
    print(f" Pytorch thread count set to : {torch.get_num_threads()}")
    
    if start_wandb:
        init_wandb(ns, opt)
        
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

    # ns.config_filename = 'run_config_seed_%04d.txt' % (opt['random_seed'])
    ns.config_filename = 'run_configuration.txt'  
    write_config_report(opt, filename = ns.config_filename)    
    display_config(opt)

    return opt


def init_wandb(ns, opt, resume = "allow", verbose=False ):
    if wandb.run is not None:
        print(f" End in-flight wandb run . . .")
        wandb.finish()

    # opt['exp_id'] = wandb.util.generate_id()
    # print_dbg(f"{opt['exp_id']}, {opt['exp_name']}, {opt['project_name']}", verbose) 
    
    ns.wandb_run = wandb.init(project=opt['project_name'], 
                                     entity="kbardool", 
                                     id = opt['exp_id'], 
                                     name = opt['exp_name'],
                                     notes = opt['exp_description'],
                                     resume=resume )
    wandb.config.update(ns.args)
    wandb.config.update(opt,allow_val_change=True)   ## wandb.config = opt.copy()

    # wandb.watch(environ.networks['mtl-net'], log='all', log_freq=10)
    wandb.define_metric("best_accuracy", summary="last")
    wandb.define_metric("best_epoch", summary="last")
    wandb.define_metric("best_iter", summary="last")

    # assert wandb.run is None, "Run is still running"
    print(f" WandB Initialization -----------------------------------------------------------\n"
          f" PROJECT NAME: {ns.wandb_run.project}\n"
          f" RUN ID      : {ns.wandb_run.id} \n"
          f" RUN NAME    : {ns.wandb_run.name}\n"     
          f" --------------------------------------------------------------------------------")
    return 

def init_dataloaders(opt, verbose = False):
    dldrs = types.SimpleNamespace()
    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('CREATE DATALOADERS')

    dldrs.trainset0 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 0, verbose = verbose)
    dldrs.trainset1 = dldrs.trainset0
    dldrs.trainset2 = dldrs.trainset0
    # trainset1 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 1)
    # trainset2 = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 2)
    dldrs.valset    = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 1)
    dldrs.testset   = ClassRegrSparseDataset_v3(opt, split_ratios = opt['dataload']['x_split_ratios'], ratio_index = 2)

    dldrs.warmup_trn_loader = InfiniteDataLoader(dldrs.trainset0 , batch_size=opt['train']['batch_size'], num_workers = 2, pin_memory=True, collate_fn=dldrs.trainset0.collate, shuffle=True)
    dldrs.weight_trn_loader = InfiniteDataLoader(dldrs.trainset1 , batch_size=opt['train']['batch_size'], num_workers = 2, pin_memory=True, collate_fn=dldrs.trainset1.collate, shuffle=True)
    dldrs.policy_trn_loader = InfiniteDataLoader(dldrs.trainset2 , batch_size=opt['train']['batch_size'], num_workers = 2, pin_memory=True, collate_fn=dldrs.trainset2.collate, shuffle=True)
    dldrs.val_loader        = InfiniteDataLoader(dldrs.valset    , batch_size=opt['train']['batch_size'], num_workers = 1, pin_memory=True, collate_fn=dldrs.valset.collate  , shuffle=True)
    dldrs.test_loader       = InfiniteDataLoader(dldrs.testset   , batch_size=32                        , num_workers = 1, pin_memory=True, collate_fn=dldrs.testset.collate  , shuffle=True)

    opt['train']['warmup_iter_alternate'] = opt['train'].get('warmup_iter_alternate' , len(dldrs.weight_trn_loader))
    opt['train']['weight_iter_alternate'] = opt['train'].get('weight_iter_alternate' , len(dldrs.weight_trn_loader))
    opt['train']['alpha_iter_alternate']  = opt['train'].get('alpha_iter_alternate'  , len(dldrs.policy_trn_loader))    

    return dldrs

 

def init_dataloaders_by_fold_id(opt, warmup_folds=None, weight_folds = None, policy_folds = None, validation_folds= None, verbose = False):

    dldrs = types.SimpleNamespace()
    ## Identify indicies corresponding to =fold_va and !=fold_va
    ## These indices are passed to the ClassRegrSparseDataset 

    # ecfp     = load_sparse(opt['dataload']['dataroot'], opt['dataload']['x'])
    # folding  = np.load(os.path.join(opt['dataload']['dataroot'], opt['dataload']['folding']))
    # print(ecfp.shape, folding.shape)
    warmup_folds   = opt['dataload']['fold_warmup']  if warmup_folds is None else warmup_folds   
    weight_folds   = opt['dataload']['fold_weights'] if weight_folds is None else weight_folds   
    policy_folds   = opt['dataload']['fold_policy']  if policy_folds is None else policy_folds   
    validation_folds = opt['dataload']['fold_va']  if validation_folds is None else validation_folds 
    print(f" Warmup folds    : {warmup_folds}")
    print(f" Weights folds   : {weight_folds}")
    print(f" Policy folds    : {policy_folds}")
    print(f" Validation folds: {validation_folds}")
    
    dldrs = types.SimpleNamespace()
    dldrs.trainset0 = ClassRegrSparseDataset_v3(opt, folds= warmup_folds, verbose = verbose)
    dldrs.trainset1 = ClassRegrSparseDataset_v3(opt, folds= weight_folds, verbose = verbose)
    dldrs.trainset2 = ClassRegrSparseDataset_v3(opt, folds= policy_folds, verbose = verbose)
    dldrs.valset    = ClassRegrSparseDataset_v3(opt, folds= validation_folds, verbose = verbose)
    # dldrs.trainset1 = dldrs.trainset0
    # dldrs.trainset2 = dldrs.trainset0


    dldrs.warmup_trn_loader = InfiniteDataLoader(dldrs.trainset0 , batch_size=opt['train']['batch_size'], num_workers = 1, pin_memory=True, collate_fn=dldrs.trainset0.collate, shuffle=True)
    dldrs.weight_trn_loader = InfiniteDataLoader(dldrs.trainset1 , batch_size=opt['train']['batch_size'], num_workers = 1, pin_memory=True, collate_fn=dldrs.trainset1.collate, shuffle=True)
    dldrs.policy_trn_loader = InfiniteDataLoader(dldrs.trainset2 , batch_size=opt['train']['batch_size'], num_workers = 1, pin_memory=True, collate_fn=dldrs.trainset2.collate, shuffle=True)
    dldrs.val_loader        = InfiniteDataLoader(dldrs.valset    , batch_size=opt['train']['batch_size'], num_workers = 1, pin_memory=True, collate_fn=dldrs.valset.collate   , shuffle=True)
    
    # dldrs.test_loader       = InfiniteDataLoader(dldrs.testset   , batch_size=32                        , num_workers = 1, pin_memory=True, collate_fn=dldrs.testset.collate  , shuffle=True)
    # opt['train']['weight_iter_alternate'] = opt['train'].get('weight_iter_alternate' , len(dldrs.weight_trn_loader))
    # opt['train']['alpha_iter_alternate']  = opt['train'].get('alpha_iter_alternate'  , len(dldrs.policy_trn_loader))        
    # opt['train']['val_iters']             = opt['train'].get('val_iters'             , len(dldrs.val_loader))       

    opt['train']['warmup_iter_alternate'] = len(dldrs.warmup_trn_loader) if opt['train']['warmup_iter_alternate'] == -1 else opt['train']['warmup_iter_alternate']
    opt['train']['weight_iter_alternate'] = len(dldrs.weight_trn_loader) if opt['train']['weight_iter_alternate'] == -1 else opt['train']['weight_iter_alternate']
    opt['train']['alpha_iter_alternate']  = len(dldrs.policy_trn_loader) if opt['train']['alpha_iter_alternate']  == -1 else opt['train']['alpha_iter_alternate'] 
    opt['train']['val_iters']             = len(dldrs.val_loader)        if opt['train']['val_iters']             == -1 else opt['train']['val_iters']            
    
    print(f" dataloader preparation - set number of batches per warmup training epoch to: {opt['train']['warmup_iter_alternate']}")
    print(f" dataloader preparation - set number of batches per weight training epoch to: {opt['train']['weight_iter_alternate']}")
    print(f" dataloader preparation - set number of batches per policy training epoch to: {opt['train']['alpha_iter_alternate']}")
    print(f" dataloader preparation - set number of batches per validation to           : {opt['train']['val_iters']}")
    
    return dldrs


def disp_dataloader_info(dldrs):
    """ display dataloader information"""
    print(f"\n trainset.y_class                                   :  {[ i.shape  for i in dldrs.trainset0.y_class_list]}",
          f"\n trainset1.y_class                                  :  {[ i.shape  for i in dldrs.trainset1.y_class_list]}",
          f"\n trainset2.y_class                                  :  {[ i.shape  for i in dldrs.trainset2.y_class_list]}",
          f"\n valset.y_class                                     :  {[ i.shape  for i in dldrs.valset.y_class_list  ]}  \n\n",
          f"                               Total                :  {len(dldrs.trainset0)+len(dldrs.trainset1)+len(dldrs.trainset2)+len(dldrs.valset)} \n")
        #   f"\n testset.y_class                                    :  {[ i.shape  for i in dldrs.testset.y_class_list  ]} ",
        #   f'\n size of test set                                   :  {len(dldrs.testset)}',
        #   f'\n                               Total                :  {len(dldrs.trainset0)+len(dldrs.trainset1)+len(dldrs.trainset2)+len(dldrs.valset)+ len(dldrs.testset)}',
        #   f"\n lenght (# batches) in test dataset                 :  {len(dldrs.test_loader)}",

    print_underline(f"Training dataset :", verbose=True)
    print(f"  Size of training set 0 (warm up)                   :  {len(dldrs.trainset0)} \n"
          f"  Number of batches in training 0 (warm up)          :  {len(dldrs.warmup_trn_loader)} \n"
          f"  Size of training set 1 (network parms)             :  {len(dldrs.trainset1)} \n"
          f"  Number of batches in training 1 (network parms)    :  {len(dldrs.weight_trn_loader)} \n"
          f"  Size of training set 2 (policy weights)            :  {len(dldrs.trainset2)} \n"
          f"  Number of batches in training 2 (policy weights)   :  {len(dldrs.policy_trn_loader)} \n"
          f"  training set num of positive                       :  {dldrs.trainset0.num_pos.sum()} \n"
          f"  training set num of negative                       :  {dldrs.trainset0.num_neg.sum()} \n"
          f"  task_weights_list[0].aggregation_weight sum        :  {dldrs.trainset0.tasks_weights_list[0].aggregation_weight.sum()}\n")

    print_underline(f"Validation dataset :",verbose=True) 
    print(f"  Rows in dataset                                    : {len(dldrs.valset)}\n"
          f"  Number of batches in dataset                       : {len(dldrs.val_loader)}\n"
          f"  validation set num of positive                     : {dldrs.valset.num_pos.sum()}\n"
          f"  validation set num of negative                     : {dldrs.valset.num_neg.sum()}\n"
          f"  task_weights_list[0].aggregation_weight sum        : {dldrs.valset.tasks_weights_list[0].aggregation_weight.sum()}\n")

def init_environment(ns, opt, is_train = True, display_cfg = False, verbose = False):
    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************
    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = SparseChemEnv(opt = opt, 
                            is_train         = is_train,

                            init_neg_logits  = opt['train']['init_neg_logits'], 
                            init_temperature = opt['train']['init_temp'], 
                            temperature_decay= opt['train']['decay_temp'], 
                            verbose          = False)

    wandb_watch(environ.networks['mtl-net'], log='all', log_freq= 1000)     ###  Weights and Biases Initialization         
    cfg = environ.print_configuration()
    write_config_report(opt, cfg, filename = ns.config_filename, mode = 'a')

    if display_cfg:
        print(cfg)        
    
    return environ

def wandb_watch(item = None, log = 'all', log_freq = 1000):
    """
    Note: Increasing the log frequency can result in longer run times
    """
    if item is not None:
        wandb.watch(item, log='all', log_freq= log_freq)     ###  Weights and Biases Initialization         

def wandb_log_metrics(val_metrics, step = None):

    wandb.log({ **val_metrics['parms'], 
                **val_metrics['aggregated'],
                'ERRORS'    : {**val_metrics['total']}, 
                'SHARING'   : {**val_metrics['sharing']}, 
                'SPARSITY'  : {**val_metrics['sparsity']},
                'epoch'     : val_metrics['epoch']}, step = step)

    # wandb.log({'epoch': val_metrics['epoch']}, step = step)


def check_for_resume_training(ns, opt, environ, epoch = 0 , iter = 0):
    ## TODO: Remove hard coded RESUME_MODEL_CKPT and RESUME_METRICS_CKPT
    ns.loaded_epoch, ns.loaded_iter = None, None
    ns.val_metrics   = {}
    ns.best_metrics  = {}
    ns.best_accuracy = 0
    ns.best_roc_auc  = 0  
    ns.best_iter     = 0
    ns.best_epoch    = 0    
    ns.current_epoch  = epoch
    ns.current_iter   = iter
    
    print(f"opt['train']['which_iter'] :  {opt['train']['which_iter']}")
    
    if opt['train']['resume']:
        RESUME_MODEL_CKPT = ""
        RESUME_METRICS_CKPT = ""    
        # opt['train']['which_iter'] = 'warmup_ep_40_seed_0088'
        print_separator('Resume training')
        print(opt['train']['which_iter'])
        print(f" Resume training from folder : {opt['resume_path']}")
        print(f" Resume label is             : {opt['resume_ckpt']}")
        print(f" Resume metrics filename     : {opt['resume_metrics']}")
        ns.loaded_iter, ns.loaded_epoch = environ.load_checkpoint(opt['resume_ckpt'], path = opt['resume_path'], verbose = True)

        ns.current_epoch  = ns.loaded_epoch
        ns.current_iter   = ns.loaded_iter
        opt['train']['retrain_total_iters'] = opt['train'].get('retrain_total_iters', opt['train']['total_iters'])

        print(f" Checkpoint loaded - loaded epoch:{ns.loaded_epoch}   loaded iteration:{ns.loaded_iter}")    
        print(f"opt['train']['retrain_total_iters']:   {opt['train']['retrain_total_iters']}")        
        ## In resume, we DO NOT RESET LOGITS
        environ.display_trained_policy(ns.current_epoch)
        environ.display_trained_logits(ns.current_epoch)
        
        # ns.val_metrics = load_from_pickle(path=opt['paths']['checkpoint_dir'], filename=RESUME_METRICS_CKPT)
        # ns.val_metrics = load_from_pickle(path=opt['resume_path'], filename=opt['resume_metrics'])
        # training_prep(ns, opt, environ, dldrs, epoch = loaded_epoch, iter = loaded_iter )
        
        loaded_metrics   = load_from_pickle(path=opt['resume_path'], filename=opt['resume_metrics'])

        if 'val_metrics' not in loaded_metrics:
            print("old style")
            ns.val_metrics = loaded_metrics 
        else:
            print("new style")
            print(loaded_metrics.keys())
            ns.val_metrics   = loaded_metrics['val_metrics'] 
            ns.best_metrics  = loaded_metrics.get('best_metrics' , {})
            ns.best_accuracy = loaded_metrics.get('best_accuracy', 0)
            ns.best_roc_auc  = loaded_metrics.get('best_roc_auc' , 0)  
            ns.best_iter     = loaded_metrics.get('best_iter'    , 0)
            ns.best_epoch    = loaded_metrics.get('best_epoch'   , 0)            
            print('Resume mode - load successful!!')
            print(f"ns.best_accuracy:   {ns.best_accuracy}")
            print(f"ns.best_roc_auc :   {ns.best_roc_auc }")
            print(f"ns.best_iter    :   {ns.best_iter    }")
            print(f"ns.best_epoch   :   {ns.best_epoch   }")
    else:
        print_separator('Initiate Training from scratch ')


def model_initializations(ns, opt, environ, phase = 'update_weights', policy_learning = False, verbose = False):
    environ.define_optimizer(policy_learning=policy_learning, verbose = verbose)
    print(f" Model optimizers defined . . . policy_learning: {policy_learning}")

    environ.define_scheduler(policy_learning=policy_learning, verbose = verbose)
    print(f" Model schedulers defined . . . policy_learning: {policy_learning}")
    
    environ.write_metrics_csv_heading()    
    print(" Metrics CSV file header written . . . ")
    
    # model_fix_weights(ns, opt, environ, phase = phase)
    print(" Model initializations complete . . . ")


def model_fix_weights(ns, opt, environ, phase):
    # Fix Alpha -     
    if phase == 'update_weights':
        print(' Allow weight updates -- fix alpha')
        ns.flag = phase
        environ.fix_alpha()
        environ.free_weights(opt['fix_BN'])
    elif phase == 'update_alpha':
        print(' Update alpha updates -- fix weights')
        ns.flag = phase
        environ.fix_weights()
        environ.free_alpha() 
    else: 
        raise ValueError('training mode/phase %s  is not valid' % phase)


def training_initializations(ns, opt, environ, dldrs, 
                             warmup,
                             warmup_iterations = 0, 
                             weight_iterations = 0,
                             policy_iterations = 0,
                             eval_iterations   = 0,
                             write_checkpoint  = None,
                             epoch = 0, iter = 0, verbose = False):

    if torch.cuda.is_available():
        print_dbg(f" training preparation: - check for CUDA - cuda available as device id: {opt['gpu_ids']}", True)
        environ.cuda(opt['gpu_ids'])
    else:
        print_dbg(f" training preparation: - check for CUDA - cuda not available", verbose = True)
        environ.cpu()

    ns.print_freq  = opt['train']['print_freq'] if opt['train']['print_freq'] != -1  else len(dldrs.warmup_trn_loader)

    ns.trn_iters_warmup = opt['train']['warmup_iter_alternate'] if warmup_iterations == 0 else warmup_iterations
    ns.trn_iters_weights = opt['train']['weight_iter_alternate'] if weight_iterations == 0 else weight_iterations
    ns.trn_iters_policy = opt['train']['alpha_iter_alternate']  if policy_iterations == 0 else policy_iterations
    ns.eval_iters  = opt['train']['val_iters']  if eval_iterations   == 0 else eval_iterations

    print(f" training preparation: - set print_freq to                                 : {ns.print_freq} ")
    print(f" training preparation: - set number of batches per warmup training epoch to: {ns.trn_iters_warmup}")
    print(f" training preparation: - set number of batches per weight training epoch to: {ns.trn_iters_weights}")
    print(f" training preparation: - set number of batches per policy training epoch to: {ns.trn_iters_policy}")
    print(f" training preparation: - set number of batches per validation to           : {ns.eval_iters }")
 
    ns.p_epoch            = 0
    ns.num_train_layers   = None     
    ns.leave              = False   
    # ns.flag_warmup        = warmup

    # ns.num_prints         = 0
    ns.num_blocks         = sum(environ.networks['mtl-net'].layers)
    ns.warmup_epochs      = opt['train']['warmup_epochs']
    ns.training_epochs    = opt['train']['training_epochs']
    ns.curriculum_speed   = opt['curriculum_speed'] 
    
    # Wait periods before starting to check for performance improvement
    ns.check_for_improvment_wait  = 0
    if opt['is_curriculum']:
        ns.curriculum_epochs = (environ.num_layers * opt['curriculum_speed'])
    else:
        ns.curriculum_epochs  = 0

    if write_checkpoint is not None:
        ns.write_checkpoint = write_checkpoint
    else:
        ns.write_checkpoint   = True
    print(f" training preparation complete . . .")
    return





def warmup_phase(ns,opt, environ, dldrs, disable_tqdm = True, epochs = None, verbose = False):
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
   
    environ.fix_alpha()
    environ.free_weights(opt['fix_BN'])
   
    while ns.current_epoch < ns.stop_epoch_warmup:
        start_time = time.time()
        ns.current_epoch+=1

        environ.train()    
        #-----------------------------------------
        # Train & Update the network weights
        #-----------------------------------------   
        with trange(+1, ns.trn_iters_warmup+1 , initial = 0 , total = ns.trn_iters_warmup, position=0, file=sys.stdout,
                    leave= False, disable = disable_tqdm, desc=f" Warmup Epoch {ns.current_epoch}/{ns.stop_epoch_warmup}") as t_warmup :
            for _ in t_warmup:
                ns.current_iter += 1            

                batch = next(dldrs.warmup_trn_loader)            
                environ.set_inputs(batch, input_size)

                environ.optimize(is_policy=False, 
                                 flag='update_weights', 
                                 verbose = verbose)
            
                t_warmup.set_postfix({'curr_iter':ns.current_iter, 
                                    'Loss': f"{environ.losses['total']['total'].item():.4f}"})

        ns.trn_losses = environ.losses
        environ.write_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Warmup Trn]", to_tb=False)
        # wandb.log(environ.losses, step = ns.current_epoch)

        ##--------------------------------------------------------------- 
        ## validation
        ##--------------------------------------------------------------- 
        ns.val_metrics = environ.evaluate(dldrs.val_loader,
                                        is_policy       = False, 
                                        num_train_layers= None,
                                        eval_iters      = ns.eval_iters, 
                                        disable_tqdm    = disable_tqdm,
                                        leave           = False,
                                        verbose         = False)

        # environ.write_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Warmup Val]")    
        print_metrics_cr(ns.current_epoch,  time.time() - start_time, ns.trn_losses, ns.val_metrics,
                         line_count, out=[sys.stdout, environ.log_file]) 
        line_count +=1
        wandb_log_metrics(ns.val_metrics, step = ns.current_epoch)

        # environ.schedulers['weights'].step(ns.val_metrics['total']['task'])
        # environ.schedulers['alphas'].step(ns.val_metrics['total']['task'])            
        
        # Checkpoint on best results
        check_for_improvement(ns,opt,environ)    
        
    wrapup_phase(ns, opt, environ)
    return 


def weight_policy_training(ns, opt, environ, dldrs, disable_tqdm = True, epochs = None, display_policy = False, verbose = False):

    ns.phase = 'train'
    if epochs is not None:
        ns.training_epochs = epochs

    ns.stop_epoch_training = ns.current_epoch + ns.training_epochs


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
        # Train Network Weights
        #-----------------------------------------
        if ns.flag == 'update_weights':
            start_time = time.time()
            environ.fix_alpha()
            environ.free_weights(opt['fix_BN'])
            environ.train()
            
            with trange(+1, ns.trn_iters_weights+1 , initial = 0, total = ns.trn_iters_weights,  file=sys.stdout,
                        position=0, ncols = 132, leave= False, disable = disable_tqdm,
                        desc=f"Ep: {ns.current_epoch} [weights]") as t_weights :
                
                for _ in t_weights:    
                    ns.current_iter += 1
                    batch = next(dldrs.weight_trn_loader)
                    environ.set_inputs(batch , weight_input_size)

                    environ.optimize(is_policy=opt['policy'], 
                                     flag=ns.flag, 
                                     num_train_layers=ns.num_train_layers,
                                     hard_sampling=opt['train']['hard_sampling'],
                                     verbose = False)

                    t_weights.set_postfix({'it' : ns.current_iter, 
                                           'Lss': f"{environ.losses['task']['total'].item():.4f}" , 
                                           'Spr': f"{environ.losses['sparsity']['total'].item():.4e}",  
                                           'Shr': f"{environ.losses['sharing']['total'].item():.4e}",
                                           'lyr': f"{ns.num_train_layers}"})    
    
            ns.trn_losses = environ.losses
            environ.write_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Trn]", to_tb=False)
            # wandb.log(environ.losses, step = ns.current_epoch)
                        
            #--------------------------------------------------------------------
            # validation process (here current_iter_w and stop_iter_w are equal)
            #--------------------------------------------------------------------
            ns.val_metrics = environ.evaluate(dldrs.val_loader,  
                                              is_policy        = opt['policy'],
                                              num_train_layers = ns.num_train_layers,
                                              hard_sampling    = opt['train']['hard_sampling'],
                                              eval_iters       = ns.eval_iters, 
                                              disable_tqdm     = disable_tqdm, 
                                              leave = False, verbose = False)  

            # environ.write_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Val]")
            print_metrics_cr(ns.current_epoch, time.time() - start_time, ns.trn_losses, environ.val_metrics,
                             line_count, out=[sys.stdout, environ.log_file]) 
            line_count +=1
            wandb_log_metrics(ns.val_metrics, step = ns.current_epoch)

            ## Comment out to stop reducing LR during warmup phase
            environ.schedulers['weights'].step(ns.val_metrics['task']['total'])
            

            # Checkpoint on best results
            check_for_improvement(ns,opt,environ)                                 

            #-----------------------------------------------------------------------
            # END validation process 
            #-----------------------------------------------------------------------
            ns.flag = 'update_alpha'
            
        #-----------------------------------------
        # Policy Training  
        #-----------------------------------------
        if ns.flag == 'update_alpha':
            start_time = time.time()        
            environ.fix_weights()
            environ.free_alpha()
            environ.train()
            
            with trange( +1, ns.trn_iters_policy+1 , initial = 0, total = ns.trn_iters_policy,   file=sys.stdout,
                        position=0, dynamic_ncols = True, leave= False,  disable = disable_tqdm, 
                        desc=f"Ep:{ns.current_epoch} [policy] ") as t_policy :
                for _ in t_policy:    
                    ns.current_iter += 1
                    batch = next(dldrs.policy_trn_loader)
                    environ.set_inputs(batch, policy_input_size)

                    environ.optimize(is_policy        = opt['policy'],  
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
            ns.trn_losses = environ.losses
            environ.write_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Policy Trn]", to_tb=False)
            # wandb.log(environ.losses, step = ns.current_epoch)
            
            #--------------------------------------------------------------------
            # validation process (here current_iter_a and stop_iter_a are equal)
            #--------------------------------------------------------------------        
            ns.val_metrics = environ.evaluate(dldrs.val_loader, 
                                           is_policy        = opt['policy'],
                                           num_train_layers = ns.num_train_layers, 
                                           hard_sampling    = opt['train']['hard_sampling'],
                                           eval_iters       = ns.eval_iters, 
                                           disable_tqdm     = disable_tqdm, 
                                           leave = False, verbose = False)  

            # environ.write_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Policy Val]")
            print_metrics_cr(ns.current_epoch, time.time() - start_time, ns.trn_losses, environ.val_metrics, 
                             line_count, out=[sys.stdout, environ.log_file])      
            line_count +=1
            wandb_log_metrics(ns.val_metrics, step = ns.current_epoch)

            environ.schedulers['alphas'].step(ns.val_metrics['total']['total'])

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
            
    #         environ.display_trained_logits(current_epoch)        
    #         print_loss(current_epoc, current_iter, environ.val_metrics, title = f"[Policy trn]  ep:{current_epoch}   it:{current_iter}")
        
        #--------------------------------------------------
        # End of one iteration of Weight / Policy Training  
        #--------------------------------------------------
        if should(ns.current_epoch, 5):
            environ.save_checkpoint('model_latest_weights_policy', ns.current_iter, ns.current_epoch)        
            print_loss(environ.val_metrics, title = f"\n[e] Policy training epoch:{ns.current_epoch}  it:{ns.current_iter}",
                      out=[sys.stdout, environ.log_file])
            environ.display_trained_policy(ns.current_epoch,out=[sys.stdout, environ.log_file])
            environ.log_file.flush()
            line_count = 0
        else:
            if display_policy:
                environ.display_trained_logits(ns.current_epoch,out=[sys.stdout, environ.log_file])
                # environ.display_trained_policy(ns.current_epoch,out=[sys.stdout, environ.log_file])
            
    wrapup_phase(ns, opt, environ)
    return


def retrain_phase(ns, opt, environ, dldrs, epochs = None, disable_tqdm = True,
                  display_policy = False, verbose = False):
    ns.phase = 'retrain'
    if epochs is not None:
        ns.training_epochs = epochs

    ns.stop_epoch_training = ns.current_epoch + ns.training_epochs

    print_heading(f" Last Epoch Completed: {ns.current_epoch}   # of epochs to do:  {ns.training_epochs} -  epochs {ns.current_epoch+1} to {ns.stop_epoch_training}"
                f"\n weight train iterations : {ns.trn_iters_weights}"
                f"\n policy_lr               : {opt['train']['policy_lr']}"
                f"\n lambda_sparsity         : {opt['train']['lambda_sparsity']}"
                f"\n lambda_sharing          : {opt['train']['lambda_sharing']}", verbose = True)

    if  ns.current_epoch >=  ns.stop_epoch_training:
        return 
    line_count = 0
    weight_input_size = dldrs.weight_trn_loader.dataset.input_size
    # policy_input_size = dldrs.policy_trn_loader.dataset.input_size

    while (ns.current_epoch < ns.stop_epoch_training):
        ns.current_epoch+=1    
        start_time = time.time()

        with trange(+1, ns.trn_iters_weights+1 , initial = 0, total = ns.trn_iters_weights, position=0,  file=sys.stdout,
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

            ns.trn_losses = environ.losses
            environ.write_trn_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Trn]", to_tb=False)
            # wandb.log(environ.losses, step = ns.current_epoch)

        # validation   
        ns.val_metrics = environ.evaluate(dldrs.val_loader, 
                                        is_policy        = True,
                                        policy_sampling_mode = 'fix_policy',
                                        hard_sampling    = opt['train']['hard_sampling'],
                                        eval_iters       = ns.eval_iters, 
                                        disable_tqdm     = disable_tqdm, 
                                        leave = False, verbose = False)      

        # environ.write_val_metrics(ns.current_epoch, ns.current_iter, start_time, title = f"[Weight Val]")
        print_metrics_cr(ns.current_epoch, time.time() - start_time, ns.trn_losses, environ.val_metrics,
                         line_count, out=[sys.stdout, environ.log_file]) 
        line_count +=1        
        wandb_log_metrics(ns.val_metrics, step = ns.current_epoch)

        
    wrapup_phase(ns, opt, environ)
    return 


def wrapup_phase(ns, opt, environ, label = None):
    label = ns.phase if label is None else label

    # ns.model_label   = 'model_%s_ep_%d_seed_%04d'  % (label, ns.current_epoch, opt['random_seed'])
    # ns.metrics_label = 'metrics_%s_ep_%d_seed_%04d.pickle' % (label,ns.current_epoch, opt['random_seed'])
    
    # Write model checkpoint
    if ns.write_checkpoint:
        ns.model_label = f'model_{label}_last_ep_{ns.current_epoch}' 
        environ.save_checkpoint(ns.model_label, ns.current_iter, ns.current_epoch) 
        print_to(f" save {label} checkpoint  to :  {ns.model_label}", out=[sys.stdout, environ.log_file])    
    
    # write metrics to pickle file 
    ns.metrics_label = f'metrics_{label}_last_ep_{ns.current_epoch}.pickle' 

    save_to_pickle({'val_metrics'   : environ.val_metrics,
                    'best_metrics'  : ns.best_metrics,
                    'best_accuracy' : ns.best_accuracy, 
                    'best_roc_auc'  : ns.best_roc_auc,
                    'best_iter'     : ns.best_iter   ,
                    'best_epoch'    : ns.best_epoch  },
                    environ.opt['paths']['checkpoint_dir'], ns.metrics_label)

    print_to(f" save {label} metrics to     :  {ns.metrics_label}", out=[sys.stdout, environ.log_file])
    print_loss(environ.val_metrics, title = f"[Final] ep:{ns.current_epoch}  it:{ns.current_iter}",out=[sys.stdout])
    
    # Display training results 
    environ.display_trained_policy(ns.current_epoch,out=[sys.stdout, environ.log_file])
    environ.display_trained_logits(ns.current_epoch,out=[sys.stdout, environ.log_file])
    environ.log_file.flush()
    return 


def check_for_improvement(ns,opt,environ):
    label = 'best' 
    #------------------------------------------------------------------------ 
    #  Save Best Checkpoint Code (saved below and in sparsechem_env_dev.py)
    #----------------------------------------------------------------------- 
    ## ns.curriculum_epochs = (environ.num_layers * opt['curriculum_speed']) 

    if (ns.current_epoch - ns.check_for_improvment_wait) >= ns.curriculum_epochs:    

        # if environ.val_metrics['aggregated']['avg_prec_score'] > ns.best_accuracy:
        if environ.val_metrics['aggregated']['roc_auc_score'] > ns.best_roc_auc:
            print(f'Previous best_epoch: {ns.best_epoch:5d}   best iter: {ns.best_iter:5d}'
                  f'   best_accuracy: {ns.best_accuracy:.5f}    best ROC auc: {ns.best_roc_auc:.5f}')        
            ns.best_metrics     = environ.val_metrics
            ns.best_accuracy    = environ.val_metrics['aggregated']['avg_prec_score']
            ns.best_roc_auc     = environ.val_metrics['aggregated']['roc_auc_score']
            ns.best_iter        = ns.current_iter
            ns.best_epoch       = ns.current_epoch
            wandb.log({"best_roc_auc"  : ns.best_roc_auc,
                       "best_accuracy" : ns.best_accuracy,
                       "best_epoch"    : ns.best_epoch,
                       "best_iter"     : ns.best_iter}, 
                       step = ns.current_epoch)        

            print(f'Previous best_epoch: {ns.best_epoch:5d}   best iter: {ns.best_iter:5d}'
                  f'   best_accuracy: {ns.best_accuracy:.5f}    best ROC auc: {ns.best_roc_auc:.5f}')        
            
            # ns.metrics_label = f"metrics_{label}_seed_{opt['random_seed']:%04d}.pickle"
            ns.metrics_label = f'metrics_{label}.pickle' 
            save_to_pickle({'val_metrics'   : environ.val_metrics,
                            'best_metrics'  : ns.best_metrics,
                            'best_accuracy' : ns.best_accuracy, 
                            'best_roc_auc'  : ns.best_roc_auc,
                            'best_iter'     : ns.best_iter   ,
                            'best_epoch'    : ns.best_epoch  },
                            environ.opt['paths']['checkpoint_dir'], ns.metrics_label)
            print_to(f" save {label} metrics to     :  {ns.metrics_label}", out=[sys.stdout, environ.log_file])    

            if ns.write_checkpoint:
                # ns.model_label = f"model_{label}_seed_{opt['random_seed']:%04d}"
                ns.model_label = f"model_{label}"  
                environ.save_checkpoint(ns.model_label, ns.current_iter, ns.current_epoch) 
                print_to(f" save  {label} checkpoint to :  {ns.model_label}", out=[sys.stdout, environ.log_file])    
    return


def disp_info_1(ns, opt, environ):
    print(
            f"\n Num_blocks                : {sum(environ.networks['mtl-net'].layers)}"    
            f"                                \n"
            f"\n batch size                : {opt['train']['batch_size']}",    
            f"\n # batches / Warmup epoch  : {ns.trn_iters_weights}",
            f"\n # batches / Weight epoch  : {ns.trn_iters_weights}",
            f"\n # batches / Policy epoch  : {ns.trn_iters_policy}",
            # f"\n Total iterations          : {opt['train']['total_iters']}",
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
            # f"\n stop_iter_w               : {ns.trn_iters_weights}",
        )

 
def disp_gpu_device_info():
    
    print_underline('GPU Device Info ', verbose=True)
    for i in range(torch.cuda.device_count()):
        print(f" Device : cuda:{i}")
        print('   name:       ', torch.cuda.get_device_name(i))
        print('   capability: ', torch.cuda.get_device_capability(i))
        print('   properties: ', torch.cuda.get_device_properties(i))
        ## current GPU memory usage by tensors in bytes for a given device
        print('   Allocated : ', torch.cuda.memory_allocated(i) ) 
        ## current GPU memory managed by caching allocator in bytes for a given device, 
        ## in previous PyTorch versions the command was torch.cuda.memory_cached
        print('   Reserved  : ', torch.cuda.memory_reserved(i) )   
        print()

                        
def display_gpu_info():
    from GPUtil import showUtilization as gpu_usage
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex ,nvmlDeviceGetMemoryInfo   

    if torch.cuda.is_available():
        torch_gpu_id = torch.cuda.current_device()
        print_underline('CUDA Device(s) available', verbose=True)
        print(' CUDA device count   : ', torch.cuda.device_count())
        print(' CUDA current device : ', torch_gpu_id, '  name: ', torch.cuda.get_device_name(torch_gpu_id))
        print(' GPU Processes       : ', torch.cuda.list_gpu_processes())
        print()


        disp_gpu_device_info()
        nvmlInit()

        print_underline('GPU Usage Stats ', verbose=True)
        gpu_usage()     
        print()
        # print_underline('torch.device() : ', verbose=True)
        # device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        # print(device)
        # torch.cuda package supports CUDA tensor types but works with GPU computations. Hence, if GPU is used, it is common to use CUDA. 
        # torch.cuda.device_count()
        # torch.cuda.current_device()
        # torch.cuda.get_device_name(0)

        print(' torch.cuda.current-device(): ', torch_gpu_id)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
            print(' ids : ', ids)
            nvml_gpu_id = ids[torch_gpu_id] # remap
        else:
            nvml_gpu_id = torch_gpu_id
        
        print()
        print_underline(f"nvml_gpu_id: {nvml_gpu_id}", verbose=True)
        nvml_handle = nvmlDeviceGetHandleByIndex(nvml_gpu_id)
        print(f" nvml handle: {nvml_handle}")
        info = nvmlDeviceGetMemoryInfo(nvml_handle)
        print_underline(f"nvml Device Memory Info", verbose=True)
        print(info)
        
    else :
        print_underline(' No CUDA devices found ',verbose=True)


# def retrain_prep(ns, opt, environ, dldrs, phase = 'update_w', epoch = 0, iter = 0, verbose = False):
    
    # if torch.cuda.is_available():
    #     print_dbg(f" training preparation: - check for CUDA - cuda available as device id: {opt['gpu_ids']}", True)
    #     environ.cuda(opt['gpu_ids'])
    # else:
    #     print_dbg(f" training preparation: - check for CUDA - cuda not available", verbose = True)
    #     environ.cpu()

    # if opt['train']['print_freq'] == -1:
    #     print(f" training preparation: - set print_freq to length of train loader: {len(dldrs.warmup_trn_loader)}")
    #     ns.print_freq = len(dldrs.warmup_trn_loader)
    # else:
    #     print(f" training preparation: -  set print_freq to opt[train][print_freq]: {opt['train']['print_freq']}")
    #     ns.print_freq = opt['train']['print_freq']     

    # if opt['train']['val_iters'] == -1:
    #     print(f" training preparation: - set eval_iters to length of val loader : {len(dldrs.val_loader)}")
    #     ns.eval_iters    = len(dldrs.val_loader)    
    # else:
    #     print(f" training preparation: - set eval_iters to opt[train][val_iters]: {opt['train']['val_iters']}")
    #     ns.eval_iters    = opt['train']['val_iters']
        
    # ns.trn_iters_weights =  len(dldrs.weight_trn_loader) 
    # print(f" training preparation: - set number of batches per weight training epoch to: {opt['train']['weight_iter_alternate']}")
    # print(f" training preparation: - set number of batches per policy training epoch to: {opt['train']['alpha_iter_alternate']}")

    # ns.trn_iters_weights = opt['train']['weight_iter_alternate']
    # ns.trn_iters_policy = opt['train']['alpha_iter_alternate'] 

    # Fix Alpha -     
    # if phase == 'update_w':
    #     ns.flag = phase
    #     environ.fix_alpha()
    #     environ.free_weights(opt['fix_BN'])
    # elif phase == 'update_alpha':
    #     ns.flag = phase
    #     environ.fix_weights()
    #     environ.free_alpha() 
    # else: 
    #     raise ValueError('training mode/phase %s  is not valid' % phase)

    # ns.current_epoch  = epoch
    # ns.current_iter   = iter

    # ns.best_results   = {}
    # ns.best_metrics   = None
    # ns.best_accuracy  = 0
    # ns.best_roc_auc   = 0       
    # ns.best_iter      = 0
    # ns.best_epoch     = 0 
    # ns.check_for_improvment_wait  = 0

    # ns.write_checkpoint = True

    # opt['train']['retrain_total_iters'] = opt['train'].get('retrain_total_iters', opt['train']['total_iters'])
    # print(f"opt['train']['retrain_total_iters']:   {opt['train']['retrain_total_iters']}")
    # refer_metrics = get_reference_metrics(opt)
