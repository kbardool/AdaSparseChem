import wandb 

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
    wandb.define_metric("best_roc_auc", summary="last")
    wandb.define_metric("best_epoch", summary="last")
    wandb.define_metric("best_iter", summary="last")

    # assert wandb.run is None, "Run is still running"
    print(f" WandB Initialization -----------------------------------------------------------\n"
          f" PROJECT NAME: {ns.wandb_run.project}\n"
          f" RUN ID      : {ns.wandb_run.id} \n"
          f" RUN NAME    : {ns.wandb_run.name}\n"     
          f" --------------------------------------------------------------------------------")
    return 

def wandb_watch(item = None, log = 'all', log_freq = 1000):
    """
    Note: Increasing the log frequency can result in longer run times
    """
    if item is not None:
        wandb.watch(item, log='all', log_freq= log_freq)        


def wandb_log_metrics(val_metrics, step = None):

    wandb.log({ **val_metrics['parms'], 
                **val_metrics['aggregated'],
                'ERRORS'    : {**val_metrics['total']}, 
                'SHARING'   : {**val_metrics['sharing']}, 
                'SPARSITY'  : {**val_metrics['sparsity']},
                'epoch'     : val_metrics['epoch']}, step = step)

    return

