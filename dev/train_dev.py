import torch

import scipy.special
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm.notebook import tqdm,trange
from dev.sparsechem_utils_dev import aggregate_results, compute_metrics
from utils.util import timestring, print_heading, print_dbg
    
import time



# def eval_dev(environ, dataloader, tasks, is_policy=False, num_train_layers=None, hard_sampling=False, num_seg_cls=-1, eval_iter=3):
# def evaluate_class_regr(net, loader, loss_class, loss_regr, tasks_class, tasks_regr, dev, progress=True):

def evaluate(environ, dataloader, tasks_class, is_policy= False, num_train_layers=None, hard_sampling=False,
            device = None, progress=True, eval_iter=-1, leave = False, verbose = False):

    val_metrics = {}
    loss_sum = 0.0
    loss_sum_mean = 0.0
    task_loss_sum = {}
    task_loss_sum_mean = {}
    task_loss_avg= {}
    task_class_weights  = {}
    data = {}
    num_class_tasks  = dataloader.dataset.class_output_size
    # num_regr_tasks   = dataloader.dataset.regr_output_size
    # class_w = tasks_class.aggregation_weight

    environ.eval()

    for t_id, task in enumerate(environ.tasks):
        task_key = f"task{t_id+1}"
        data[task_key] = { "yc_ind":  [],
                    "yc_data": [],
                    "yc_hat":  [],
                    }
        task_loss_sum[task_key] = 0.0
        task_loss_sum_mean[task_key] = 0.0
        task_loss_avg[task_key] = 0.0
        task_class_weights[task_key]  = 0.0

    if eval_iter == -1:
        eval_iter = len(dataloader)


    with torch.no_grad():

        ## Note: len(tt) is equal to len(dataloader)
        # with tqdm(dataloader, total=eval_iter, initial=0, leave=True, disable=(progress==False), position=0, desc = "validation") as tt:
        #     for batch_idx, batch in enumerate(tt,1) :
        #         # if batch_idx > eval_iter :
        #         #     break

        with trange(1,eval_iter+1, total=eval_iter, initial=0, leave=leave, disable=(progress==False), position=0, desc = "validation") as t_validation:
            
            for batch_idx in t_validation:
        
                print_dbg(f"\n + Validation Loop Start - batch_idx:{batch_idx}  eval_iter: {eval_iter}  t_validation: {len(t_validation)} \n",verbose=verbose)

                batch = next(dataloader)
                environ.set_inputs(batch, dataloader.dataset.input_size)  
                environ.val2(is_policy, num_train_layers, hard_sampling, verbose = verbose)            
                
                for t_id, _ in enumerate(environ.tasks):
                    task_key = f"task{t_id+1}"
                    loss_sum      += environ.metrics[task_key]['cls_loss']
                    loss_sum_mean += environ.metrics[task_key]['cls_loss_mean']

                    task_loss_sum[task_key]      += environ.metrics[task_key]['cls_loss']                    
                    task_loss_sum_mean[task_key] += environ.metrics[task_key]['cls_loss_mean']
                    
                    task_class_weights[task_key]  += environ.metrics[task_key]['yc_wghts_sum']
                    
                    data[task_key]['yc_trn_weights']  = environ.metrics[task_key]['yc_trn_weights']
                    data[task_key]['yc_aggr_weights'] = environ.metrics[task_key]['yc_aggr_weights']
                    
                    ## storing Y data for EACH TASK (for metric computations)
                    for key in ["yc_ind", "yc_data", "yc_hat"]:
                        if (key in environ.metrics[task_key]) and (environ.metrics[task_key][key] is not None):
                            data[task_key][key].append(environ.metrics[task_key][key].cpu())            
                
                t_validation.set_postfix({'bch_idx': batch_idx, 
                                'loss'   : f"{loss_sum.item()/batch_idx:.4f}" ,
                                'row_ids': f"{batch['row_id'][0]} - {batch['row_id'][-1]}" ,
                                'task_wght': f"{environ.metrics[task_key]['yc_wghts_sum']}"})

                if verbose:                
                    for t_id, _ in enumerate(environ.tasks):
                        task_key = f"task{t_id+1}"
                        print_dbg(f" + Validation Loop - batch_idx:{batch_idx}  eval_iter: {eval_iter}\n"
                                  f"    task {t_id+1:3d}: loss     : {environ.metrics[task_key]['cls_loss']:.4f}   sum(err)     : {task_loss_sum[task_key]:.4f}    avg(err)=sum(err)/batch_id: {task_loss_sum[task_key]/batch_idx:.4f}   \n"
                                  f"    task {t_id+1:3d}: loss_mean: {environ.metrics[task_key]['cls_loss_mean']:.4f}   sum(err_mean): {task_loss_sum_mean[task_key]:.4f}    avg(err_mean): {task_loss_sum_mean[task_key]/batch_idx:.4f}",
                                  verbose = True)
                        print_dbg(f"    environ.metrics[task_key][yc_wghts_sum] {environ.metrics[task_key]['yc_wghts_sum']}  task_weights[task_key]: {task_class_weights[task_key]}  ", verbose = True)
                    
                    print_dbg(f"\n + Validation Loop end - batch_idx:{batch_idx}  eval_iter: {eval_iter} \n"
                              f"    all tasks: loss_sum     : {loss_sum:.4f}   avg(loss_sum)     : {loss_sum/batch_idx:.4f} \n"
                              f"    all tasks: loss_sum_mean: {loss_sum_mean:6.4f}    avg(loss_sum_mean): {loss_sum_mean / batch_idx:.4f}", verbose = True)
                

        ##-----------------------------------------------------------------------
        ## All Validation batches have been feed to network - calcualte metrics
        ##-----------------------------------------------------------------------
        if verbose:
            print_heading(f" + Validation Loops complete- batch_idx:{batch_idx}  eval_iter: {eval_iter}", verbose = True)
            for t_id, _ in enumerate(environ.tasks):
                task_key = f"task{t_id+1}"
                print_dbg(f"    task {t_id+1:3d}: sum(err)     : {task_loss_sum[task_key]:6.4f}    avg(err)=sum(err)/batch_id: {task_loss_sum[task_key]/batch_idx:.4f}   \n"
                          f"    task {t_id+1:3d}: sum(err_mean): {task_loss_sum_mean[task_key]:6.4f}   avg(err_mean): {task_loss_sum_mean[task_key]/batch_idx:.4f}\n"
                          f"    environ.metrics[task_key][yc_wghts_sum] {environ.metrics[task_key]['yc_wghts_sum']}  task_weights[task_key]: {task_class_weights[task_key]} \n", verbose = True)
                # print_dbg(f"    yc_aggr_weight          : {data[task_key]['yc_aggr_weights']}", verbose = True)
            
            print_dbg(f"    all tasks: loss_sum     : {loss_sum:.4f}   avg(loss_sum)     : {loss_sum/eval_iter:.4f}      \n"
                      f"    all tasks: loss_sum_mean: {loss_sum_mean:.4f}   avg(loss_sum_mean): {loss_sum_mean / eval_iter:.4f} \n ", verbose = True)


        loss_sum /= batch_idx
        loss_sum_mean /= batch_idx
 
        val_metrics["loss"]  = {"total": loss_sum.item()}
        val_metrics["loss_mean"] = {"total" : loss_sum_mean.item()}
        task_classification_metrics = []
        task_aggregation_weights = [] 

        for t_id, task in enumerate(environ.tasks):
            task_key = f"task{t_id+1}"
            val_metrics["loss"][task_key] = task_loss_sum[task_key].cpu().item() / batch_idx
            val_metrics["loss_mean"][task_key] = task_loss_sum_mean[task_key].cpu().item() / batch_idx 

            val_metrics[task_key] = {}
            # yc_ind  = torch.cat(batch["yc_ind"], dim=1).numpy()
            # yc_data = torch.cat(batch["yc_data"], dim=0).numpy()
            # yc_hat  = torch.cat(data["yc_hat"], dim=0).numpy()
            yc_ind  = torch.cat(data[task_key]["yc_ind"] , dim=1).numpy()
            yc_data = torch.cat(data[task_key]["yc_data"], dim=0).numpy()
            yc_hat  = torch.cat(data[task_key]["yc_hat"] , dim=0).numpy()
            yc_aggr_weight = data[task_key]["yc_aggr_weights"]
            
            # print(f" num_tasks: {num_class_tasks[t_id]}")
            # print(f" yc_ind  : {type(yc_ind)}  shape: {yc_ind.shape}")
            # print(f" yc_data : {type(yc_data)} shape: {yc_data.shape}")
            # print(f" yc_hat  : {type(yc_hat)}  shape: {yc_hat.shape}")
            # print(f" yc_aggr_weight  : {type(yc_aggr_weight)}  shape: {yc_aggr_weight}")

            # out["classification"]     = compute_metrics(yc_ind[1], y_true=yc_data, y_score=yc_hat, num_tasks=num_class_tasks)
            val_metrics[task_key]["classification"]     = compute_metrics(yc_ind[1], y_true=yc_data, y_score=yc_hat, num_tasks=num_class_tasks[t_id])

            # out["classification_agg"] = aggregate_results(out["classification"], weights=class_w)
            val_metrics[task_key]["classification_agg"] = aggregate_results(val_metrics[task_key]["classification"], weights=yc_aggr_weight)

            ## Convert pandas series to dict to make it compatible with print_loss()
            val_metrics[task_key]["classification_agg"] = val_metrics[task_key]["classification_agg"].to_dict()
         
            # print_heading(f" {timestring()} aggregate_results()", verbose)


            val_metrics[task_key]['classification_agg']['sc_loss'] = task_loss_sum[task_key].cpu().item() / eval_iter 
            val_metrics[task_key]["classification_agg"]["logloss"] = task_loss_sum[task_key].cpu().item() / task_class_weights[task_key].cpu().item()

            task_classification_metrics.append(val_metrics[task_key]['classification'])
            task_aggregation_weights.append(yc_aggr_weight)
        
            # val_metrics[task_key]["classification_agg"]["logloss"] = loss_class_sum[task_key].cpu().item() / loss_class_weights[task_key].cpu().item()    
            # val_metrics[task_key]["classification_agg"]["num_tasks_total"] = dataloader.dataset.class_output_size[t_id]
            # val_metrics[task_key]["classification_agg"]["num_tasks_agg"]   = (tasks_class.aggregation_weight > 0).sum()


            # print_dbg(f" loss_class_sum    :  {type(loss_class_sum[task_key])}      {loss_class_sum[task_key]}", verbose)
            # print_dbg(f" loss_class_weights:  {type(loss_class_weights[task_key])} {loss_class_weights[task_key]}", verbose)
            # print_dbg(f" val_metrics[task_key][classification_agg] : { val_metrics[task_key]['classification_agg']} ", verbose)
            # val_metrics[task_key]['classification_agg']['yc_weights_sum'] = environ.metrics[task_key]["yc_wghts_sum"].cpu().item()
 
        ## Calculate aggregated metrics across all task groups
        
        val_metrics['aggregated'] = aggregate_results( pd.concat(task_classification_metrics),
                                                               np.concatenate(task_aggregation_weights)).to_dict()
    
        
        environ.train()
        return val_metrics