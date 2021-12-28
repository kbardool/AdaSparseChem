import torch
import sklearn.metrics
import scipy.special
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook,  trange, tnrange
from utils.util import timestring, print_heading, print_dbg
from sklearn.metrics import confusion_matrix
import time

def calc_acc_kappa(recall, fpr, num_pos, num_neg):
    """Calculates accuracy from recall and precision."""
    num_all = num_neg + num_pos
    tp = np.round(recall * num_pos).astype(np.int)
    fn = num_pos - tp
    fp = np.round(fpr * num_neg).astype(np.int)
    tn = num_neg - fp
    acc   = (tp + tn) / num_all
    pexp  = num_pos / num_all * (tp + fp) / num_all + num_neg / num_all * (tn + fn) / num_all
    kappa = (acc - pexp) / (1 - pexp)
    return acc, kappa


def all_metrics(y_true, y_score):
    """
    Compute classification metrics.
    
    Args:
        y_true     true labels (0 / 1)
        y_score    logit values
    """
    if len(y_true) <= 1 or (y_true[0] == y_true).all():
        df = pd.DataFrame({"roc_auc_score": [np.nan], "auc_pr": [np.nan], "avg_prec_score": [np.nan], "f1_max": [np.nan], "p_f1_max": [np.nan], "kappa": [np.nan], "kappa_max": [np.nan], "p_kappa_max": [np.nan], "bceloss": [np.nan]})
        return df

    fpr, tpr, tpr_thresholds = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
    roc_auc_score = sklearn.metrics.auc(x=fpr, y=tpr)
    precision, recall, pr_thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true, probas_pred = y_score)

    bceloss = F.binary_cross_entropy_with_logits(
        input  = torch.FloatTensor(y_score),
        target = torch.FloatTensor(y_true),
        reduction="none").mean().item()

    ## calculating F1 for all cutoffs
    F1_score       = np.zeros(len(precision))
    mask           = precision > 0
    F1_score[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
    f1_max_idx     = F1_score.argmax()
    f1_max         = F1_score[f1_max_idx]
    p_f1_max       = scipy.special.expit(pr_thresholds[f1_max_idx])

    auc_pr = sklearn.metrics.auc(x = recall, y = precision)
    avg_prec_score = sklearn.metrics.average_precision_score(
          y_true  = y_true,
          y_score = y_score)
    y_classes = np.where(y_score >= 0.0, 1, 0)

    ## accuracy for all thresholds
    acc, kappas   = calc_acc_kappa(recall=tpr, fpr=fpr, num_pos=(y_true==1).sum(), num_neg=(y_true==0).sum())
    kappa_max_idx = kappas.argmax()
    kappa_max     = kappas[kappa_max_idx]
    p_kappa_max   = scipy.special.expit(tpr_thresholds[kappa_max_idx])

    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_classes)
    df = pd.DataFrame({"roc_auc_score": [roc_auc_score], "auc_pr": [auc_pr], "avg_prec_score": [avg_prec_score], "f1_max": [f1_max], "p_f1_max": [p_f1_max], "kappa": [kappa], "kappa_max": [kappa_max], "p_kappa_max": p_kappa_max, "bceloss": bceloss})
    return df


def compute_metrics(cols, y_true, y_score, num_tasks):
    """
    Compute metrics for the SparseChem classification
    """
    if len(cols) < 1:
        return pd.DataFrame({
            "roc_auc_score" : np.nan,
            "auc_pr"        : np.nan,
            "avg_prec_score": np.nan,
            "f1_max"        : np.nan,
            "p_f1_max"      : np.nan,
            "kappa"         : np.nan,
            "kappa_max"     : np.nan,
            "p_kappa_max"   : np.nan,
            "bceloss"       : np.nan}, index=np.arange(num_tasks))
    
    df   = pd.DataFrame({"task": cols, "y_true": y_true, "y_score": y_score})
    
    metrics = df.groupby("task", sort=True).apply(lambda g:
              all_metrics( y_true  = g.y_true.values,
                           y_score = g.y_score.values))
    
    metrics.reset_index(level=-1, drop=True, inplace=True)
    
    return metrics.reindex(np.arange(num_tasks))


def aggregate_results(df, weights):
    """Compute aggregates based on the weights"""
    wsum = weights.sum()
    if wsum == 0:
        return pd.Series(np.nan, index=df.columns)
    df2 = df.where(pd.isnull, 1) * weights[:,None]
    return (df2.multiply(1.0 / df2.sum(axis=0), axis=1) * df).sum(axis=0)


# def eval_dev(environ, dataloader, tasks, policy=False, num_train_layers=None, hard_sampling=False, num_seg_cls=-1, eval_iter=3):
# def evaluate_class_regr(net, loader, loss_class, loss_regr, tasks_class, tasks_regr, dev, progress=True):

def eval_dev(environ, dataloader, tasks_class, policy= False, num_train_layers=None, hard_sampling=False,
            device = None, progress=True, eval_iter=-1, verbose = False):

    val_metrics = {}
    # batch_size = []
    # records = {}
    loss_sum = 0.0
    task_loss_sum = {}
    task_weights  = {}
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
        task_weights[task_key] = 0.0 

    if eval_iter == -1:
        eval_iter = len(dataloader)


    with torch.no_grad():
        # with  trange(eval_iter, position = 1, leave=False, desc = "validation") as tt:
            # for val_iter in tt:
                # if eval_iter != -1:
            # print(f"** {timestring()}  VALIDATION iteration {batch_idx} of {eval_iter}")
            # batch = next(iter(dataloader))
        batch_ctr = 0
        # for batch_idx, batch in enumerate(tqdm(dataloader, leave=False, disable=(progress==False), position=0, desc = "validation") ):
        with tqdm_notebook(dataloader, total=eval_iter, leave=False, disable=(progress==False), position=0, desc = "validation") as tt:
            for batch_idx, batch in enumerate(tt,1) :
         
                if batch_idx > eval_iter:
                    break
                
                time.sleep(1)
                batch_ctr += 1
                environ.set_inputs(batch, dataloader.dataset.input_size)  
                environ.val2(policy, num_train_layers, hard_sampling)            
                
                for t_id, _ in enumerate(environ.tasks):
                    task_key = f"task{t_id+1}"
                    loss_sum += environ.metrics[task_key]['err']
                    task_loss_sum[task_key] += environ.metrics[task_key]['err']
                    task_weights[task_key]  += environ.metrics[task_key]['yc_wghts_sum']
                    
                    data[task_key]['yc_trn_weights']  = environ.metrics[task_key]['yc_trn_weights']
                    data[task_key]['yc_aggr_weights'] = environ.metrics[task_key]['yc_aggr_weights']
                    
                    ## storing data for AUCs for EACH TASK
                    for key in ["yc_ind", "yc_data", "yc_hat"]:
                        if (key in environ.metrics[task_key]) and (environ.metrics[task_key][key] is not None):
                            data[task_key][key].append(environ.metrics[task_key][key].cpu())            
                
                tt.set_postfix({'bch_idx': batch_idx, 
                                'loss'   : f"{loss_sum.item()/batch_ctr:.4f}" ,
                                'row_ids': f"{batch['row_id'][0]} - {batch['row_id'][-1]}" })
                if verbose:                
                    for t_id, _ in enumerate(environ.tasks):
                        task_key = f"task{t_id+1}"
                        print_dbg(f"     batch_ctr:{batch_ctr}   task_loss: {environ.metrics[task_key]['err']:.4f}  sum_task_loss: {task_loss_sum[task_key]:.4f} task_loss_avg: {task_loss_sum[task_key]/batch_ctr:.4f}"  
                                  f"   loss_sum : {loss_sum:.4f}    loss: {loss_sum/batch_ctr:.4f}      ", verbose = True)
            
        ##-----------------------------------------------------------------------
        ## All Validation batches have been feed to network - calcualte metrics
        ##-----------------------------------------------------------------------
        if verbose :
            for t_id, _ in enumerate(environ.tasks):
                task_key = f"task{t_id+1}"
                print_dbg(f" ++ Validation - eval_iter:{eval_iter}    sum_task_loss: {task_loss_sum[task_key]:.4f}   task_loss_avg: {task_loss_sum[task_key]/batch_ctr:.4f}",
                        verbose = True)
            print_dbg(f" ++ Validation - eval_iter:{eval_iter}    loss_sum : {loss_sum:.4f}  loss_sum_avg:{loss_sum/eval_iter:.4f} ", verbose = True)

        loss_sum /= eval_iter
        # if len(data["yc_ind"]) == 0:
        #     ## there are no data for classification
        #     out["classification"] = compute_metrics([], y_true=[], y_score=[], num_tasks=num_class_tasks)
        #     out["classification_agg"] = out["classification"].reindex(labels=[]).mean(0)
        #     out["classification_agg"]["logloss"] = np.nan
        # else:
        val_metrics["loss"] = {"total": loss_sum.item()}

        for t_id, task in enumerate(environ.tasks):
            task_key = f"task{t_id+1}"
            val_metrics[task_key] = {}
            yc_ind  = torch.cat(data[task_key]["yc_ind"] , dim=1).numpy()
            yc_data = torch.cat(data[task_key]["yc_data"], dim=0).numpy()
            yc_hat  = torch.cat(data[task_key]["yc_hat"] , dim=0).numpy()
            yc_aggr_weight = data[task_key]["yc_aggr_weights"]
            
            # print(f" num_tasks: {num_class_tasks[t_id]}")
            # print(f" yc_ind  : {type(yc_ind)}  shape: {yc_ind.shape}")
            # print(f" yc_data : {type(yc_data)} shape: {yc_data.shape}")
            # print(f" yc_hat  : {type(yc_hat)}  shape: {yc_hat.shape}")
            # print(f" yc_aggr_weight  : {type(yc_aggr_weight)}  shape: {yc_aggr_weight}")
            # print_heading(f" {timestring()} compute_metrics()", verbose)
            val_metrics[task_key]["classification"]     = compute_metrics(yc_ind[1], y_true=yc_data, y_score=yc_hat, num_tasks=num_class_tasks[t_id])
         
            # print_heading(f" {timestring()} aggregate_results()", verbose)
            val_metrics[task_key]["classification_agg"] = aggregate_results(val_metrics[task_key]["classification"], weights=yc_aggr_weight)
            
            val_metrics[task_key]["classification_agg"]["logloss"] = task_loss_sum[task_key].cpu().item() / task_weights[task_key].cpu().item()
            # val_metrics[task_key]["classification_agg"]["logloss"] = loss_class_sum[task_key].cpu().item() / loss_class_weights[task_key].cpu().item()
            
            val_metrics[task_key]["classification_agg"]["num_tasks_total"] = dataloader.dataset.class_output_size[t_id]
            # val_metrics[task_key]["classification_agg"]["num_tasks_agg"]   = (tasks_class.aggregation_weight > 0).sum()

            ## Convert pandas series to dict to make it compatible with print_loss()
            val_metrics[task_key]["classification_agg"] = val_metrics[task_key]["classification_agg"].to_dict()

            # print_dbg(f" loss_class_sum:      {type(loss_class_sum[task_key])}      {loss_class_sum[task_key]}", verbose)
            # print_dbg(f" loss_class_weights:  {type(loss_class_weights[task_key])} {loss_class_weights[task_key]}", verbose)
            # print_dbg(f" val_metrics[task_key][classification_agg] : { val_metrics[task_key]['classification_agg']} ", verbose)
            val_metrics[task_key]['classification_agg']['err'] = environ.metrics[task_key]['err']
            val_metrics[task_key]['classification_agg']['weights'] = environ.metrics[task_key]["yc_wghts_sum"]
 
            # yc_ind  = torch.cat(batch["yc_ind"], dim=1).numpy()
            # yc_data = torch.cat(batch["yc_data"], dim=0).numpy()
            # yc_hat  = torch.cat(data["yc_hat"], dim=0).numpy()

            # out["classification"]     = compute_metrics(yc_ind[1], y_true=yc_data, y_score=yc_hat, num_tasks=num_class_tasks)
            # out["classification_agg"] = aggregate_results(out["classification"], weights=class_w)
            # out["classification_agg"]["logloss"] = loss_class_sum.cpu().item() / loss_class_weights.cpu().item()
    
            # out["classification_agg"]["num_tasks_total"] = dataloader.dataset.class_output_size
            # out["classification_agg"]["num_tasks_agg"]   = (tasks_class.aggregation_weight > 0).sum()
        
        environ.train()
        return val_metrics

 
    



def eval_old(environ, dataloader, tasks, policy=False, num_train_layers=None, hard_sampling=False, num_seg_cls=-1, eval_iter=3):
    batch_size = []
    records = {}
    val_metrics = {}
    
    if 'seg' in tasks:
        assert (num_seg_cls != -1)
        records['seg'] = {'mIoUs': [], 'pixelAccs': [],  'errs': [], 'conf_mat': np.zeros((num_seg_cls, num_seg_cls)),
                          'labels': np.arange(num_seg_cls)}
    if 'sn' in tasks:
        records['sn'] = {'cos_similaritys': []}
    if 'depth' in tasks:
        records['depth'] = {'abs_errs': [], 'rel_errs': [], 'sq_rel_errs': [], 'ratios': [], 'rms': [], 'rms_log': []}
    if 'keypoint' in tasks:
        records['keypoint'] = {'errs': []}
    if 'edge' in tasks:
        records['edge'] = {'errs': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if eval_iter != -1:
                if batch_idx > eval_iter:
                    break
            print(f"** {timestring()}  VALIDATION iteration ({batch_idx})")
            
            environ.set_inputs(batch)
            # seg_pred, seg_gt, pixelAcc, cos_similarity = environ.val(policy, num_train_layers, hard_sampling)
            
            metrics = environ.val2(policy, num_train_layers, hard_sampling)

            # environ.networks['mtl-net'].task1_logits
            # mIoUs.append(mIoU)
            
            if 'seg' in tasks:
                new_mat = confusion_matrix(metrics['seg']['gt'], metrics['seg']['pred'], records['seg']['labels'])
                assert (records['seg']['conf_mat'].shape == new_mat.shape)
                records['seg']['conf_mat'] += new_mat
                records['seg']['pixelAccs'].append(metrics['seg']['pixelAcc'])
                records['seg']['errs'].append(metrics['seg']['err'])
            if 'sn' in tasks:
                records['sn']['cos_similaritys'].append(metrics['sn']['cos_similarity'])
            if 'depth' in tasks:
                records['depth']['abs_errs'].append(metrics['depth']['abs_err'])
                records['depth']['rel_errs'].append(metrics['depth']['rel_err'])
                records['depth']['sq_rel_errs'].append(metrics['depth']['sq_rel_err'])
                records['depth']['ratios'].append(metrics['depth']['ratio'])
                records['depth']['rms'].append(metrics['depth']['rms'])
                records['depth']['rms_log'].append(metrics['depth']['rms_log'])
            if 'keypoint' in tasks:
                records['keypoint']['errs'].append(metrics['keypoint']['err'])
            if 'edge' in tasks:
                records['edge']['errs'].append(metrics['edge']['err'])
            batch_size.append(len(batch['img']))

    print(f" {timestring()}   VALIDATION LOOP COMPLETE ({batch_idx})")
            
            
    # overall_mIoU = (np.array(mIoUs) * np.array(batch_size)).sum() / sum(batch_size)
    if 'seg' in tasks:
        val_metrics['seg'] = {}
        jaccard_perclass = []
        for i in range(num_seg_cls):
            if not records['seg']['conf_mat'][i, i] == 0:
                jaccard_perclass.append(records['seg']['conf_mat'][i, i] / (np.sum(records['seg']['conf_mat'][i, :]) +
                                                                            np.sum(records['seg']['conf_mat'][:, i]) -
                                                                            records['seg']['conf_mat'][i, i]))

        val_metrics['seg']['mIoU']      = np.sum(jaccard_perclass) / len(jaccard_perclass)
        val_metrics['seg']['Pixel Acc'] = (np.array(records['seg']['pixelAccs']) * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['seg']['err']       = (np.array(records['seg']['errs']) * np.array(batch_size)).sum() / sum(batch_size)

    if 'sn' in tasks:
        val_metrics['sn'] = {}
        overall_cos = np.clip(np.concatenate(records['sn']['cos_similaritys']), -1, 1)

        angles = np.arccos(overall_cos) / np.pi * 180.0
        val_metrics['sn']['cosine_similarity'] = overall_cos.mean()
        val_metrics['sn']['Angle Mean']        = np.mean(angles)
        val_metrics['sn']['Angle Median']      = np.median(angles)
        val_metrics['sn']['Angle RMSE']        = np.sqrt(np.mean(angles ** 2))
        val_metrics['sn']['Angle 11.25']       = np.mean(np.less_equal(angles, 11.25)) * 100
        val_metrics['sn']['Angle 22.5']        = np.mean(np.less_equal(angles, 22.5)) * 100
        val_metrics['sn']['Angle 30']          = np.mean(np.less_equal(angles, 30.0)) * 100
        val_metrics['sn']['Angle 45']          = np.mean(np.less_equal(angles, 45.0)) * 100

    if 'depth' in tasks:
        val_metrics['depth'] = {}
        records['depth']['abs_errs']         = np.stack(records['depth']['abs_errs'], axis=0)
        records['depth']['rel_errs']         = np.stack(records['depth']['rel_errs'], axis=0)
        records['depth']['sq_rel_errs']      = np.stack(records['depth']['sq_rel_errs'], axis=0)
        records['depth']['ratios']           = np.concatenate(records['depth']['ratios'], axis=0)
        records['depth']['rms']              = np.concatenate(records['depth']['rms'], axis=0)
        records['depth']['rms_log']          = np.concatenate(records['depth']['rms_log'], axis=0)
        records['depth']['rms_log']          = records['depth']['rms_log'][~np.isnan(records['depth']['rms_log'])]
          
        val_metrics['depth']['abs_err']      = (records['depth']['abs_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['rel_err']      = (records['depth']['rel_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['sq_rel_err']   = (records['depth']['sq_rel_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['sigma_1.25']   = np.mean(np.less_equal(records['depth']['ratios'], 1.25)) * 100
        val_metrics['depth']['sigma_1.25^2'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 2)) * 100
        val_metrics['depth']['sigma_1.25^3'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 3)) * 100
        val_metrics['depth']['rms']          = (np.sum(records['depth']['rms']) / len(records['depth']['rms'])) ** 0.5
        # val_metrics['depth']['rms_log']    = (np.sum(records['depth']['rms_log']) / len(records['depth']['rms_log'])) ** 0.5

    return val_metrics## Prepare dataloaders