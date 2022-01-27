import os 
import sys
import numpy as np
import torch
import types
import pandas as pd
from numpy.core.numeric import full
import scipy.sparse
import scipy.io
import scipy.special
import sklearn.metrics
from collections import namedtuple
from utils.util import print_heading, print_dbg, debug_on, debug_off
from utils.flops_benchmark import add_flops_counting_methods
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F 
# import torch
# import numpy as np

##########################################################################################################
##
##  SparseChem functions 
##
##########################################################################################################


def count_params(model):
    num_params = 0.
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        v_shape = v.shape
        num_params += np.prod(v_shape)

    print('Number of Parameters = %.2f M' % (num_params/1e6))


def compute_flops(model, input, kwargs_dict):
    model = add_flops_counting_methods(model)
    model = model.cuda().train()

    model.start_flops_count()

    _ = model(input, **kwargs_dict)
    gflops = model.compute_average_flops_cost()

    return gflops

def sparse_split2(tensor, split_size, dim=0):
    """
    Splits tensor into two parts.
    Args:
        split_size   index where to split
        dim          dimension which to split
    """
    assert tensor.layout == torch.sparse_coo
    indices = tensor._indices()
    values  = tensor._values()

    shape  = tensor.shape
    shape0 = shape[:dim] + (split_size,) + shape[dim+1:]
    shape1 = shape[:dim] + (shape[dim] - split_size,) + shape[dim+1:]

    mask0 = indices[dim] < split_size
    X0 = torch.sparse_coo_tensor(
            indices = indices[:, mask0],
            values  = values[mask0],
            size    = shape0)

    indices1       = indices[:, ~mask0]
    indices1[dim] -= split_size
    X1 = torch.sparse_coo_tensor(
            indices = indices1,
            values  = values[~mask0],
            size    = shape1)
    return X0, X1


def load_sparse(dataroot = None , filename = None):
    """Loads sparse from Matrix market or Numpy .npy file."""
    if filename is None or dataroot is None:
        return None
    full_path = os.path.join(dataroot, filename)
    if filename.endswith('.mtx'):
        return scipy.io.mmread(full_path).tocsr()
    elif filename.endswith('.npy'):
        return np.load(full_path, allow_pickle=True).item().tocsr()
    elif filename.endswith('.npz'):
        return scipy.sparse.load_npz(full_path).tocsr()
    raise ValueError(f"Loading '{full_path}' failed. It must have a suffix '.mtx', '.npy', '.npz'.")


    
def load_check_sparse(filename, shape):
    y = load_sparse(filename)
    if y is None:
        return scipy.sparse.csr_matrix(shape, dtype=np.float32)
    assert y.shape == shape, f"Shape of sparse matrix {filename} should be {shape} but is {y.shape}."
    return y


def fold_and_transform_inputs(x, folding_size=None, transform=None, verbose = False):
    """Fold and transform sparse matrix x:
    Args:
        x             sparse matrix
        folding_size  modulo for folding
        transform     "none", "binarize", "tanh", "log1p"
    Returns folded and transformed x.
    """
    print_dbg(f" fold and transform - folding_size:{folding_size}  transform: {transform}" , verbose = verbose)
    if folding_size is not None and x.shape[1] > folding_size:
        ## collapse x into folding_size columns
        idx = x.nonzero()
        folded = idx[1] % folding_size
        x = scipy.sparse.csr_matrix((x.data, (idx[0], folded)), shape=(x.shape[0], folding_size))
        x.sum_duplicates()    ## <--- duplicate entries are already summed together 

    if transform is None:
        pass
    elif transform == "binarize":
        x.data = (x.data > 0).astype(np.float32)
    elif transform == "tanh":
        x.data = np.tanh(x.data).astype(np.float32)
    elif transform == "log1p":
        x.data = np.log1p(x.data).astype(np.float32)
    else:
        raise ValueError(f"Unknown input transformation '{transform}'.")
        
    return x

##
## SparseChem Metric calculations 
##
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

##
## SparseChem Metric calculations 
##
def all_metrics(y_true, y_score):   
    """
    Compute classification metrics.
    
    Args:
        y_true     true labels (0 / 1)
        y_score    logit values
    """
    ## Setup pandas datafrme for metrics
    if len(y_true) <= 1 or (y_true[0] == y_true).all():
        df = pd.DataFrame({"roc_auc_score": [np.nan], "auc_pr": [np.nan], "avg_prec_score": [np.nan], "f1_max": [np.nan], "p_f1_max": [np.nan], "kappa": [np.nan], "kappa_max": [np.nan], "p_kappa_max": [np.nan], "bceloss": [np.nan]})
        return df

    ### ROC  (TPR / FPR Curve)
    fpr, tpr, tpr_thresholds = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
    
    ### AUC
    roc_auc_score = sklearn.metrics.auc(x=fpr, y=tpr)
    
    ### Precision / Recall curve
    precision, recall, pr_thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true, probas_pred = y_score)

    # with np.errstate(divide='ignore'):
    #      #precision can be zero but can be ignored so disable warnings (divide by 0)
    #      precision_cal = 1/(((1/precision - 1)*cal_fact_aucpr_task)+1)

    ### Binary Cross Entopy Loss 
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

    ## expit() is the logistic sigmoid function 
    p_f1_max       = scipy.special.expit(pr_thresholds[f1_max_idx])

    ### Calculate area under Precsion/Recall curve
    auc_pr = sklearn.metrics.auc(x = recall, y = precision)

    # auc_pr_cal = sklearn.metrics.auc(x = recall, y = precision_cal)
    
    avg_prec_score = sklearn.metrics.average_precision_score(
          y_true  = y_true,
          y_score = y_score)
    y_classes = np.where(y_score >= 0.0, 1, 0)

    ## accuracy for all thresholds
    acc, kappas   = calc_acc_kappa(recall=tpr, fpr=fpr, num_pos=(y_true==1).sum(), num_neg=(y_true==0).sum())

    kappa_max_idx = kappas.argmax()
    kappa_max     = kappas[kappa_max_idx]
    p_kappa_max   = scipy.special.expit(tpr_thresholds[kappa_max_idx])

    ## Cohen Kappa Score Cohen’s kappa [1], a score that expresses the level of agreement between two annotators 
    ## on a classification problem. It is defined as  
    ##
    ##             (p_o - p_e)
    ##    kappa =  -----------
    ##              ( 1 - p_e)  
    ##  where 
    ##        p_o : is the empirical probability of agreement on the label assigned to any sample 
    ##               (the observed agreement ratio)
    ##        p_e : is the expected agreement when both annotators assign labels randomly.  
    ##              p_e is estimated using a per-annotator empirical prior over the class labels [2].    
    
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_classes)

    df = pd.DataFrame({"roc_auc_score" : [roc_auc_score], 
                       "auc_pr"        : [auc_pr], 
                       "avg_prec_score": [avg_prec_score], 
                       "f1_max"        : [f1_max], 
                       "p_f1_max"      : [p_f1_max], 
                       "kappa"         : [kappa], 
                       "kappa_max"     : [kappa_max], 
                       "p_kappa_max"   : p_kappa_max, 
                       "bceloss"       : bceloss})
    return df


##
## SparseChem Metric calculations 
##
def compute_metrics(cols, y_true, y_score, num_tasks):
    """
    Compute metrics for the SparseChem classification for each individual task in a task group
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
    
    metrics = df.groupby("task", sort=True).apply(lambda g: all_metrics( y_true  = g.y_true.values,
                                                                         y_score = g.y_score.values))
    
    metrics.reset_index(level=-1, drop=True, inplace=True)
    
    return metrics.reindex(np.arange(num_tasks))


##
## SparseChem Metric calculations 
##
def aggregate_results(df, weights):
    """
    Compute aggregate metrics for tasks in a task group based on 
    """
    wsum = weights.sum()
    if wsum == 0:
        return pd.Series(np.nan, index=df.columns)
    df2 = df.where(pd.isnull, 1) * weights[:,None]
    return (df2.multiply(1.0 / df2.sum(axis=0), axis=1) * df).sum(axis=0)


def load_task_weights(filename, y, label, verbose = False):
    """
    Loads and processes task weights, otherwise raises an error using the label.
    Args:
        df      DataFrame with weights
        y       csr matrix of labels
        label   name for error messages
    
    Returns tuple of
        training_weight
        aggregation_weight
        task_type
    """
    print_dbg(f"\t load_task_weights - filename: {filename} label: {label}", verbose = verbose)
    res = types.SimpleNamespace(training_weight=None, aggregation_weight=None, task_type=None, censored_weight=torch.FloatTensor())

    if y is None:
        assert filename is None, f"\tWeights file {filename} provided for {label}, please add also --{label}"
        res.training_weight = torch.ones(0)
        return res

    if filename is None:
        print_dbg(f"\t load_task_weights - no weights file provided for {label}, training_weights for all  {y.shape[1]} classes set to 1 ", verbose = verbose)
        res.training_weight = torch.ones(y.shape[1])
        return res

    df = pd.read_csv(filename)
    
    ## verify presence of proper column names
    df.rename(columns={"weight": "training_weight"}, inplace=True)
    ## also supporting plural form column names:
    df.rename(columns={c + "s": c for c in ["task_id", "training_weight", "aggregation_weight", "task_type", "censored_weight"]}, inplace=True)

    assert "task_id" in df.columns, "task_id is missing in task info CVS file"
    assert "training_weight" in df.columns, "training_weight is missing in task info CSV file"
    df.sort_values("task_id", inplace=True)

    cols = ["", "task_id", "training_weight", "aggregation_weight", "task_type", "censored_weight"]
    for col in df.columns:
        assert col in cols, f"Unsupported colum '{col}' in task weight file. Supported columns: {cols}."

    assert y.shape[1] == df.shape[0], f"task weights for '{label}' have different size ({df.shape[0]}) to {label} columns ({y.shape[1]})."
    assert (0 <= df.training_weight).all(), f"task weights (for {label}) must not be negative"
    assert (df.training_weight <= 1).all(), f"task weights (for {label}) must not be larger than 1.0"

    assert df.task_id.unique().shape[0] == df.shape[0], f"task ids (for {label}) are not all unique"
    assert (0 <= df.task_id).all(), f"task ids in task weights (for {label}) must not be negative"
    assert (df.task_id < df.shape[0]).all(), f"task ids in task weights (for {label}) must be below number of tasks"

    res.training_weight = torch.FloatTensor(df.training_weight.values)

    if "aggregation_weight" in df:
        assert (0 <= df.aggregation_weight).all(), f"Found negative aggregation_weight for {label}. Aggregation weights must be non-negative."
        res.aggregation_weight = df.aggregation_weight.values

    if "task_type" in df:
        res.task_type = df.task_type.values

    if "censored_weight" in df:
        assert (0 <= df.censored_weight).all(), f"Found negative censored_weight for {label}. Censored weights must be non-negative."
        res.censored_weight = torch.FloatTensor(df.censored_weight.values)

    return res


def class_fold_counts(y_class, folding):
    """
    Create matrix containing number of pos/neg labels falling into the folding scheme

    If the folding consists of 5 unique folds, we sum the number of Y's falling into each fold by 
    Y class. Result is 5 x 100
    """
    folds = np.unique(folding)
    num_pos = []
    num_neg = []
    for fold in folds:
        yf = y_class[folding == fold]
        # print_dbg(f"yf: {yf.shape}")
        num_pos.append( np.array((yf == +1).sum(0)).flatten() )
        num_neg.append( np.array((yf == -1).sum(0)).flatten() )
    return np.row_stack(num_pos), np.row_stack(num_neg) 

def print_metrics(epoch, train_time, metrics_tr, metrics_va, header):
    if metrics_tr is None:
        if header:
            print("Epoch\tlogl_va |  auc_va | aucpr_va | aucpr_cal_va | maxf1_va | tr_time")
        output_fstr = (
            f"{epoch}.\t{metrics_va['logloss']:.5f}"
            f" | {metrics_va['roc_auc_score']:.5f}"
            f" |  {metrics_va['auc_pr']:.5f}"
            f" |  {metrics_va['auc_pr_cal']:.5f}"
            f" |  {metrics_va['f1_max']:.5f}"
            f" | {train_time:6.1f}"
        )
        print(output_fstr)
        return

    ## full print
    if header:
        print("Epoch\tlogl_tr  logl_va |  auc_tr   auc_va | aucpr_tr  aucpr_va | maxf1_tr  maxf1_va | tr_time")
    output_fstr = (
        f"{epoch}.\t{metrics_tr['logloss']:.5f}  {metrics_va['logloss']:.5f}"
        f" | {metrics_tr['roc_auc_score']:.5f}  {metrics_va['roc_auc_score']:.5f}"
        f" |  {metrics_tr['auc_pr']:.5f}   {metrics_va['auc_pr']:.5f}"
        f" |  {metrics_tr['f1_max']:.5f}   {metrics_va['f1_max']:.5f}"
        f" | {train_time:6.1f}"
    )
    print(output_fstr)

def print_table(formats, data):
    for key, fmt in formats.items():
        print(fmt.format(data[key]), end="")

Column = namedtuple("Column", "key size dec title")
columns_start = [
    Column("epoch",         size=6, dec= 0, title="Epoch"),
    Column(None,            size=1, dec=-1, title="|"),
]

columns_task = [
    Column("logloss",       size=8, dec= 5, title="logloss"),
    Column("bceloss",       size=8, dec= 5, title="bceloss"),
    Column("roc_auc_score", size=8, dec= 5, title="aucroc"),
    Column("auc_pr",        size=8, dec= 5, title="aucpr"),
    # Column("auc_pr_cal",  size=9, dec= 5, title="aucpr_cal"),
    Column("f1_max",        size=8, dec= 5, title="f1_max"),
    # Column(None,            size=1, dec=-1, title="|"),
    # Column("rmse",          size=9, dec= 5, title="rmse"),
    # Column("rsquared",      size=9, dec= 5, title="rsquared"),
    # Column("corrcoef",      size=9, dec= 5, title="corrcoef"),
    Column(None,            size=1, dec=-1, title="|"),
]

columns_loss = [
    Column("task1",          size=8, dec= 4, title="t1 loss"),
    Column("task2",          size=8, dec= 4, title="t2 loss"),
    Column("task3",          size=8, dec= 4, title="t3 loss"),
    Column("total",          size=8, dec= 4, title="ttl loss"),
]

columns_end = [
    Column(None,             size=1, dec=-1, title="|"),
    Column("train_time",     size=7, dec= 1, title="tr_time"),
    Column(None,             size=1, dec=-1, title="|")
]

def print_cell(value, size, dec, left, end=""):
    align = "<" if left else ">"
    if type(value) == str:
        print(("{:" + align + str(size) + "}").format(value), end=end)
    else:
        print(("{:" + align + str(size) + "." + str(dec) + "f}").format(value), end=end)

def print_metrics_cr(epoch, train_time, results_tr, results_va, printed_lines, new_header = 20):
    # data = pd.concat([results_va["classification_agg"], results_va["regression_agg"]])
    header = (printed_lines % new_header) == 0
    data = results_va 
    data["train_time"] = train_time
    data["epoch"] = epoch

    if header:
        for i, col in enumerate(columns_start+columns_task+columns_loss+columns_end):
            print_cell(col.title, col.size, dec=0, left=(i==0))
        print()
    
    ## printing row with values
    for i, col in enumerate(columns_start):
        print_cell(data.get(col.key, col.title), col.size, dec=col.dec, left=(i==0))
    
    for i, col in enumerate(columns_task):
        print_cell(data["aggregated"].get(col.key, col.title), col.size, dec=col.dec, left=False)

    for i, col in enumerate(columns_loss):
        print_cell(data["loss"].get(col.key, col.title), col.size, dec=col.dec, left=False)
    
    for i, col in enumerate(columns_end):
        print_cell(data.get(col.key, col.title), col.size, dec=col.dec, left=(i==0))
    
    # print()


    