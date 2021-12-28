import os 
import numpy as np
import torch
import types
import pandas as pd
from numpy.core.numeric import full
import scipy.sparse
import scipy.io
import scipy.special
from utils.util import print_heading, print_dbg, debug_on, debug_off

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