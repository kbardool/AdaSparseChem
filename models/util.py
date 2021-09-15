import sys
sys.path.insert(0, '..')
import torch
import numpy as np
from utils.flops_benchmark import add_flops_counting_methods


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


##########################################################################################################
##
##  SparseChem functions 
##
##########################################################################################################
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
