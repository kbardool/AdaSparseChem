# Copyright (c) 2020 KU Leuven
import os, copy
import torch
from torch.utils.data import Dataset, DataLoader 
import scipy.sparse
import numpy as np
from .sparsechem_utils_dev import load_sparse, load_task_weights, class_fold_counts, fold_and_transform_inputs
from utils.util import print_heading, timestring, print_dbg, print_underline, debug_off, debug_on


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def sparse_collate(batch):
    x_ind  = [b["x_ind"]  for b in batch]
    x_data = [b["x_data"] for b in batch]
    y_ind  = [b["y_ind"]  for b in batch]
    y_data = [b["y_data"] for b in batch]

    ## x matrix
    xrow = np.repeat(np.arange(len(x_ind)), [len(i) for i in x_ind])
    xcol = np.concatenate(x_ind)
    xv   = np.concatenate(x_data)

    ## y matrix
    yrow  = np.repeat(np.arange(len(y_ind)), [len(i) for i in y_ind])
    ycol  = np.concatenate(y_ind).astype(np.int64)

    return {
        "x_ind":  torch.LongTensor([xrow, xcol]),
        "x_data": torch.from_numpy(xv),
        "y_ind":  torch.stack([torch.from_numpy(yrow), torch.from_numpy(ycol)], dim=0),
        "y_data": torch.from_numpy(np.concatenate(y_data)),
        "batch_size": len(batch),
    }


def get_row(csr, row):
    """returns row from csr matrix: indices and values."""
    start = csr.indptr[row]
    end   = csr.indptr[row + 1]
    return csr.indices[start:end], csr.data[start:end]



def patterns_match(x, y):
    if y.shape != x.shape:             return False
    if y.nnz != x.nnz:                 return False
    if (y.indices != x.indices).any(): return False
    if (y.indptr != x.indptr).any():   return False
    return True



class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        '''
        Args:
            X (sparse matrix):  input [n_samples, features_in]
            Y (sparse matrix):  output [n_samples, features_out]
        '''
        assert x.shape[0]==y.shape[0], f"Input has {x.shape[0]} rows, output has {y.shape[0]} rows."

        self.x = x.tocsr(copy=False).astype(np.float32)
        self.y = y.tocsr(copy=False).astype(np.float32)
        # scale labels from {-1, +1} to {0, 1}, zeros are stored explicitly
        self.y.data = (self.y.data + 1) / 2.0

    def __len__(self):
        return(self.x.shape[0])

    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        x_start = self.x.indptr[idx]
        x_end   = self.x.indptr[idx + 1]
        x_indices = self.x.indices[x_start:x_end]
        x_data    = self.x.data[x_start:x_end]

        y_start = self.y.indptr[idx]
        y_end   = self.y.indptr[idx + 1]
        y_indices = self.y.indices[y_start:y_end]
        y_data = self.y.data[y_start:y_end]

        return {
            "x_ind":  x_indices,
            "x_data": x_data,
            "y_ind":  y_indices,
            "y_data": y_data,
        }

    def batch_to_x(self, batch, dev):
        """Takes 'xind' and 'x_data' from batch and converts them into a sparse tensor.
        Args:
            batch  batch
            dev    device to send the tensor to
        """
        return torch.sparse_coo_tensor(
                batch["x_ind"].to(dev),
                batch["x_data"].to(dev),
                size=[batch["batch_size"], self.x.shape[1]])




class ClassRegrSparseDataset(Dataset):

    def __init__(self, x, y_class, y_regr, y_censor=None, indicies = None):
        '''
        Creates dataset for two outputs Y.
        Args:
            x (sparse matrix):        input [n_sampes, features_in]
            y_class  (sparse matrix): class data [n_samples, class_tasks]
            y_regr   (sparse matrix): regression data [n_samples, regr_tasks]
            y_censor (sparse matrix): censoring matrix, for regression data [n_samples, regr_task]
        '''
        if indicies is not None:
            x = x[indicies]
            if y_class is not None:
                y_class = y_class[indicies]
            if y_regr is not None:
                y_regr  = y_regr[indicies]
            if y_censor is not None:
                y_censor= y_censor[indicies]

        if y_regr is not None:
            if y_censor is None:
                y_censor = scipy.sparse.csr_matrix(y_regr.shape)
            else:
                assert y_regr.shape==y_censor.shape, f"Regression data has shape {y_regr.shape} and censor data has shape {y_censor.shape[0]}. Must be equal."

            self.y_censor = y_censor.tocsr(copy=False).astype(np.float32)
            assert y_regr.shape[1] > 0, "No labels provided (both y_class and y_regr are missing)"
            assert x.shape[0]==y_regr.shape[0], f"Input has {x.shape[0]} rows and regression data has {y_regr.shape[0]} rows. Must be equal."
            self.y_regr   = y_regr.tocsr(copy=False).astype(np.float32)

            if self.y_censor.nnz > 0:
                assert patterns_match(self.y_regr, self.y_censor), "y_regr and y_censor must have the same shape and sparsity pattern (nnz, indices and indptr)"
                d = self.y_censor.data
                assert ((d == -1) | (d == 0) | (d == 1)).all(), "Values of regression censor (y_censor) must be either -1, 0 or 1."
        else:
            self.y_regr = scipy.sparse.csr_matrix((0,0))
            self.y_censor = scipy.sparse.csr_matrix((0,0))
        
        assert y_class.shape[1]  > 0, "No labels provided (both y_class and y_regr are missing)"
        assert x.shape[0]==y_class.shape[0], f"Input has {x.shape[0]} rows and class data {y_class.shape[0]} rows. Must be equal."

        self.x        = x.tocsr(copy=False).astype(np.float32)
        self.y_class  = y_class.tocsr(copy=False).astype(np.float32)


        # scale labels from {-1, +1} to {0, 1}, zeros are stored explicitly
        self.y_class.data = (self.y_class.data + 1) / 2.0

    def __len__(self):
        return(self.x.shape[0])

    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y_class.shape[1] + self.y_regr.shape[1]

    @property
    def class_output_size(self):
        return self.y_class.shape[1]

    @property
    def regr_output_size(self):
        return self.y_regr.shape[1]


    def __getitem__(self, idx):
        out = {}
        
        out["x_ind"], out["x_data"] = get_row(self.x, idx)

        if self.class_output_size > 0:
            out["yc_ind"], out["yc_data"] = get_row(self.y_class, idx)

        if self.regr_output_size > 0:
            out["yr_ind"], out["yr_data"] = get_row(self.y_regr, idx)
            if self.y_censor.nnz > 0:
                out["ycen_ind"], out["ycen_data"] = get_row(self.y_censor, idx)

        return out


    def batch_to_x(self, batch, dev):
        """Takes 'xind' and 'x_data' from batch and converts them into a sparse tensor.
        Args:
            batch  batch
            dev    device to send the tensor to
        """
        return torch.sparse_coo_tensor(
                batch["x_ind"].to(dev),
                batch["x_data"].to(dev),
                size=[batch["batch_size"], self.x.shape[1]])

    def collate(self, batch):
        lists = {}
        for key in batch[0].keys():
            lists[key] = [b[key] for b in batch]

        out = {}
        out["x_ind"]  = to_idx_tensor(lists["x_ind"])
        out["x_data"] = torch.from_numpy(np.concatenate(lists["x_data"]))

        if "yc_ind" in lists:
            out["yc_ind"]  = to_idx_tensor(lists["yc_ind"])
            out["yc_data"] = torch.from_numpy(np.concatenate(lists["yc_data"]))

        if "yr_ind" in lists:
            out["yr_ind"]  = to_idx_tensor(lists["yr_ind"])
            out["yr_data"] = torch.from_numpy(np.concatenate(lists["yr_data"]))

        if "ycen_ind" in lists:
            out["ycen_ind"]  = to_idx_tensor(lists["ycen_ind"])
            out["ycen_data"] = torch.from_numpy(np.concatenate(lists["ycen_data"]))
        else:
            out["ycen_ind"]  = None
            out["ycen_data"] = None

        out["batch_size"] = len(batch)
        return out



class ClassRegrSparseDataset_v3(Dataset):
 
    def __init__(self, opt, x = None, y_files = None, folding =  None, y_censor=None, 
                split_ratios = None, ratio_index = None,  index = None, verbose = False):
        '''
        Creates the 
        Creates dataset for two outputs Y.
        Args:
            x (sparse matrix):        input [n_sampes, features_in]
            y_class  (sparse matrix): class data [n_samples, class_tasks]
            y_regr   (sparse matrix): regression data [n_samples, regr_tasks]
            y_censor (sparse matrix): censoring matrix, for regression data [n_samples, regr_task]
        '''
        assert len(opt['dataload']['y_tasks']) == len(opt['tasks']), "List of y_tasks and tasks must contain same number of elements"
        
        if verbose:
            print_heading(f" {timestring()}  Create new  {self.name()} instance ", verbose = verbose)
            print_dbg(f" verbose        : {verbose}", verbose = verbose)

        dataroot      = opt['dataload']['dataroot']
        y_files       = opt['dataload']['y_tasks']
        yc_weights    = opt['dataload']['weights_class']

        yc_weights = [None for i in (y_files)]  if yc_weights is None else yc_weights

        self.ecfp     = load_sparse(dataroot, opt['dataload']['x'])
        self.folding  = np.load(os.path.join(dataroot, opt['dataload']['folding']))
        assert self.ecfp.shape[0] == self.folding.shape[0], "x and folding must have same number of rows" 
        
        if index is None:
            total_input   = self.ecfp.shape[0]
            ranges        = (np.cumsum([0]+split_ratios)* total_input).astype(np.int32)
            index = np.arange(ranges[ratio_index], ranges[ratio_index+1])
            size = len(index)
            print_dbg(f" totalInput     : {total_input} ", verbose = verbose)
            print_dbg(f" ranges         : {ranges}      ", verbose = verbose) 
            print_dbg(f" ratio_index    : {ratio_index}", verbose = verbose)
            print_dbg(f" select rows    : {ranges[ratio_index]:6d} to {ranges[ratio_index+1]:6d}", verbose = verbose)
            print_dbg(f" rows in dataset: {size} ", verbose = verbose)
            print_dbg(f" indexes used   : {index}", verbose = verbose)
 

        self.class_tasks = 0 
        self.regr_tasks  = 0
        self.y_class_list = [] 
        self.tasks_weights_list = []
        self.batch_id = 0 
        
        if verbose:
            print_dbg(f" Input          : {opt['dataload']['x']:32s} - type : {type(self.ecfp)} shape : {self.ecfp.shape}", verbose = verbose)
            print_dbg(f" Folding        : {opt['dataload']['folding']:32s} - type : {type(self.folding)} shape : {self.folding.shape}", verbose = verbose)
            print_dbg(f" Index len      : {'-':32s} {len(index) :6d}  - {(index)} ", verbose = verbose)
            print_dbg(f" yc_weights     : {yc_weights}", verbose = verbose)
        
        ## Load label files for each task
        self.ecfp = fold_and_transform_inputs(self.ecfp, folding_size=opt['dataload']['fold_inputs'], 
                                                         transform=opt['dataload']['input_transform'])
        if index is not None :
            ## removing test data
            self.x = copy.copy(self.ecfp[index])
            self.folding = copy.copy(self.folding[index])

        ## Iterate through list of y_files and set up Y label file for each task 
        for task_id, y_file in enumerate(y_files):
            
            print_dbg(f"\n* load y_task{task_id+1} file : {y_file} . . .", verbose = verbose)
            y_temp =  load_sparse(dataroot, y_file)


            if y_temp is None:
                y_temp =  scipy.sparse.csr_matrix((self.ecfp.shape[0], 0))
                stat = 'Not Supplied - Created'
            else:
                # self.y_class  = y_class.tocsr(copy=False).astype(np.float32)
                stat = 'Supplied'
            
            ## Get number of positive / neg and total for each classes
            ## ensure there are no labels besides {-1,0,+1}
            # y_class = getattr(self, 'y_task{:d}'.format(i))
            num_pos    = np.array((y_temp == +1).sum(0)).flatten()
            num_neg    = np.array((y_temp == -1).sum(0)).flatten()
            num_nonzero= np.array((y_temp !=  0).sum(0)).flatten()
            # print(num_pos, num_neg, num_nonzero)
            
            
            if (num_nonzero != num_pos + num_neg).any():
                raise ValueError("\t For classification all y values (--y_class/--y) must be 1 or -1.")
            else:
                print_dbg(f"\t y_task[{task_id+1}]  All y values are 0, 1, or -1.", verbose = verbose)

            if index is not None :
                y_temp = y_temp[index]
                # for i, y_file in enumerate(y_tasks, 1):
                    # setattr(self, 'y_task{:d}'.format(i), eval('y_task{:d}[index].format(i)'))                 
                # y_regr  = y_regr[keep]
                # self.y_censor= self.y_censor[keep]
            

            # setattr(self, 'y_task{:d}'.format(i), y_class.tocsr(copy=False).astype(np.float32)) 
            # print(f"\t  y_task{i}  {stat:22s} - type : {eval('type(self.y_task{:d})'.format(i))}"\
                    # f" shape : {eval('self.y_task{:d}.shape'.format(i))}"  )




            ##-----------------------------------------------------------------------------------
            ## Classification task aggregation weights 
            ##-----------------------------------------------------------------------------------
            weights_temp = load_task_weights(yc_weights[task_id], y=y_temp, label=f"y_task{task_id+1}", verbose = verbose)
            if weights_temp.aggregation_weight is None:
                '''
                fold classes 
                '''
                ## using min_samples rule
                fold_pos, fold_neg = class_fold_counts(y_temp, self.folding)
                n = opt['dataload']['min_samples_class']
                weights_temp.aggregation_weight = ((fold_pos >= n).all(0) & (fold_neg >= n)).all(0).astype(np.float64)
                print_dbg(f"\t tasks_class.aggregation_weight WAS NOT passed ", verbose = verbose)
                print_dbg(f"\t min_samples_class: {opt['dataload']['min_samples_class']}", verbose = verbose)
                print_dbg(f"\t Class fold counts: \n  fold_pos:\n{fold_pos}  \n\n  fold_neg:\n{fold_neg}", verbose = verbose)
            else:
                print_dbg(f"\t  tasks_class.aggregation_weight passed ", verbose = verbose)
                
            print_dbg(f"\t task_weights.aggregation_weight.shape: {weights_temp.aggregation_weight.shape} - {weights_temp.aggregation_weight}", verbose = verbose)

            ##-----------------------------------------------------------------------------------
            ## Load censor file if present 
            ##-----------------------------------------------------------------------------------
            # y_censor = load_sparse(dataroot, y_censor[i])
            # if y_censor is None:
            #     stat = 'Not Supplied - Created'
            #     # setattr(self, 'y_censor', scipy.sparse.csr_matrix(self.y_task1.shape))
            #     self.y_censor = scipy.sparse.csr_matrix(self.y_task1.shape)
            # else:
            #     stat = 'Supplied'
            # print(f" y_censor {stat:22s} - type : {type(self.y_censor)}  shape : {self.y_censor.shape}")
            # setattr(self, "y_censor" , self.y_censor.tocsr(copy=False).astype(np.float32) )

            # assert y_class.shape[1] + y_regr.shape[1] > 0, "No labels provided (both y_class and y_regr are missing)"
            # assert x.shape[0]==y_class.shape[0], f"Input has {x.shape[0]} rows and class data {y_class.shape[0]} rows. Must be equal."
            # assert x.shape[0]==y_regr.shape[0], f"Input has {x.shape[0]} rows and regression data has {y_regr.shape[0]} rows. Must be equal."
            # assert y_regr.shape==y_censor.shape, f"Regression data has shape {y_regr.shape} and censor data has shape {y_censor.shape[0]}. Must be equal."

            # self.y_class = y_class.tocsr(copy=False).astype(np.float32)
            # self.y_regr  = y_regr.tocsr(copy=False).astype(np.float32)
            # self.y_censor = y_censor.tocsr(copy=False).astype(np.float32)




            ##-----------------------------------------------------------------------------------
            ## Regression task aggregation weights 
            ##-----------------------------------------------------------------------------------
            # if tasks_regr.aggregation_weight is None:
            #     if y_censor.nnz == 0:
            #         y_regr2 = y_regr.copy()
            #         y_regr2.data[:] = 1
            #     else:
            #         ## only counting uncensored data
            #         y_regr2      = y_censor.copy()
            #         y_regr2.data = (y_regr2.data == 0).astype(np.int32)
            #     fold_regr, _ = sc.class_fold_counts(y_regr2, folding)
            #     del y_regr2
            #     tasks_regr.aggregation_weight = (fold_regr >= args.min_samples_regr).all(0).astype(np.float64)        

            # scale labels from {-1, +1} to {0, 1}, zeros are stored explicitly

            y_temp.data  = (y_temp.data + 1) / 2.0
            # if y_cat_columns is not None:
                # self.y_cat_class.data = (self.y_cat_class.data + 1) / 2.0
            
            ## Finally, append to list of y_class files
            self.y_class_list.append(y_temp)
            self.tasks_weights_list.append(weights_temp)
            self.class_tasks += 1
            
            if verbose:
                print_dbg(f"\t y_task[{task_id+1}]  {stat:22s} - type : {type(y_temp)}    shape : {y_temp.shape}", verbose = verbose)
                print_dbg(f"\t y_task[{task_id+1}]  {stat:22s} - type : {type(self.y_class_list[task_id])}    shape : {self.y_class_list[task_id].shape}"  , verbose = verbose)
                print_dbg(f"\t Input dimension       : {self.ecfp.shape[1]}", verbose = verbose)
                print_dbg(f"\t # samples             : {self.ecfp.shape[0]}", verbose = verbose)
                print_dbg(f"\t # classification tasks: {self.class_tasks}", verbose = verbose)
                print_dbg(f"\t # regression tasks    : {self.regr_tasks}", verbose = verbose)
                # print_dbg(f"Using {(tasks_regr.aggregation_weight > 0).sum()} regression tasks for calculating metrics (RMSE, Rsquared, correlation).")
                # print_dbg(f" Using {(tasks_class.aggregation_weight > 0).sum()} classification tasks for calculating aggregated metrics (AUCROC, F1_max, etc).")
        
        print_heading(f"{self.name()} Create complete", verbose = verbose)
        return 

    def __getitem__(self, idx, verbose = False):
        """
        returns: 
        out[x_ind] : indices to columns of X row containing data 
        out[x_data]: data corresponding to columns in out[x_ind]

        out[task0_ind]:   indices to columns of Y0 (Labels of task 0)row containing data
        out[task0_data]:  data corresponding to columns in out[task0_ind]
        ...
        ...  repeats from task0 to task(n-1) for n tasks 

        """
        out = {}

        out["row_id"] = idx
        out["x_ind"], out["x_data"] = get_row(self.x, idx)
        print_dbg(f" x indexes: {out['x_ind'].shape}   {out['x_ind'][:10]}", verbose = verbose)
        print_dbg(f" x data   : {out['x_data'].shape}  {out['x_data'][:10]}", verbose = verbose)

        for i in range(len(self.y_class_list)):
            ind_key     = "task{:d}_ind".format(i+1)
            data_key    = "task{:d}_data".format(i+1)
            weights_key = "task{:d}_weights".format(i+1)
            out[ind_key], out[data_key]= get_row(self.y_class_list[i], idx)
            out[weights_key] = self.tasks_weights_list[i]
           
            print_dbg(f"\t  load task[{i}] file  . . . {ind_key}    {data_key}", verbose = verbose)
            print_dbg(f"\t  out[{ind_key}] : {out[ind_key].shape}   {out[ind_key]}", verbose = verbose)
            print_dbg(f"\t  out[{data_key}]: {out[data_key].shape}  {out[data_key]}", verbose = verbose)


        # if self.regr_output_size > 0:
            # out["yr_ind"], out["yr_data"] = get_row(self.y_regr, idx)
            # if self.y_censor.nnz > 0:
                # out["ycen_ind"], out["ycen_data"] = get_row(self.y_censor, idx)
        return out

    def collate(self, batch, verbose = False):
        """
        batch : list of N objects  (N = batch size) 
                each object is the output of __getitem__
        """
        lists = {}

        for key in batch[0].keys():
            lists[key] = [b[key] for b in batch]

        out = {}
        
        sum = np.sum([len(i) for i in lists["x_ind"]])
        
        out["x_ind"]  = to_idx_tensor(lists["x_ind"])
        out["x_data"] = torch.from_numpy(np.concatenate(lists["x_data"]))
        out["row_id"] = lists['row_id']
        print_dbg(f" output keys: {lists.keys()}", verbose = verbose)
        print_dbg(f" out[row_id]: {out['row_id']}", verbose = verbose)
        print_dbg(f" len of lists[x_ind] {len(lists['x_ind'])}   sum:{sum}", verbose = verbose)
        print_dbg(f" len of out[x_ind]   {out['x_ind'].shape}", verbose = verbose)
        print_dbg(f" len of out[x_data]  {out['x_data'].shape} ", verbose = verbose)
        

        for i in range(len(self.y_class_list)):
            ind_key = "task{:d}_ind".format(i+1)
            data_key = "task{:d}_data".format(i+1)
            weights_key = "task{:d}_weights".format(i+1)
         
            out[ind_key]  = to_idx_tensor(lists[ind_key]) 
            out[data_key] = torch.from_numpy(np.concatenate(lists[data_key]))
            out[weights_key] = self.tasks_weights_list[i]

            print_dbg(f"\n\t load task[{i+1}] file ", verbose = verbose)
            print_dbg(f"\t out[{ind_key}]    shape  {out[ind_key].shape}", verbose = verbose)
            print_dbg(f"\t out[{data_key}]   shape  {out[data_key].shape} ", verbose = verbose)
            print_dbg(f"\t out[{weights_key}].training_weight      {out[weights_key].training_weight} ", verbose = verbose)
            print_dbg(f"\t out[{weights_key}].aggregation_weight   {out[weights_key].aggregation_weight} ", verbose = verbose)
    
    #     if "yr_ind" in lists:
    #         out["yr_ind"]  = to_idx_tensor(lists["yr_ind"])
    #         out["yr_data"] = torch.from_numpy(np.concatenate(lists["yr_data"]))

    #     if "ycen_ind" in lists:
    #         out["ycen_ind"]  = to_idx_tensor(lists["ycen_ind"])
    #         out["ycen_data"] = torch.from_numpy(np.concatenate(lists["ycen_data"]))
    #     else:
    #         out["ycen_ind"]  = None
    #         out["ycen_data"] = None

        out["batch_size"] = len(batch)
        return out

        
    def __len__(self):
        return(self.x.shape[0])
   
    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y_class.shape[1] + self.y_regr.shape[1]

    @property
    def class_output_size(self):
        return [i.shape[1] for i in self.y_class_list]

    @property
    def regr_output_size(self):
        return self.y_regr.shape[1]    

    def name(self):
        return 'Chembl_23_Dev'



def to_idx_tensor(idx_list):
    """Turns list of lists into a  [2, num_lists] tensor of coordinates"""
    ## idx_lists contains batch_size sub lists, each corresponding to one row.
    ## following line takes [0,1,...,15] and repeats each element n times, where 
    ## is the number of elements  in the corresponding idx_list member.  
    xrow = np.repeat(np.arange(len(idx_list)), [len(i) for i in idx_list])
    xcol = np.concatenate(idx_list)

    ## added np.array() to convert list of arrays to numpy array before converting to Tensor.
    # print(f" to_idx_tensor: {type(idx_list)}   xrow: {type(xrow)}    xcol: {type(xcol)}")

    return torch.LongTensor(np.array([xrow, xcol]))
