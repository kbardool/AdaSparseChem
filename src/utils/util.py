import os
import sys
import argparse
import yaml
import random
import pickle
import pprint
from IPython import get_ipython
import wandb.util as wbutils
import torch
from torchvision import utils as vu
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
timestring = lambda : datetime.now().strftime('%F %H:%M:%S:%f')
timestring()

# ---------------------------------------------------------------------------------------
# Decorators 
# ---------------------------------------------------------------------------------------
def debug_on(fn):
    def wrapper(*args, **kwargs):
        # print(f"debug_on wrapper for function {fn.__name__} - verbose is {kwargs.get('verbose', 'not provided')}")
        kwargs['verbose'] = True
        return fn(*args, **kwargs)
    return wrapper

def debug_off(fn):
    def wrapper(*args, **kwargs):
        # print(f"debug_off wrapper for function {fn.__name__} - verbose is {kwargs.get('verbose', 'not provided')}")
        kwargs['verbose'] = False
        return fn(*args, **kwargs)
    return wrapper

def debug_off_m(method):
    def wrapper(*args, **kwargs):
        # print(f"debug_off_m wrapper for function {fn.__name__} - verbose is {kwargs.get('verbose', 'not provided')}")
        kwargs['verbose'] = False
        return method(*args, **kwargs)
    return wrapper

# ---------------------------------------------------------------------------------------
# Display routines 
# ---------------------------------------------------------------------------------------
def vprint(s="", verbose = False):
    if verbose:
        print(s)


def print_separator(text, total_len=50):
    print('#' * total_len)
    text = f" {text} "
    text_len = len(text)
    left_width = (total_len - text_len )//2
    right_width = total_len - text_len - left_width
    print("#" * left_width + text + "#" * right_width)
    print('#' * total_len)

def print_dbg(text, verbose = False):
    if verbose:
        print(text)

# @debug_off
def print_heading(text,  verbose = False, force = False, out=[sys.stdout]):
    len_ttl = max(len(text)+4, 50)
    len_ttl = min(len_ttl, 120)
    if verbose or force: 
        for file in out:
            print('-' * len_ttl, file = file)
            print(f"{text}", file = file)
            # left_width = (total_len - len(text))//2
            # right_width = total_len - len(text) - left_width
            # print("#" * left_width + text + "#" * right_width)
            print('-' * len_ttl, '\n', file=file)


def print_underline(text,  verbose = False, out=[sys.stdout]):
    len_ttl = len(text)+2
    if verbose:
        for file in out:
            print(f"\n {text}", file=file)
            print('-' * len_ttl, file=file)


def print_yaml(opt):
    lines = []
    if isinstance(opt, dict):
        # lines += [f"CALL print_yaml with {key}"]
        lines += [""]
        for key in opt.keys():
            tmp_lines = print_yaml(opt[key])
            tmp_lines = ["%s.%s" % (key, line) for line in tmp_lines]
            lines += tmp_lines
    else:
        lines = [" : " + str(opt)]
    return lines


def print_yaml2(opt, key = ""):
    lines = []
    if isinstance(opt, dict):
        # lines += [f"CALL print_yaml with {key}"]
        if key != "":
            lines += ["", key, '-'*len(key)]
        for key in opt.keys():
            tmp_lines = print_yaml2(opt[key], key)
            tmp_lines = ["%s" % (line) for line in tmp_lines]
            lines += tmp_lines
    else:
        lines = [f"{key:>20s} : {str(opt)}"]
    return lines

def print_loss(losses, title='Iteration', out = [sys.stdout]):
    """
    print loss summary line 
    """
    loss_display = f"{title} -  Total Loss: {losses['total']['total']:.4f}     \nTask: {losses['task']['total']:.4f}" 
    if 'sparsity' in losses:
        loss_display += f"   Sparsity: {losses['sparsity']['total']:.5e}    Sharing: {losses['sharing']['total']:.5e} "
    # loss_display = f"{title}  {current_iter} \nLosses: Task: {self.losses['losses']['total']:.4f}   " 
    # if 'sparsity' in self.losses:
    #     loss_display += f"Spar: {self.losses['sparsity']['total']:.5e}   Shr: {self.losses['sharing']['total']:.5e}"
    # loss_display += f"   Ttl: {self.losses['total']['total']:.4f}"
    print_to(loss_display, out = out)

def print_to(data, out = [sys.stdout]):
    for file in out:
        print(data, file = file)

def show_batch(batch):
    normed = batch * 0.5 + 0.5
    is_video_batch = len(normed.size()) > 4

    if is_video_batch:
        rows = [vu.make_grid(b.permute(1, 0, 2, 3), nrow=b.size(1)).numpy() for b in normed]
        im = np.concatenate(rows, axis=1)
    else:
        im = vu.make_grid(normed).numpy()

    im = im.transpose((1, 2, 0))

    plt.imshow(im)
    plt.show(block=True)


def listopt(opt, f=None):
    """Pretty-print a given namespace either to console or to a file.

    :param opt: A namespace
    :param f: The file descriptor to write to. If None, write to console
    """
    args = vars(opt)

    if f is not None:
        f.write('------------ Options -------------\n')
    else:
        print('------------ Options -------------')

    for k, v in sorted(args.items()):
        if f is not None:
            f.write('%s: %s\n' % (str(k), str(v)))
        else:
            print('%s: %s' % (str(k), str(v)))

    if f is not None:
        f.write('-------------- End ----------------\n')
    else:
        print('-------------- End ----------------')


def write_metrics_txt(log_name, epoch, iteration, elapsed, losses):
    sorted_keys = sorted( [ 'task', 'task_mean','parms', 'sharing', 'sparsity', 'total'])
    
    for key in sorted_keys:
        if key not in losses:
            continue
        message = 'epoch: %4d   iter: %4d, timestamp: %s wall clock time: %7.3f  %12s :' % (epoch, iteration, timestring(), elapsed, key)
        if isinstance(losses[key], dict):
            for subkey, value  in losses[key].values():
                message += ' %s: %s ' % (subkey, str(value))
        elif (isinstance(losses[key], float)):
            message += ' %s: %.3f ' % (key, losses[key])

    with open(log_name, 'a') as log_file:
        log_file.write('%s \n' % message)


def write_loss_csv_heading(log_name,  losses):
    message = ' epoch, iteration, timestamp,elapsed,' 
    sorted_keys = sorted(['task', 'task_mean', 'parms', 'sharing', 'sparsity', 'total'])
    
    for key in sorted_keys:
        if key not in losses:
            continue

        if isinstance(losses[key], dict):
            for subkey in sorted(losses[key].keys()):
                message += f"{key:s}.{subkey:s}," 
        elif (isinstance(losses[key], float)):
                message += f"{key:s}.{key},"
    
    with open(log_name, 'a') as log_file:
        log_file.write('%s \n' % message.rstrip(" ,"))


def write_metrics_csv(log_name, epoch, iteration, elapsed, losses):
    message = '%4d,%4d,%26s,%6.3f,' % (epoch, iteration, timestring(), elapsed)
    sorted_keys = sorted([ 'task', 'task_mean', 'parms', 'sharing', 'sparsity', 'total'])
    
    for key in sorted_keys:
        if key not in losses:
            continue

        if isinstance(losses[key], dict):
            for subkey in sorted(losses[key].keys()):
                message += f"{losses[key][subkey]}," 
        elif (isinstance(losses[key], float)):
                message += f"{losses[key]},"
    
    with open(log_name, 'a') as log_file:
        log_file.write('%s \n' % message.rstrip(" ,"))


def save_to_pickle(data, path, filename, verbose = False):
    save_path = os.path.join(path, filename)
    print_dbg(f" save_to_pickle(): save data to {save_path}", verbose = verbose)
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle( path, filename, verbose = False):

    load_path = os.path.join(path, filename)

    print_dbg(f" load_metrics(): load  data from {load_path}", verbose = verbose)
    with open(load_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def images_to_visual(tensor):
    generated = torch.clamp(tensor.data.cpu(), min=-1, max=1)
    generated = (generated + 1) / 2
    return generated


def videos_to_visual(tensor):
    # [batch, c, t, h, w] -> [batch, t, c, h, w] -> [batch * t, c, h, w]
    s = tensor.data.size()
    generated = tensor.data.permute(0, 2, 1, 3, 4).view(-1, s[1], s[3], s[4])
    generated = (generated + 1) / 2
    return generated


def videos_to_numpy(tensor):
    # [batch, c, t, h, w] -> [batch, t, h, w, c]
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 4, 1).clip(-1, 1)
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def rgb2gray(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = None
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            gray_ = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
            gray = torch.unsqueeze(gray_, 0)
        elif image.dim() == 4:
            gray_ = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
            gray = torch.unsqueeze(gray_, 1)
        else:
            raise ValueError('The dimension of tensor is %d not supported in rgb2gray' % image.dim())
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 3:
                gray_ = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
                gray = np.expand_dims(gray_, 0)
            else:
                gray_ = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
                gray = np.expand_dims(gray_, -1)
        elif image.ndim == 4:
            gray_ = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
            gray = np.expand_dims(gray_, 1)
        else:
            raise ValueError('The dimension of np.ndarray is %d not supported in rgb2gray' % image.ndim)
    return gray


def one_hot(category_labels, num_categories):
    '''

    :param category_labels: a np.ndarray or a tensor with size [batch_size, ]
    :return: a tensor with size [batch_size, num_categories]
    '''
    if isinstance(category_labels, torch.Tensor):
        labels = category_labels.cpu().numpy()
    else:
        labels = category_labels
    num_samples = labels.shape[0]
    one_hot_labels = np.zeros((num_samples, num_categories), dtype=np.float32)  # [num_samples. dim_z_category]
    one_hot_labels[np.arange(num_samples), labels] = 1
    one_hot_labels = torch.from_numpy(one_hot_labels)

    if torch.cuda.is_available():
        one_hot_labels = one_hot_labels.cuda()
    return one_hot_labels


def compute_grad(inputs):
    """
    :param inputs: a tensor with size [batch_size, c, h, w]
    :return: a tensor with size [batch_size, 2c, h, w]
    """
    batch_size, n_channels, h, w = int(inputs.size()[0]), int(inputs.size()[1]), int(inputs.size()[2]), int(inputs.size()[3])
    grad = torch.zeros((batch_size, 2 * n_channels, h, w))
    grad[:, : n_channels, :-1] = (inputs[:, :, :-1] - inputs[:, :, 1:])/2
    grad[:, n_channels:, :, :-1] = (inputs[:, :, :, :-1] - inputs[:, :, :, 1:])/2
    if torch.cuda.is_available():
        grad = grad.cuda()
    return grad


class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Conv3d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:    
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)

def display_config(opt, key = ""):
    output = print_yaml2(opt)
    for line in output:
        print(line)

def write_config_report(opt, output = None, filename = 'run_params.txt', mode = 'w+'):
    with open(os.path.join(opt['paths']['log_dir'], filename), mode= mode) as f:
        if output is None:
            output = print_yaml2(opt)
            for line in output:
                f.writelines(line+"\n")
        else:
                f.writelines(output+"\n")


##
##  Command line and YAML configuration files
##
def get_command_line_args(input = None, display = True):
    """ get and parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config"           , required=True, help="Path for the config file")
    parser.add_argument("--exp_id"           , type=str,   help="experiment unqiue id, used by wandb - defaults to wandb.util.generate_id()")
    parser.add_argument("--exp_name"         , type=str,   help="experiment name, used as folder prefix and wandb name, defaults to mmdd_hhmm")
    parser.add_argument("--folder_sfx"       , type=str,   help="experiment folder suffix, defaults to None")
    parser.add_argument("--exp_desc"         , type=str,   nargs='+', default=[] , help="experiment description")
    parser.add_argument("--hidden_sizes"     , type=int,   nargs='+', default=[] , help="hiddenlayers sizes")
    parser.add_argument("--tail_hidden_size" , type=int,   nargs='+', default=[] , help="tail hidden layers sizes")
    parser.add_argument("--warmup_epochs"    , type=int,   help="Warmup epochs")
    parser.add_argument("--training_epochs"  , type=int,   help="Training epochs")
    parser.add_argument("--seed_idx"         , type=int,   default=0, help="Seed index - default is 0")
    parser.add_argument("--batch_size"       , type=int,   help="Batchsize - default read from config file")
    parser.add_argument("--first_dropout"    , type=float, help="Dropout ratio for Sparse Input Layer")
    parser.add_argument("--middle_dropout"   , type=float, help="Dropout ratio for middle (hidden) layers")
    parser.add_argument("--last_dropout"     , type=float, help="Dropout ratio for final (task head) layers")
    parser.add_argument("--backbone_lr"      , type=float, help="Backbone Learning Rate Override - default read from config file")
    parser.add_argument("--task_lr"          , type=float, help="Task Heads Learning Rate Override - default read from config file")
    parser.add_argument("--policy_lr"        , type=float, help="Policy Net Learning Rate Override - default read from config file")
    parser.add_argument("--decay_lr_rate"    , type=float, help="LR Decay Rate Override - default read from config file")
    parser.add_argument("--decay_lr_freq"    , type=float, help="LR Decay Frequency Override - default read from config file")
    parser.add_argument("--lambda_sparsity"  , type=float, help="Sparsity Regularization - default read from config file")
    parser.add_argument("--lambda_sharing"   , type=float, help="Sharing Regularization - default read from config file")
    parser.add_argument("--gpu_ids"          , type=int,   nargs='+', default=[0],  help="GPU Device Ids")
    # parser.add_argument("--policy"           , action="store_true",  help="Train policies")
    parser.add_argument("--skip_residual"    , default=False, action="store_true",  help="Skip all residual layers")
    parser.add_argument("--skip_hidden"      , default=False, action="store_true",  help="Skip all hidden layers")
    parser.add_argument("--resume"           , default=False, action="store_true",  help="Resume previous run")
    parser.add_argument("--cpu"              , default=False, action="store_true",  help="CPU instead of GPU")
    parser.add_argument("--min_samples_class", type=int,   help="Minimum number samples in each class and in each fold for AUC "\
                                               "calculation (only used if aggregation_weight is not provided in --weights_class)")    
    
    if input is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input)
    
    if args.resume:
        assert args.exp_id is not None and args.exp_name is not None, " exp_id & exp_name must be provided when specifying --resume"
    
    args.exp_desc = ' '.join(str(e) for e in args.exp_desc)
    
    if args.exp_id is None:
        args.exp_id = wbutils.generate_id()
    
    if display:
        print_underline(' command line parms : ', True)
        for key, val in vars(args).items():
            print(f" {key:.<25s}  {val}")
        print('\n\n')
    
    return args

# def read_yaml():
#     # get command line arguments
#     args = get_command_line_args()
#     # torch.cuda.set_device(args.gpu)
#     with open(args.config) as f:
#         opt = yaml.load(f)
#     opt['exp_name'] = args.exp_name
#     opt['cpu'] = args.cpu
#     return opt, args.gpus, args.exp_ids
 

def read_yaml(args = None, exp_name = None):
    """ read yaml passing command line arguments """
    if args is None:
        args = get_command_line_args()

    with open(args.config) as f:
        opt = yaml.safe_load(f)

    opt['config']= args.config

    opt["exp_id"] = args.exp_id 


    if exp_name is not None:
        opt['exp_name_pfx'] = exp_name
    elif args.exp_name is not None:
        opt['exp_name_pfx'] = args.exp_name
    else: 
        opt['exp_name_pfx'] = datetime.now().strftime("%m%d_%H%M")    
    
    opt['exp_name']   = opt['exp_name_pfx'] 
    
    if args.folder_sfx is not None:
        opt['folder_sfx'] = args.folder_sfx

    if args.exp_desc is not None:
        opt['exp_description'] = args.exp_desc 

    if args.hidden_sizes is not None:
        opt['hidden_sizes'] = args.hidden_sizes
    if args.tail_hidden_size is not None:
        opt['tail_hidden_size'] = args.tail_hidden_size

    if args.warmup_epochs is not None:
        opt['train']['warmup_epochs'] = args.warmup_epochs
    if args.training_epochs is not None:
        opt['train']['training_epochs'] = args.training_epochs

    opt["random_seed"] = opt["seed_list"][args.seed_idx]
    
    if args.batch_size is not None:
        opt['train']['batch_size']


    if args.first_dropout is not None:
        opt['first_dropout'] = args.first_dropout
    if args.middle_dropout is not None:
        opt['middle_dropout'] = args.middle_dropout
    if args.last_dropout is not None:
        opt['last_dropout'] = args.last_dropout
    if args.task_lr  is not None:
        opt['train']['backbone_lr'] = args.backbone_lr
    if args.task_lr  is not None:
        opt['train']['task_lr'] = args.task_lr
    if args.policy_lr  is not None:
        opt['train']['policy_lr'] = args.policy_lr
    if args.decay_lr_rate is not None:
        opt['train']['decay_lr_rate'] = args.decay_lr_rate
    if args.decay_lr_freq is not None:
        opt['train']['decay_lr_freq'] = args.decay_lr_freq
    if args.lambda_sparsity is not None:
        opt['train']['lambda_sparsity'] = args.lambda_sparsity
    if args.lambda_sharing  is not None:
        opt['train']['lambda_sharing'] = args.lambda_sharing

    opt['gpu_ids'] = args.gpu_ids
    
    opt['skip_residual'] = args.skip_residual
    if args.skip_residual:
        if  opt['folder_sfx'] is not None:
            opt['folder_sfx'] += "no_resid"
        else:
            opt['folder_sfx'] = "no_resid"

  
    opt['skip_hidden'] = args.skip_hidden
    if args.skip_hidden:
        if  opt['folder_sfx'] is not None:
            opt['folder_sfx'] += "skip_hidden"
        else:
            opt['folder_sfx'] = "skip_hidden"
  
  
  
    opt['cpu'] = args.cpu
    opt['train']['resume'] = args.resume
    
    if args.min_samples_class is not None:
        opt['dataload']['min_samples_class'] = args.min_samples_class

    if  opt['folder_sfx'] is not None:
        opt['exp_name']  += f"_{opt['folder_sfx']}"
        
    opt['exp_folder'] = build_exp_folder_name(opt)

    # if args.seed_idx is not None:
    #     opt["random_seed"] = opt["seed_list"][args.seed_idx]
    # else:    
    #     opt["random_seed"] = opt["seed_list"][0]
    return opt


def build_exp_folder_name(opt):
    num_heads = len(opt['dataload']['y_tasks'])
    folder_name = f"{opt['hidden_sizes'][0]}x{len(opt['hidden_sizes'])}" \
                    f"_{opt['exp_name_pfx']}"\
                    f"_lr{opt['train']['backbone_lr']}"     \
                    f"_do{opt['middle_dropout']}" 
    
                    # f"_plr{opt['train']['policy_lr']}" \
                    # f"_sp{opt['train']['lambda_sparsity']}" \
                    # f"_sh{opt['train']['lambda_sharing']}"  \
                    # f"_dr{opt['train']['decay_lr_rate']:3.2f}" \
                    # f"_df{opt['train']['decay_lr_freq']:04d}"      
    
    if opt['folder_sfx'] is not None:
        folder_name += f"_{opt['folder_sfx']}"
        
    return folder_name 


def create_path(opt):
    # opt['exp_folder'] = build_exp_folder_name(opt)
    # for k, v in opt['paths'].items():
    # folder_path = opt['paths']['log_dir']
    # print(f" Create folder {full_folder_path}")
    print('\n')
    for k, v in opt['paths'].items():
        full_folder_path = os.path.join(v, opt['exp_folder'])
        if not os.path.isdir(full_folder_path):
            print(f" {k:20s} create folder:  {full_folder_path}")
            makedir(full_folder_path)
        else:
            print(f" {k:20s} folder exists:  {full_folder_path}")
        opt['paths'][k] = full_folder_path
    print()


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)




def should(current_freq, freq):
    return current_freq % freq == 0


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_meta_labels(n_way, n_support, n_query):
    train_labels = torch.from_numpy(np.repeat(np.arange(n_way), n_support).astype('int'))
    test_labels = torch.from_numpy(np.repeat(np.arange(n_way), n_query).astype('int'))
    return train_labels, test_labels


def create_meta_batch(batch, n_way, n_support, n_query):
    train_labels, test_labels = create_meta_labels(n_way, n_support, n_query)
    whole_batch = {'videos': batch['videos'], 'labels': torch.cat((train_labels, test_labels), dim=0), 'names': batch['names']}
    train_batch = {'videos': batch['videos'][: n_way * n_support], 'labels': train_labels, 'names': batch['names'][: n_way * n_support]}
    test_batch  = {'videos': batch['videos'][n_way * n_support:], 'labels': test_labels, 'names': batch['names'][n_way * n_support:]}

    return whole_batch, train_batch, test_batch


def shuffle_batch(batch):
    batch_size = len(batch['videos'])
    shuffled_inds = np.random.permutation(batch_size).tolist()
    batch = {'videos': batch['videos'][shuffled_inds], 'labels': batch['labels'][shuffled_inds],
             'names': [batch['names'][idx] for idx in shuffled_inds]}
    return batch


def random_color():
    r_v = np.random.randint(low=0, high=256)
    g_v = np.random.randint(low=0, high=256)
    b_v = np.random.randint(low=0, high=256)
    color_int = r_v * 256 * 256 + g_v * 256 + b_v
    color_hex = hex(color_int)
    return color_hex


def parse_config():
    print_separator('READ YAML')
    opt, gpu_ids = read_yaml()
    fix_random_seed(opt["random_seed"])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    print('-----------------------------------------------------')
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], 'opt.txt'), 'w+') as f:
        f.writelines(lines)
    return opt, gpu_ids


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def get_iou(pred, gt, n_classes=21):
    total_miou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        miou = (sum(iou) / len(iou))
        total_miou += miou

    total_miou = total_miou // len(pred)
    return total_miou


def in_ipynb():
    from IPython import get_ipython
    try:
        cfg = get_ipython().config 
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter



def init_records(tasks, num_seg_cls):
    records = {}
    if 'seg' in tasks:
        assert (num_seg_cls != -1)
    records['seg'] = {'mIoUs'    : [], 
                      'pixelAccs': [],  
                      'errs'     : [], 
                      'conf_mat' : np.zeros((num_seg_cls, num_seg_cls)),
                      'labels'   : np.arange(num_seg_cls), 
                      'gts': [], 
                      'preds': []}
    if 'sn' in tasks:
        records['sn'] = {'cos_similaritys': []}
    if 'depth' in tasks:
        records['depth'] = {'abs_errs': [], 
                            'rel_errs': [], 
                            'sq_rel_errs': [], 
                            'ratios'  : [], 
                            'rms'     : [], 
                            'rms_log' : []}
    if 'keypoint' in tasks:
        records['keypoint'] = {'errs': []}
    if 'edge' in tasks:
        records['edge'] = {'errs': []}
    return records


def populate_records(records, metrics, tasks):
    from sklearn.metrics import confusion_matrix
    if 'seg' in tasks:
        new_mat = confusion_matrix(metrics['seg']['gt'], metrics['seg']['pred'], records['seg']['labels'])
        assert (records['seg']['conf_mat'].shape == new_mat.shape)
        records['seg']['conf_mat'] += new_mat
        records['seg']['pixelAccs'].append(metrics['seg']['pixelAcc'])
        records['seg']['errs'].append(metrics['seg']['err'])
        records['seg']['gts'].append(metrics['seg']['gt'])
        records['seg']['preds'].append(metrics['seg']['pred'])

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

    return records


def populate_val_metrics(records, tasks, num_seg_cls, batch_size):
    val_metrics = {}
    
    if 'seg' in tasks:
        val_metrics['seg'] = {}
        jaccard_perclass = []
        for i in range(num_seg_cls):
            if not records['seg']['conf_mat'][i, i] == 0:
                jaccard_perclass.append(records['seg']['conf_mat'][i, i] / (np.sum(records['seg']['conf_mat'][i, :]) +
                                                                            np.sum(records['seg']['conf_mat'][:, i]) -
                                                                            records['seg']['conf_mat'][i, i]))

        val_metrics['seg']['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)

        val_metrics['seg']['Pixel Acc'] = (np.array(records['seg']['pixelAccs']) * np.array(batch_size)).sum() / sum(
            batch_size)

        val_metrics['seg']['err'] = (np.array(records['seg']['errs']) * np.array(batch_size)).sum() / sum(batch_size)

        
    if 'sn' in tasks:
        val_metrics['sn'] = {}
        overall_cos = np.clip(np.concatenate(records['sn']['cos_similaritys']), -1, 1)

        angles = np.arccos(overall_cos) / np.pi * 180.0
        val_metrics['sn']['cosine_similarity'] = overall_cos.mean()
        val_metrics['sn']['Angle Mean'] = np.mean(angles)
        val_metrics['sn']['Angle Median'] = np.median(angles)
        val_metrics['sn']['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
        val_metrics['sn']['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
        val_metrics['sn']['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
        val_metrics['sn']['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100
        
    if 'depth' in tasks:
        val_metrics['depth'] = {}
        records['depth']['abs_errs'] = np.stack(records['depth']['abs_errs'], axis=0)
        records['depth']['rel_errs'] = np.stack(records['depth']['rel_errs'], axis=0)
        records['depth']['ratios'] = np.concatenate(records['depth']['ratios'], axis=0)

        val_metrics['depth']['abs_err'] = (records['depth']['abs_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['rel_err'] = (records['depth']['rel_errs'] * np.array(batch_size)).sum() / sum(batch_size)
       
        val_metrics['depth']['sigma_1.25'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25)) * 100
        val_metrics['depth']['sigma_1.25^2'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 2)) * 100
        val_metrics['depth']['sigma_1.25^3'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 3)) * 100

        
    if 'keypoint' in tasks:
        val_metrics['keypoint'] = {}
        val_metrics['keypoint']['err'] = (np.array(records['keypoint']['errs']) * np.array(batch_size)).sum() / sum(batch_size)

    if 'edge' in tasks:
        val_metrics['edge'] = {}
        val_metrics['edge']['err'] = (np.array(records['edge']['errs']) * np.array(batch_size)).sum() / sum(batch_size)
 
    return val_metrics


def get_reference_metrics(opt):
    if opt['dataload']['dataset'] == 'NYU_v2':
        if len(opt['tasks_num_class']) == 2:
            refer_metrics = {'seg': {'mIoU': 0.413, 'Pixel Acc': 0.691},
                             'sn': {'Angle Mean': 15, 'Angle Median': 11.5, 'Angle 11.25': 49.2, 'Angle 22.5': 76.7,
                                    'Angle 30': 86.8}}
        elif len(opt['tasks_num_class']) == 3:
            refer_metrics = {'seg': {'mIoU': 0.275, 'Pixel Acc': 0.589},
                             'sn': {'Angle Mean': 17.5, 'Angle Median': 14.2, 'Angle 11.25': 34.9, 'Angle 22.5': 73.3,
                                    'Angle 30': 85.7},
                             'depth': {'abs_err': 0.62, 'rel_err': 0.25, 'sigma_1.25': 57.9,
                                       'sigma_1.25^2': 85.8, 'sigma_1.25^3': 95.7}}
        else:
            raise ValueError('num_class = %d is invalid' % len(opt['tasks_num_class']))

    elif opt['dataload']['dataset'] == 'CityScapes':
        num_seg_class = opt['tasks_num_class'][opt['tasks'].index('seg')] if 'seg' in opt['tasks'] else -1

        if num_seg_class == 7 and opt['dataload']['small_res']:
            refer_metrics = {'seg': {'mIoU': 0.519, 'Pixel Acc': 0.722},
                         'depth': {'abs_err': 0.017, 'rel_err': 0.33, 'sigma_1.25': 70.3,
                                   'sigma_1.25^2': 86.3, 'sigma_1.25^3': 93.3}}
        elif num_seg_class == 7 and not opt['dataload']['small_res']:
            refer_metrics = {'seg': {'mIoU': 0.644, 'Pixel Acc': 0.778},
                         'depth': {'abs_err': 0.017, 'rel_err': 0.33, 'sigma_1.25': 70.3,
                                   'sigma_1.25^2': 86.3, 'sigma_1.25^3': 93.3}}
        
        elif num_seg_class == 19 and not opt['dataload']['small_res']:
            refer_metrics = {'seg': {'mIoU': 0.402, 'Pixel Acc': 0.747},
                            'depth': {'abs_err': 0.017, 'rel_err': 0.33, 'sigma_1.25': 70.3,
                                    'sigma_1.25^2': 86.3, 'sigma_1.25^3': 93.3}}
        else:
            raise ValueError('num_seg_class = %d and small res = %d are not supported' % (num_seg_class, opt['dataload']['small_res']))
 
    elif opt['dataload']['dataset'] == 'Taskonomy':
        refer_metrics = {'seg': {'err': 0.517},
                         'sn': {'cosine_similarity': 0.716},
                         'depth': {'abs_err': 0.021},
                         'keypoint': {'err': 0.197},
                         'edge': {'err': 0.212}}
    
    else:
        raise NotImplementedError('Dataset %s is not implemented' % opt['dataload']['dataset'])
    
    return refer_metrics

def check_for_best_metrics(best_value, current_iter, refer_metrics, val_metrics, opt):
    new_value = 0
    for k in refer_metrics.keys():
        if k in val_metrics.keys():
            for kk in val_metrics[k].keys():
                if not kk in refer_metrics[k].keys():
                    continue
                if (k == 'sn' and kk in ['Angle Mean', 'Angle Median']) or (
                        k == 'depth' and not kk.startswith('sigma')) or (kk == 'err'):
                    value = refer_metrics[k][kk] / val_metrics[k][kk]
                else:
                    value = val_metrics[k][kk] / refer_metrics[k][kk]

                value = value / len(list(set(val_metrics[k].keys()) & set(refer_metrics[k].keys())))
                new_value += value

    if new_value > best_value:
        best_value = new_value
        best_metrics = val_metrics
        best_iter = current_iter
        # environ.save_checkpoint('retrain%03d_policyIter%s_best' % (exp_id, opt['train']['policy_iter']), current_iter)
        
        print('new value: %.3f' % new_value)
        print('best iter: %d, best value: %.3f' % (best_iter, best_value), best_metrics)    
    return best_value, best_iter, best_metrics


