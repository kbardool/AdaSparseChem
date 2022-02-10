import os
import argparse
import yaml
import random

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
    left_width = (total_len - len(text))//2
    right_width = total_len - len(text) - left_width
    print("#" * left_width + text + "#" * right_width)
    print('#' * total_len)

def print_dbg(text, verbose = False):
    if verbose:
        print(text)

# @debug_off
def print_heading(text,  verbose = False, force = False):
    len_ttl = max(len(text)+4, 50)
    len_ttl = min(len_ttl, 120)
    if verbose or force:
        print('-' * len_ttl)
        print(f"{text}")
        # left_width = (total_len - len(text))//2
        # right_width = total_len - len(text) - left_width
        # print("#" * left_width + text + "#" * right_width)
        print('-' * len_ttl, '\n')


def print_underline(text,  verbose = False):
    len_ttl = len(text)+2
    if verbose:
        print(f"\n {text}")
        print('-' * len_ttl)


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


def print_current_errors(log_name, update, log_key, errors, t):
    message = 'update: %d, timestamp: %s wall clock time: %.3f  %s errors:' % (update, timestring(), t, log_key)
    for k, v in errors.items():
        if k.startswith('Update'):
            message += '%s: %s ' % (k, str(v))
        else:
            message += '%s: %.3f ' % (k, v)

    # print(message)
    with open(log_name, 'a') as log_file:
        log_file.write('%s \n' % message)


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


def write_parms_report(opt, output = None, mode = 'w+'):
    with open(os.path.join(opt['paths']['log_dir'], opt['paths']['exp_folder'], 'run_parms.txt'), mode= mode) as f:
        if output is None:
            output = print_yaml2(opt)
            for line in output:
                f.writelines(line+"\n")
        else:
                f.writelines(output+"\n")

def create_path(opt):
    # for k, v in opt['paths'].items():
    folder_path = opt['paths']['log_dir']

    full_folder_path = os.path.join(folder_path, opt['paths']['exp_folder'])
    print(f" Create folder {full_folder_path}")
    makedir(full_folder_path)


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


##
##  Command line and YAML configuration files
##
def get_command_line_args(input = None):
    """ get and parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config"       , required=True, help="Path for the config file")
    parser.add_argument("--exp_instance" , type=str, help="experiment instance, used as folder prefix, defaults to mmdd_hhmm")
    parser.add_argument("--exp_ids"      , type=int, nargs='+', default=[0], help="Seed Override - default is 0")
    parser.add_argument("--batch_size"   , type=int, help="Backbone Learning Rate Override - default read from config file")
    parser.add_argument("--backbone_lr"  , type=float, help="Backbone Learning Rate Override - default read from config file")
    parser.add_argument("--task_lr"      , type=float, help="Task Heads Learning Rate Override - default read from config file")
    parser.add_argument("--decay_lr_rate", type=float, help="LR Decay Rate Override - default read from config file")
    parser.add_argument("--decay_lr_freq", type=float, help="LR Decay Frequency Override - default read from config file")
    parser.add_argument("--gpus"         , type=int, nargs='+', default=[0], help="GPU Device Ids")
    parser.add_argument("--cpu"          , default=False, action="store_true",  help="CPU instead of GPU")
    if input is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input)
    print(' command line parms : ' , vars(args))
    return args


def read_yaml():
    # get command line arguments
    args = get_command_line_args()
    
    # torch.cuda.set_device(args.gpu)
    
    with open(args.config) as f:
        opt = yaml.load(f)
    opt['exp_instance'] = args.exp_instance
    opt['cpu'] = args.cpu
    opt['paths']['exp_folder'] = build_exp_folder_name(opt)
    return opt, args.gpus, args.exp_ids


def read_yaml_from_input(args = None, exp_instance = None):
    """ read yaml passing command line arguments """
        
    # torch.cuda.set_device(args.gpu)
    with open(args.config) as f:
        opt = yaml.safe_load(f)


    opt['cpu'] = args.cpu
    
    if exp_instance is not None:
        opt['exp_instance'] = exp_instance
    elif args.exp_instance is not None:
        opt['exp_instance'] = args.exp_instance

    opt['paths']['exp_folder'] = build_exp_folder_name(opt)
    return opt, args.gpus, args.exp_ids

def build_exp_folder_name(opt):
    folder_name = f"{opt['hidden_sizes'][0]}x{len(opt['hidden_sizes'])}" \
                  f"_{opt['exp_instance']}"\
                  f"_plr{opt['train']['policy_lr']}" \
                  f"_sp{opt['train']['lambda_sparsity']}" \
                  f"_sh{opt['train']['lambda_sharing']}"   
    #   f"_bs{opt['train']['batch_size']:03d}" \
    #   f"_lr{opt['train']['backbone_lr']}"   \
    #   f"_dr{opt['train']['decay_lr_rate']:3.2f}" \
    #   f"_df{opt['train']['decay_lr_freq']:04d}"      
    return folder_name 


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
    fix_random_seed(opt["seed"])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    print('-----------------------------------------------------')
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], opt['paths']['exp_folder'], 'opt.txt'), 'w+') as f:
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
