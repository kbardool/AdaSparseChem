import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from torch.utitensorboardX import SummaryWriter
from utils.util import print_current_errors, print_heading, print_dbg, print_underline
# from data_utils.image_decoder import inv_preprocess, decode_labels2


class BaseEnv():
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_name, tasks_num_class, device=0, is_train=True, opt=None, verbose = None):
        """
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        print_heading(f"{self.name}.super() init()  Start - verbose: {verbose}", verbose = True)
        self.verbose = False if verbose is None else verbose
        
        # self.verbose = verbose if verbose is not None else False
        self.checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)
        self.is_train = is_train
        self.tasks_num_class = tasks_num_class
        self.device_id = device
        self.opt = opt
        self.device  = 'cpu' if self.opt['cpu'] else 'gpu'
        self.dataset = self.opt['dataload']['dataset']
        self.tasks = self.opt['tasks']

        if torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % device)
    
        print_underline(f"Input parms :", verbose = self.verbose)
        print(  ' log_dir        : ', self.log_dir, 
                '\n checkpoint_dir : ', self.checkpoint_dir, 
                '\n exp_name       : ', exp_name, 
                '\n tasks_num_class: ', self.tasks_num_class,
                '\n device         : ', self.device,
                '\n device id      : ', self.device_id,
                '\n dataset        : ', self.dataset, 
                '\n tasks          : ', self.tasks, 
                '\n')

        self.networks   = {}
        self.losses     = {}
        self.optimizers = {}
        self.schedulers = {}
        self.metrics    = {}

        self.define_networks(tasks_num_class, verbose = verbose)
        self.define_loss()

        
        if is_train:
            # define optimizer
            self.define_optimizer()
            self.define_scheduler()
            # define summary writer
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.print_configuration()
        print_heading(f"{self.name}.super() init()  end", verbose = True)

        return

    def print_configuration(self):
        if self.verbose :
            print()
            print_heading(f" {self.name} - Final Configuration ", verbose = True)
            print_heading( f"networks       :", verbose = True)
            print_dbg( f" {self.networks  }", verbose = True)
            print_heading( f"losses         :", verbose = True)
            print_dbg( f" {self.losses    }", verbose = True)
            print_heading( f"optimizers     :", verbose = True)
            print_dbg( f" {self.optimizers}", verbose = True)
            print_heading( f"schedulers     :", verbose = True)
            print_dbg( f" {self.schedulers}", verbose = True)

    # ##################### define networks / optimizers / losses ####################################

    def define_loss(self):
        pass

    def define_networks(self, tasks_num_class):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    def set_inputs(self, batch):
        pass

    def extract_features(self):
        pass

    def get_loss_dict(self, verbose = False):
        print_dbg(f"get loss dict from self.losses", verbose)
        loss = {}
        for key in self.losses.keys():
            loss[key] = {}
            for subkey, v in self.losses[key].items():
                print_dbg(f"  key:  {key}   subkey: {subkey}  value: {v:.4f}", verbose)
                loss[key][subkey] = v.data
        return loss

    def print_loss(self, current_iter, start_time, metrics=None, title='Iteration', verbose = False):
        if metrics is None:
            loss = self.get_loss_dict()
        else:
            # loss = {'metrics': metrics}
            loss = metrics

        loss_display = f"{title}  {current_iter} -  Total Loss: {loss['total']['total']:.4f}     Task Loss: {loss['tasks']['total']:.4f}  " 
        if 'sparsity' in loss:
            loss_display += f"Policy Losses:  Sparsity: {loss['sparsity']['total']:.4f}      Sharing: {loss['hamming']['total']:.5e} "

        print_dbg(loss_display, verbose = verbose)
                      
        for key in loss.keys():
            # print(key + ':')
            if isinstance(loss[key], dict):
                for subkey in loss[key].keys():
                    self.writer.add_scalar('%s/%s'%(key, subkey), loss[key][subkey], current_iter)
                    print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, loss[key], time.time() - start_time)
            elif (isinstance(loss[key], float)):
                    self.writer.add_scalar('%s'%(key), loss[key], current_iter)
                    print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, loss[key], time.time() - start_time)



    def print_metrics(self, current_iter, start_time, metrics=None, title='Iteration', verbose = False):
        if metrics is None:
            loss = self.get_loss_dict()
        
        for t_id, task in enumerate(self.tasks):
            task_key = f"task{t_id+1}"
            print_heading(f"{title}  {current_iter}  {task_key} : {metrics[task_key]['classification_agg']}", verbose = verbose)

            for key, item  in metrics[task_key]['classification_agg'].items():
                self.writer.add_scalar('%s/%s'%(task_key, key), metrics[task_key]['classification_agg'][key], current_iter)
                    # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, loss[key], time.time() - start_time)


    def get_current_state(self, current_iter):
        # ##################### change the state of each module ####################################
        current_state = {}
        for k, v in self.networks.items():
            if isinstance(v, nn.DataParallel):
                current_state[k] = v.module.state_dict()
            else:
                current_state[k] = v.state_dict()
        for k, v in self.optimizers.items():
            current_state[k] = v.state_dict()
        current_state['iter'] = current_iter
        return current_state

    def save_checkpoint(self, label, current_iter, verbose = False):
        """
        Save the current checkpoint
        :param label: str, the label for the loading checkpoint
        :param current_iter: int, the current iteration
        """
        current_state = self.get_current_state(current_iter)
        save_filename = '%s_model.pth.tar' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(current_state, save_path)
        print_heading(f" Saved checkpoint to {save_path}", verbose = verbose)

    def load_snapshot(self, snapshot):
        """
        load snapshot
        """
        for k, v in self.networks.items():
            if k in snapshot.keys():
                # loading values for the existed keys
                model_dict = v.state_dict()
                pretrained_dict = {}
                for kk, vv in snapshot[k].items():
                    if kk in model_dict.keys() and model_dict[kk].shape == vv.shape:
                        pretrained_dict[kk] = vv
                    else:
                        print('skipping %s' % kk)
                model_dict.update(pretrained_dict)
                self.networks[k].load_state_dict(model_dict)
                # self.networks[k].load_state_dict(snapshot[k])
        if self.is_train:
            for k, v in self.optimizers.items():
                if k in snapshot.keys():
                    self.optimizers[k].load_state_dict(snapshot[k])
        return snapshot['iter']

    def load_checkpoint(self, label, path=None):
        """
        load the checkpoint
        :param label: str, the label for the loading checkpoint
        :param path: str, specify if knowing the checkpoint path
        """
        save_filename = '%s_model.pth.tar' % label
        if path is None:
            save_path = os.path.join(self.checkpoint_dir, save_filename)
        else:
            save_path = os.path.join(path,save_filename)
            # save_path = path

        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            if self.device == 'gpu':
                print(f'Loading to GPU {self.device_id}')
                snapshot = torch.load(save_path, map_location='cuda:%d' % self.device_id)
            else:
                print(f'Loading to CPU')
                snapshot = torch.load(save_path, map_location='cpu')
            return self.load_snapshot(snapshot)
        else:
            raise ValueError('snapshot %s does not exist' % save_path)

    def visualize(self):
        pass
        # ##################### visualize #######################
        #     # TODO: implement the visualization of depth
        #     save_results = {}
        #     if 'seg' in self.tasks:
        #         num_seg_class = self.tasks_num_class[self.tasks.index('seg')]
        #         self.save_seg = decode_labels2(torch.argmax(self.seg_output, dim=1).unsqueeze(dim=1), num_seg_class, 'seg', self.seg)
        #         self.save_gt_seg = decode_labels2(self.seg, num_seg_class, 'seg', self.seg)
        #         save_results['save_seg'] = self.save_seg
        #         save_results['save_gt_seg'] = self.save_gt_seg
        #     if 'sn' in self.tasks:
        #         self.save_normal = decode_labels2(F.normalize(self.sn_output) * 255, None, 'normal', F.normalize(self.normal.float()) * 255)
        #         self.save_gt_normal = decode_labels2(F.normalize(self.normal.float()) * 255, None, 'normal', F.normalize(self.normal.float()) * 255,)
        #         save_results['save_sn'] = self.save_normal
        #         save_results['save_gt_sn'] = self.save_gt_normal
        #     if 'depth' in self.tasks:
        #         self.save_depth = decode_labels2(self.depth_output, None, 'depth', self.depth.float())
        #         self.save_gt_depth = decode_labels2(self.depth.float(), None, 'depth', self.depth.float())
        #         save_results['save_depth'] = self.save_depth
        #         save_results['save_gt_depth'] = self.save_gt_depth
        #     self.save_img = inv_preprocess(self.img)
        #     save_results['save_img'] = self.save_img
        #     return save_results
        # #######################################################

    def train(self):
        # ##################### change the state of each module ####################################
        """
        Change to the training mode
        """
        for k, v in self.networks.items():
            v.train()

    def eval(self):
        """
        Change to the eval mode
        """
        for k, v in self.networks.items():
            v.eval()

    def cuda(self, gpu_ids):
        if len(gpu_ids) == 1:
            for k, v in self.networks.items():
                v.to(self.device)
        else:
            for k, v in self.networks.items():
                self.networks[k] = nn.DataParallel(v, device_ids=gpu_ids)
                self.networks[k].to(self.device)

    def cpu(self):
        print(f'base_env.cpu()')
        for k, v in self.networks.items():
            print(f' Network item {k} moved to cpu')
            v.cpu()

    def name(self):
        return 'BaseEnv'