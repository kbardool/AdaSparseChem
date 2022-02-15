import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from torch.utitensorboardX import SummaryWriter
from utils.util import write_loss_txt, print_heading, print_dbg, print_underline, write_parms_report, write_loss_csv
# from data_utils.image_decoder import inv_preprocess, decode_labels2


class BaseEnv():
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_instance, tasks_num_class, device=0, is_train=True, opt=None, verbose = None):
        """
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        print_heading(f"{self.name}.super() init()  Start - verbose: {verbose}", verbose = True)
        self.verbose = False if verbose is None else verbose
        
        # self.verbose = verbose if verbose is not None else False
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.is_train = is_train
        self.tasks_num_class = tasks_num_class
        self.device_id = device
        self.opt = opt
        self.device  = 'cpu' if self.opt['cpu'] else 'gpu'
        self.dataset = self.opt['dataload']['dataset']
        self.tasks = self.opt['tasks']
        self.loss_csv_file = os.path.join(self.log_dir, 'loss.csv')
        self.log_file  = open(os.path.join(self.log_dir, 'log.txt'), 'a')

        if torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % device)
    
        print_underline(f"Input parms :", verbose = self.verbose)
        print(    ' log_dir        : ', self.log_dir, 
                '\n checkpoint_dir : ', self.checkpoint_dir, 
                '\n exp_name       : ', exp_instance, 
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

        self.write_run_info()

        print_heading(f"{self.name}.super() init()  end", verbose = True)

        return


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

    def display_parameters(self):

        task_specific_params = self.get_task_specific_parameters()
        arch_parameters      = self.get_arch_parameters()
        backbone_parameters  = self.get_backbone_parameters()
        
        print('-----------------------')
        print(' task specific parms  :')
        print('-----------------------')
        for i,j  in enumerate(task_specific_params):
            print(i, type(j), j.shape)
        print('\n')
        print('-----------------------')
        print('\n arch_parameters    :')
        print('-----------------------')
        for i,j in enumerate(arch_parameters):
            print(i, type(j), j.shape)
        print('\n')
        print('-----------------------')
        print('\n backbone parameters:')
        print('-----------------------')
        for i,j in enumerate(backbone_parameters):
            print(i, type(j), j.shape)
        
        return

    def print_configuration(self, verbose = False):
        config = f" \n " \
                 f"---------------------------------------- \n" \
                 f" {self.name} - Network Configuration       \n" \
                 f"---------------------------------------- \n" \
                 f"\n"                     \
                 f"----------------\n"     \
                 f"networks       :\n"     \
                 f"----------------\n"     \
                 f" {self.networks}\n\n"   \
                 f"----------------\n"     \
                 f"optimizers     :\n"     \
                 f"----------------\n"     \
                 f" {self.optimizers}\n\n" \
                 f"----------------\n"     \
                 f"schedulers     :\n"     \
                 f"----------------\n"    

        for name, sch in self.schedulers.items():
            for key,val in sch.__dict__.items(): 
                config +=f"{key:30s}: {val} \n"
                #  f"----------------\n"     \
                #  f"losses         :\n"     \
                #  f"----------------\n"     \
                #  f" {self.losses}  \n\n"   \
        return config

    def get_loss_dict(self, verbose = False):
        print_dbg(f"get loss dict from self.losses", verbose)
        loss = {}
        for key in self.losses.keys():
            loss[key] = {}
            for subkey, v in self.losses[key].items():
                print_dbg(f"  key:  {key}   subkey: {subkey}  value: {v:.4f}", verbose)
                if isinstance(v, torch.Tensor):
                    loss[key][subkey] = v.data
                else:
                    loss[key][subkey] = v
        return loss


    def write_run_info(self):
        split_rto = self.opt['dataload']    ['x_split_ratios']
        md = f"""
### Run Information

    Experiment Group:    {self.opt['exp_name']}  
    Run id          :    {self.opt['exp_instance']}

### Description:


** {self.opt['exp_description']} **


    Batch Size          : {self.opt['train']['batch_size']}             
    Data split ratios   :   Warmup: {split_rto[0]}    Weight:{split_rto[1]}    Policy: {split_rto[2]} \t Validation:{split_rto[3]}
    Hidden layers       :   {len(self.opt['hidden_sizes'])} - {self.opt['hidden_sizes']} 
    starting task_lr    :   {self.opt['train']['task_lr']} 
    starting backbone_lr:   {self.opt['train']['backbone_lr']} 
    starting policy_lr  :   {self.opt['train']['policy_lr']} 
    LR Decay freq       :   {self.opt['train']['decay_lr_freq']} |
    LR Decay rate       :   {self.opt['train']['decay_lr_rate']} |

**Hyperparameters**

| param | values |
| ----- | ---------- |
| batch_size | {self.opt['train']['batch_size']} |
| # of hidden layers | {len(self.opt['hidden_sizes'])} |
| layer sizes | {self.opt['hidden_sizes']} |
| task_lr:    | {self.opt['train']['task_lr']} |
| backbone_lr:| {self.opt['train']['backbone_lr']} |
| policy_lr:  | {self.opt['train']['policy_lr']} |
| LR Decay freq | {self.opt['train']['decay_lr_freq']} |
| LR Decay rate | {self.opt['train']['decay_lr_rate']} |

"""
        self.writer.add_text('_General Info_', md, 0)

    def display_trained_policy(self, epoch=0, out = None):
        if not isinstance(out, list):
            out = [out]

        policy_softmaxs = self.get_policy_prob()
        policy_argmaxs = 1-np.argmax(policy_softmaxs, axis = -1)
        ln = "\n"
        ln += f" {epoch:3d} epochs  softmax        sel        softmax       sel        softmax       sel \n"
        ln += f" -----    ---------------   ---     ---------------  ---     ---------------  --- \n"
        for idx, (l1,l2,l3,  p1,p2,p3) in enumerate(zip(policy_softmaxs[0], policy_softmaxs[1], policy_softmaxs[2], 
                                                        policy_argmaxs[0], policy_argmaxs[1], policy_argmaxs[2]),1):
            ln += f"   {idx}      {l1[0]:.4f}   {l1[1]:.4f}   {p1:2d}   {l2[0]:9.4f}   {l2[1]:.4f}  {p2:2d}   {l3[0]:9.4f}   {l3[1]:.4f}  {p3:2d}\n"
        ln += '\n'
        for file in out:
            print(ln, file = file)

    # def display_trained_policy(self, epoch=0, file = None):

    #     policy_softmaxs = self.get_policy_prob()
    #     policy_argmaxs = 1-np.argmax(policy_softmaxs, axis = -1)
    #     print()
    #     # print(f" Trained polcies at epoch: {epoch} ")
    #     # print(f"                    task 1                          task 2                         task 3        ")
    #     print(f" Layer       softmax        sel        softmax       sel        softmax       sel ")
    #     print(f" -----    ---------------   ---     ---------------  ---     ---------------  --- ")
    #     for idx, (l1,l2,l3,  p1,p2,p3) in enumerate(zip(policy_softmaxs[0], policy_softmaxs[1], policy_softmaxs[2], 
    #                                                     policy_argmaxs[0], policy_argmaxs[1], policy_argmaxs[2]),1):
    #         print(f"   {idx}      {l1[0]:.4f}   {l1[1]:.4f}   {p1:2d}   {l2[0]:9.4f}   {l2[1]:.4f}  {p2:2d}   {l3[0]:9.4f}   {l3[1]:.4f}  {p3:2d}")
    #     print()


    def display_loss(self, current_iter,  title='Iteration'):
        loss_display = f"{title}  {current_iter} -  Total Loss: {self.losses['total']['total']:.4f}     \nTask: {self.losses['losses']['total']:.4f}" 
        if 'sparsity' in self.losses:
            loss_display += f"   Sparsity: {self.losses['sparsity']['total']:.5e}    Sharing: {self.losses['sharing']['total']:.5e} "
        # loss_display = f"{title}  {current_iter} \nLosses: Task: {self.losses['losses']['total']:.4f}   " 
        # if 'sparsity' in self.losses:
        #     loss_display += f"Spar: {self.losses['sparsity']['total']:.5e}   Shr: {self.losses['sharing']['total']:.5e}"
        # loss_display += f"   Ttl: {self.losses['total']['total']:.4f}"

        print(loss_display)


                      
    def print_loss(self, current_iter, start_time, loss=None, title='Iteration', 
                    to_tb      = True,  
                    to_csv     = True, 
                    to_display = False, 
                    to_text    = False):
        elapsed_time = time.time() - start_time

        if to_display:
            self.display_loss(current_iter, title)
        
        if to_csv:
            write_loss_csv(self.loss_csv_file, current_iter, elapsed_time, self.losses )
        
        if to_text:
            write_loss_txt(os.path.join(self.log_dir, 'loss.txt'), current_iter, elapsed_time, self.losses)

        if to_tb:
            for key in ['parms', 'losses', 'losses_mean', 'total', 'sharing', 'sparsity']:
                if key not in self.losses:
                    continue
                # print(key + ':')
                if isinstance(self.losses[key], dict):
                    for subkey in self.losses[key].keys():
                        self.writer.add_scalar('trn_%s/%s'%(key, subkey), self.losses[key][subkey], current_iter)
                elif (isinstance(self.losses[key], float)):
                    self.writer.add_scalar('trn_%s'%(key), self.losses[key], current_iter)


    def print_metrics(self, current_iter, start_time, metrics=None, title='Iteration', verbose = False):
        """ write metrics to tensorboard and optionally to sysout """
        if metrics is None:
            metrics = self.val_metrics
        
        ## Write validation losses
        # if 'loss' in metrics:
        #     for key,item in metrics['loss'].items():
        #         self.writer.add_scalar(f"val_losses/{key}_loss", item, current_iter)

        # if 'loss_mean' in metrics:
        #     for key,item in metrics['loss_mean'].items():
        #         self.writer.add_scalar(f"val_losses/{key}_loss_mean", item, current_iter)

        # for key in loss.keys():
        for key in ['loss', 'loss_mean', 'sharing', 'sparsity']:
            # if key not in metrics:
                # continue
            # print(key + ':')
            if isinstance(metrics[key], dict):
                for subkey, metric_value in metrics[key].items():
                    self.writer.add_scalar('val_%s/%s'%(key, subkey), metric_value, current_iter)
                    # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, metrics[key], time.time() - start_time)
            elif (isinstance(metrics[key], float)):
                self.writer.add_scalar('val_%s'%(key), metrics[key], current_iter)
                # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, metrics[key], time.time() - start_time)

        ## Write aggregated metrics for each group (i.e, group of tasks)
        for t_id, _ in enumerate(self.tasks):
            key = f"task{t_id+1}"
            print_heading(f"{title}  {current_iter}  {key} : {metrics[key]['classification_agg']}", verbose = verbose)

            for subkey, metric_value in metrics[key]['classification_agg'].items():
                self.writer.add_scalar(f"val_metrics:{key:s}/{subkey:s}", metric_value, current_iter)
                 # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, loss[key], time.time() - start_time)

        ## Write aggregated metrics (aggregated accross all groups/tasks)
        key = "aggregated"
        print_heading(f"{title}  {current_iter}  {key} : {metrics[key]}", verbose = verbose)

        for subkey, metric_value  in metrics[key].items():
            self.writer.add_scalar(f"val_metrics:{key:s}/{subkey:s}", metric_value, current_iter)
            # print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter,key, loss[key], time.time() - start_time)

 
    def get_current_state(self, current_iter):
        """ change the state of each module  """
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
        print_heading(f" Saved checkpoint to {save_path} iteration: {current_iter}", verbose = verbose)


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
        """
        Move network items to assigned GPU device (self.device)
        """
        if len(gpu_ids) == 1:
            for k, v in self.networks.items():
                v.to(self.device)
        else:
            for k, v in self.networks.items():
                self.networks[k] = nn.DataParallel(v, device_ids=gpu_ids)
                self.networks[k].to(self.device)

    def cpu(self):
        """
        Move network items to  CPU device
        """
        print(f'base_env.cpu()')
        for k, v in self.networks.items():
            print(f' Network item {k} moved to cpu')
            v.cpu()

    def name(self):
        return 'BaseEnv'