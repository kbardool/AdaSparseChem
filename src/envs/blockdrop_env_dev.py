import os
import pickle
from scipy.special import softmax
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
from torch import nn
from ..models.base      import Bottleneck, BasicBlock
from .MTL2_Dev             import MTL2_Dev
from .MTL_Instance_Dev     import MTL_Instance_Dev
from .base_env_dev         import BaseEnv
from utils.util            import timestring, print_heading
import pprint
pp = pprint.PrettyPrinter(indent=4)



class BlockDropEnv_Dev(BaseEnv):
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_name, tasks_num_class, 
                 init_neg_logits=-10, 
                 device=0, 
                 init_temperature=5.0, 
                 temperature_decay=0.965,
                 is_train=True, 
                 opt=None):
        """
        :param num_class: int, the number of classes in the dataset
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        print("* ",self.name(), "Initializtion")
        print(  '\n log_dir        : ', log_dir, 
                '\n checkpoint_dir : ', checkpoint_dir, 
                '\n exp_name       : ', exp_name, 
                '\n tasks_num_class: ', tasks_num_class,
                '\n init_neg_logits: ', init_neg_logits, 
                '\n gpu device     : ', device,
                '\n init temp      : ', init_temperature, 
                '\n decay temp     : ', temperature_decay, 
                '\n is_train       : ', is_train, '\n')
        
        self.init_neg_logits = init_neg_logits
        self.temp            = init_temperature
        self._tem_decay      = temperature_decay
        self.num_tasks       = len(tasks_num_class)
        print("===============================================================================")
        print("===============================================================================")
        super(BlockDropEnv_Dev, self).__init__(log_dir, checkpoint_dir, exp_name, tasks_num_class, device,
                                           is_train, opt)

    # ##################### define networks / optimizers / losses ####################################

    def define_networks(self, tasks_num_class):
        # construct a deeplab resnet 101
        if self.opt['backbone'].startswith('ResNet'):
            init_method = self.opt['train']['init_method']
            if self.opt['backbone'] == 'ResNet101':
                block = Bottleneck
                layers = [3, 4, 23, 3]
            elif self.opt['backbone'] == 'ResNet18':
                block = BasicBlock
                layers = [2, 2, 2, 2]
            elif self.opt['backbone'] == 'ResNet34':
                block = BasicBlock
                layers = [3, 4, 6, 3]
            else:
                raise NotImplementedError('backbone %s is not implemented' % self.opt['backbone'])

            if self.opt['policy_model'] == 'task-specific':
                print(f'Create MTL2 with \n block: {block} \n layers: {layers} \n tasks_num_class: {tasks_num_class} \n init_method: {init_method}')
                self.networks['mtl-net'] = MTL2_Dev(block, layers, tasks_num_class, init_method, self.init_neg_logits, self.opt['skip_layer'])

            elif self.opt['policy_model'] == 'instance-specific':
                print(f'Create MTL_Instance with \n block: {block} \n layers: {layers} \n tasks_num_class: {tasks_num_class} \n init_method: {init_method}')
                self.networks['mtl-net'] = MTL_Instance_Dev(block, layers, tasks_num_class, init_method, self.init_neg_logits, self.opt['skip_layer'])

            else:
                raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])

        else:
            raise NotImplementedError('backbone %s is not implemented' % self.opt['backbone'])


    def define_loss(self):
        """
        ignore_index – Specifies a target value that is ignored and does not contribute to the input gradient. 
                        When size_average is True, the loss is averaged over non-ignored targets.
        """
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss  = nn.L1Loss()
        self.l1_loss2 = nn.L1Loss(reduction='none')

        if self.dataset == 'NYU_v2':
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
            self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
            self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=255)

        elif self.dataset == 'Taskonomy':
        #     dataroot = self.opt['dataload']['dataroot']
        #     weight = torch.from_numpy(np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))).to(self.device).float()
        #     self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        #     self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
            self.cross_entropy_sparisty = nn.CrossEntropyLoss(ignore_index=255)

        elif self.dataset == 'CityScapes':
        #     self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        #     self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise NotImplementedError('Dataset %s is not implemented' % self.dataset)


    def define_optimizer(self, policy_learning=False):
        """"
        if we are in policy learning phase - use SGD
        otherwise, we use ADAM
        """
        task_specific_params = self.get_task_specific_parameters()
        arch_parameters      = self.get_arch_parameters()
        backbone_parameters  = self.get_backbone_parameters()

        
        # TODO: add policy learning to yaml

        print('-----------------------')
        print(' Define Optimizers    :')
        print('-----------------------')

        #----------------------------------------
        # weight optimizers
        #----------------------------------------
        if policy_learning:
            self.optimizers['weights'] = optim.SGD([{'params': task_specific_params, 'lr': self.opt['train']['lr']},
                                                    {'params': backbone_parameters , 'lr': self.opt['train']['backbone_lr']}],
                                                   momentum=0.9, weight_decay=1e-4)
        else:
            self.optimizers['weights'] = optim.Adam([{'params': task_specific_params, 'lr': self.opt['train']['lr']},
                                                     {'params': backbone_parameters , 'lr': self.opt['train']['backbone_lr']}],
                                                    betas=(0.5, 0.999), weight_decay=0.0001)
        
        print(f"\ndefine the weights optimizer - learning mode: {'policy' if policy_learning else 'non-policy'}")
        print(f" optimizers for weights : \n {self.optimizers['weights']}")

        #---------------------------------------
        # optimizers for alpha (logits??)
        #---------------------------------------
        if self.opt['train']['init_method'] == 'all_chosen':
            self.optimizers['alphas'] = optim.Adam(arch_parameters, lr=self.opt['train']['policy_lr'], weight_decay=5*1e-4)
        else:
            self.optimizers['alphas'] = optim.Adam(arch_parameters, lr=0.01, weight_decay=5*1e-4)

        print(f"\ndefine the logits optimizer (init_method: {self.opt['train']['init_method']})")
        print(f" optimizers for alphas : \n {self.optimizers['alphas']}")
            

    def define_scheduler(self, policy_learning=False):
        print('\n-----------------------')
        print(' Define Scheduler     :')
        print('-----------------------')

        print(f"learning mode: {'policy' if policy_learning else 'non-policy'}")
        if policy_learning:
            if 'policy_decay_lr_freq' in self.opt['train'].keys() and 'policy_decay_lr_rate' in self.opt['train'].keys():
                self.schedulers['weights'] = scheduler.StepLR(self.optimizers['weights'],
                                                              step_size=self.opt['train']['policy_decay_lr_freq'],
                                                              gamma=self.opt['train']['policy_decay_lr_rate'])

        else:
            if 'decay_lr_freq' in self.opt['train'].keys() and 'decay_lr_rate' in self.opt['train'].keys():
                self.schedulers['weights'] = scheduler.StepLR(self.optimizers['weights'],
                                                              step_size=self.opt['train']['decay_lr_freq'],
                                                              gamma=self.opt['train']['decay_lr_rate'])
        print(self.schedulers['weights'])


    def get_task_specific_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            task_specific_params = self.networks['mtl-net'].module.task_specific_parameters()
        else:
            task_specific_params = self.networks['mtl-net'].task_specific_parameters()

        return task_specific_params


    def get_arch_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            arch_parameters = self.networks['mtl-net'].module.arch_parameters()
        else:
            arch_parameters = self.networks['mtl-net'].arch_parameters()

        return arch_parameters


    def get_network_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            network_parameters = self.networks['mtl-net'].module.network_parameters()
        else:
            network_parameters = self.networks['mtl-net'].network_parameters()
        return network_parameters


    def get_backbone_parameters(self):
        if self.opt['backbone'] == 'WRN':
            network_parameters = []
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for name, param in self.networks['mtl-net'].module.named_parameters():
                    if name.startswith('task') and ('logits' in name or 'fc' in name):
                        continue
                    else:
                        network_parameters.append(param)
            else:
                for name, param in self.networks['mtl-net'].named_parameters():
                    if name.startswith('task') and ('logits' in name or 'fc' in name):
                        continue
                    else:
                        network_parameters.append(param)
            return network_parameters
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                backbone_parameters = self.networks['mtl-net'].module.backbone_parameters()
            else:
                backbone_parameters = self.networks['mtl-net'].backbone_parameters()
            return backbone_parameters


    ################################ train / test ####################################

    def optimize(self, lambdas, is_policy=False, flag='update_w', num_train_layers=None, hard_sampling=False):
        """
        1) Make forward pass 
        2) compute losses based on tasks 
        3)
        """
        print(f"{timestring} - BlockDrop optimize() pass start: ")
        print(f" lambdas:  {lambdas}     is_policy: {is_policy}    flag: {flag} "
              f" num_train_layers: {num_train_layers}    hard_sampling: {hard_sampling}" )
        
        ## Forward pass 
        self.forward(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)     
        
        ## Get losses for each task in self.tasks 
        print(f"{timestring()} -  BlockDrop get_losses start: ")
        
        if 'seg' in self.tasks:
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            self.get_seg_loss(seg_num_class)

        if 'sn' in self.tasks:
            self.get_sn_loss()

        if 'depth' in self.tasks:
            print(' get_depth_loss    : ', timestring())
            self.get_depth_loss()

        print(f"{timestring()} -  BlockDrop get_losses end ")

        ## Backpropagate 
        if flag == 'update_w':
            print('   ** BlockDrop backward_network start: ', timestring())
            self.backward_network(lambdas)
            print('   ** BlockDrop backward_network end  : ', timestring())

        elif flag == 'update_alpha':
            print('   ** BlockDrop backward_policy start : ', timestring())
            self.backward_policy(lambdas, num_train_layers)
            print('   ** BlockDrop backward_policy end   : ', timestring())
            
        else:
            raise NotImplementedError('Training flag %s is not implemented' % flag)

        print('** BlockDrop optimize() end  : ', timestring())


    def optimize_fix_policy(self, lambdas, num_train_layer=None):
                
        print('** BlockDrop optimize_fix_policy() pass start: ', timestring())
        print('   num_train_layer: ', num_train_layer, 'lambdas: ', lambdas)

        ## Forward pass 
        print('   ** Forward pass (fix_policy) : ', timestring())
        self.forward_fix_policy(num_train_layer)


        ## Get losses for each task in self.tasks 
        print('   ** BlockDrop get_losses (fix_policy) start: ', timestring())
        
        if 'seg' in self.tasks:
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            self.get_seg_loss(seg_num_class)

        if 'sn' in self.tasks:
            self.get_sn_loss()

        ## Backpropagate 
        print('   ** BlockDrop backward_policy (fix_policy) start : ', timestring())
        self.backward_network(lambdas)

        print('** BlockDrop optimize_fix_policy()  end  : ', timestring())


    # def val(self, policy, num_train_layers=None, hard_sampling=False):
        #         metrics = {}
        #         self.forward(is_policy=policy, num_train_layers=num_train_layers, hard_sampling=hard_sampling)
        #         self.resize_results()
        #        
        #         if 'seg' in self.tasks:
        #             metrics['seg'] = {}
        #             seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
        #             pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
        #             metrics['seg']['pred'] = pred
        #             metrics['seg']['gt'] = gt
        #             metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
        #             metrics['seg']['err'] = err
        #            
        #         if 'sn' in self.tasks:
        #             metrics['sn'] = {}
        #             cos_similarity = self.normal_error()
        #             metrics['sn']['cos_similarity'] = cos_similarity
        #         if 'depth' in self.tasks:
        #            
        #             metrics['depth'] = {}
        #             abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.depth_error()
        #             metrics['depth']['abs_err'] = abs_err
        #             metrics['depth']['rel_err'] = rel_err
        #             metrics['depth']['sq_rel_err'] = sq_rel_err
        #             metrics['depth']['ratio'] = ratio
        #             metrics['depth']['rms'] = rms
        #             metrics['depth']['rms_log'] = rms_log
        #            
        #         if 'keypoint' in self.tasks:
        #             metrics['keypoint'] = {}
        #             err = self.keypoint_error()
        #             metrics['keypoint']['err'] = err
        #            
        #         if 'edge' in self.tasks:
        #             metrics['edge'] = {}
        #             err = self.edge_error()
        #             metrics['edge']['err'] = err
        #            
        #         return metrics


    def val2(self, is_policy, num_train_layers=None, hard_sampling=False):
        metrics = {}
        
        if is_policy:
            self.forward_eval(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)
        else:
            self.forward(is_policy = is_policy, num_train_layers = num_train_layers, hard_sampling = hard_sampling)

        self.resize_results()
        
        if 'seg' in self.tasks:
            metrics['seg'] = {}
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
            metrics['seg']['pred'] = pred
            metrics['seg']['gt'] = gt
            metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
            metrics['seg']['err'] = err
                                    
        if 'sn' in self.tasks:
            metrics['sn'] = {}
            cos_similarity = self.normal_error()
            metrics['sn']['cos_similarity'] = cos_similarity
            
        # if 'depth' in self.tasks:
        #     metrics['depth'] = {}
        #     abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.depth_error()
        #     metrics['depth']['abs_err'] = abs_err
        #     metrics['depth']['rel_err'] = rel_err
        #     metrics['depth']['sq_rel_err'] = sq_rel_err
        #     metrics['depth']['ratio'] = ratio
        #     metrics['depth']['rms'] = rms
        #     metrics['depth']['rms_log'] = rms_log
            
        # if 'keypoint' in self.tasks:
        #     metrics['keypoint'] = {}
        #     err = self.keypoint_error()
        #     metrics['keypoint']['err'] = err
            
        # if 'edge' in self.tasks:
        #     metrics['edge'] = {}
        #     err = self.edge_error()
        #     metrics['edge']['err'] = err
            
        return metrics


    def val_fix_policy(self, num_train_layers=None):
        metrics = {}
        self.forward_fix_policy(num_train_layers)
        self.resize_results()
        
        if 'seg' in self.tasks:
            metrics['seg'] = {}
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
            metrics['seg']['pred'] = pred
            metrics['seg']['gt'] = gt
            metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
            metrics['seg']['err'] = err
            
        if 'sn' in self.tasks:
            metrics['sn'] = {}
            cos_similarity = self.normal_error()
            metrics['sn']['cos_similarity'] = cos_similarity
            
        # if 'depth' in self.tasks:
        #     metrics['depth'] = {}
        #     abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.depth_error()
        #     metrics['depth']['abs_err'] = abs_err
        #     metrics['depth']['rel_err'] = rel_err
        #     metrics['depth']['sq_rel_err'] = sq_rel_err
        #     metrics['depth']['ratio'] = ratio
        #     metrics['depth']['rms'] = rms
        #     metrics['depth']['rms_log'] = rms_log
            
        # if 'keypoint' in self.tasks:
        #     metrics['keypoint'] = {}
        #     err = self.keypoint_error()
        #     metrics['keypoint']['err'] = err
            
        # if 'edge' in self.tasks:
        #     metrics['edge'] = {}
        #     err = self.edge_error()
        #     metrics['edge']['err'] = err
        return metrics


    def forward(self, is_policy, num_train_layers = None, hard_sampling = False):
        # print(
        print('   ** BlockDrop forward() num_train_layers:', num_train_layers,' is_policy: ', is_policy, 'hard_smapling: ', hard_sampling, 'input img: ', self.img.shape)
        print('      BlockDrop forward() start: ', timestring())

        outputs, policys, logits = self.networks['mtl-net'](self.img, 
                                                            self.temp, 
                                                            is_policy, 
                                                            num_train_layers=num_train_layers, 
                                                            hard_sampling=hard_sampling, 
                                                            mode='train')
        # import pdb
        # pdb.set_trace()
        
        for t_id,  task in enumerate(self.tasks):
            setattr(self, '%s_pred'  % task     , outputs[t_id])
            setattr(self, 'policy%d' % (t_id+1) , policys[t_id])
            setattr(self, 'logit%d'  % (t_id+1) , logits[t_id])
        
        print('      BlockDrop forward()  end: ', timestring())


    def forward_eval(self, is_policy = True,  num_train_layers = None, hard_sampling = False):
        print('   ** BlockDrop forward_eval() num_train_layers:', num_train_layers,' is_policy: ', is_policy, 'hard_smapling: ', hard_sampling)
        print('      BlockDrop forward_eval()  start: ', timestring())

        outputs,policys, logits = self.networks['mtl-net'](self.img, 
                                                           self.temp, 
                                                           is_policy, 
                                                           num_train_layers=num_train_layers, 
                                                           hard_sampling=hard_sampling,  
                                                           mode='eval')  

        for t_id, task in enumerate(self.tasks):
            setattr(self, '%s_pred' % task, outputs[t_id])
            setattr(self, 'policy%d' % (t_id + 1), policys[t_id])
            setattr(self, 'logit%d' % (t_id + 1), logits[t_id])

        print('      BlockDrop forward_eval() end: ', timestring())


    def forward_fix_policy(self, is_policy = True, num_train_layers = None, hard_sampling = False):
        print_heading(f"{timestring()} - BlockDrop forward_fix_policy() start")
        print(f"\t   num_train_layers:  {num_train_layers}   is_policy: {is_policy}  hard_smapling:  {hard_sampling}")

        ## if Task-Specific we call MTL2.forward()
        if self.opt['policy_model'] == 'task-specific':
            outputs, _, _ = self.networks['mtl-net'](self.img, 
                                                     self.temp, 
                                                     is_policy,  
                                                     num_train_layers=num_train_layers, 
                                                     hard_sampling = hard_sampling , 
                                                     mode='fix_policy')

        ## if Instance-specific we call MTL2_Instance.forward(), which also takes the policy as an input parameter
        # elif self.opt['policy_model'] == 'instance-specific':
        #     policys = []
        #     for task in self.opt['tasks']:
        #         policys.append(getattr(self, '%s_policy' % task))
        #         print(' policy : {policys[-1]}')
        #
        #     outputs, _, _ = self.networks['mtl-net'](self.img, 
        #                                              self.temp, 
        #                                              is_policy = True, 
        #                                              policys=policys, 
        #                                              num_train_layers=num_train_layers, 
        #                                              mode='fix_policy')

        else:
            raise ValueError('policy model = %s is not supported' % self.opt['policy_model'])

        for t_id, task in enumerate(self.tasks):
            setattr(self, '%s_pred' % task, outputs[t_id])


    def get_sparsity_loss2(self, num_train_layers):
        print(f"    get_sparsity_loss    num_train_layers: {num_train_layers}  ")
        self.losses['sparsity'] = {}
        self.losses['sparsity']['total'] = 0
        num_policy_layers = None

        if self.opt['policy_model'] == 'task-specific':
            for t_id in range(self.num_tasks):
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_id + 1))
                else:
                    logits = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_id + 1))
                if num_policy_layers is None:
                    num_policy_layers = logits.shape[0]
                else:
                    assert (num_policy_layers == logits.shape[0])


                if num_train_layers is None:
                    num_train_layers = num_policy_layers

                print(f"    get_sparsity_loss    num_train_layers: {num_train_layers}  nunm_policy_layers: {num_policy_layers}  logits shape:{logits.shape}")
                
                num_blocks = min(num_train_layers, logits.shape[0])
                
                gt = torch.ones((num_blocks)).long().to(self.device)

                if self.opt['diff_sparsity_weights'] and not self.opt['is_sharing']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                    self.losses['sparsity']['task%d' % (t_id + 1)] = 2 * (loss_weights[-num_blocks:] * self.cross_entropy2(logits[-num_blocks:], gt)).mean()
                   
                    print(f"    get_sparsity_loss    loss_weights: {loss_weights}")
                    print(f"    get_sparsity_loss    cross_entropy:   { self.cross_entropy2(logits[-num_blocks:], gt)}   ")
                    print(f"    get_sparsity_loss    loss[sparsity][task{t_id+1}: {self.losses['sparsity']['task%d' % (t_id + 1)] } ")
                else:
                    self.losses['sparsity']['task%d' % (t_id + 1)] = self.cross_entropy_sparsity(logits[-num_blocks:], gt)

                self.losses['sparsity']['total'] += self.losses['sparsity']['task%d' % (t_id + 1)]

        elif self.opt['policy_model'] == 'instance-specific':
            for t_id in range(self.num_tasks):
                logit = getattr(self, 'policy%d' % (t_id+1))
                if num_policy_layers is None:
                    num_policy_layers = logit.shape[1]
                else:
                    assert (num_policy_layers == logit.shape[1])
        
                if num_train_layers is None:
                    num_train_layers = num_policy_layers
        
                num_blocks = min(num_train_layers, logit.shape[1])
                batch_size = logit.shape[0]
                gt = torch.ones((batch_size * num_blocks)).long().to(self.device)
                logit = logit[:, -num_blocks:].contiguous().view(-1, 2)
                if self.opt['diff_sparsity_weights'] and not self.opt['is_sharing']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                    loss_weights = loss_weights.view(1, -1).repeat(batch_size, 1).view(-1)
        
                    self.losses['sparsity']['task%d' % (t_id + 1)] = 2 * ( loss_weights[-batch_size * num_blocks:] * self.cross_entropy2(logit, gt)).mean()
                else:
                    self.losses['sparsity']['task%d' % (t_id + 1)] = self.cross_entropy_sparsity(logit, gt)
        
                self.losses['sparsity']['total'] += self.losses['sparsity']['task%d' % (t_id + 1)]
        
        else:
            raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])


    def get_hamming_loss(self):
        """
        Compute hamming distance 
        """
        self.losses['hamming'] = {}
        self.losses['hamming']['total'] = 0
        num_policy_layers = None
        for t_i in range(self.num_tasks):
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                logits_i = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_i + 1))
            else:
                logits_i = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_i + 1))
         
            if num_policy_layers is None:
                num_policy_layers = logits_i.shape[0]
                if self.opt['diff_sparsity_weights']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                else:
                    loss_weights = (torch.ones((num_policy_layers)).float()).to(self.device)
            else:
                assert (num_policy_layers == logits_i.shape[0])
         
            for t_j in range(t_i, self.num_tasks):
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits_j = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_j + 1))
                else:
                    logits_j = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_j + 1))

                if num_policy_layers is None:
                    num_policy_layers = logits_j.shape[0]
                else:
                    assert (num_policy_layers == logits_j.shape[0])

                print(f"    get_hamming_loss - loss weights {loss_weights.shape} \n  {loss_weights}")
                print(f"    hamming :          \n  {torch.sum(torch.abs(logits_i[:, 0] - logits_j[:, 0]))}")
                print(f"    hamming weighted:  \n  {torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0]))} ")
                self.losses['hamming']['total'] += torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0]))


    def backward_policy(self, lambdas, num_train_layers):
        """
        Compute losses on policy and back-propagate
        """
        self.optimizers['alphas'].zero_grad()
        loss = 0

        for t_id, task in enumerate(self.tasks):
            print(f"\t backward_policy - task {task}  loss is :', {lambdas[t_id]} * {self.losses[task]['total']:.4f}"
                  f" = {lambdas[t_id] * self.losses[task]['total']:.4f}" )            
            loss += lambdas[t_id] * self.losses[task]['total']

        print(f"\t backward_policy - Total Task loss :  {loss:.4f}" )

        if self.opt['is_sharing']:
            self.get_hamming_loss()
            sharing_loss = self.opt['train']['reg_w_hamming'] * self.losses['hamming']['total']
            print(f"\t backward_policy - hamming loss is :', {self.opt['train']['reg_w_hamming']} * {self.losses['hamming']['total']:.5f}"
                  f" = {sharing_loss:.5f}" )
            loss += sharing_loss

        if self.opt['is_sparse']:
            # self.get_sparsity_loss()
            self.get_sparsity_loss2(num_train_layers)
            sparsity_loss = self.opt['train']['reg_w_sparsity'] * self.losses['sparsity']['total']
            print(f"\t backward_policy - sparsity loss is :', {self.opt['train']['reg_w_sparsity']} * {self.losses['sparsity']['total']:.4f}"
                  f" = {sparsity_loss:.5f}" )
            loss += sparsity_loss
        
        print('\t backward_policy - Total (Task + Hamming+ Sparsity) loss:', loss )

        self.losses['total'] = {}
        self.losses['total']['total'] = loss
        self.losses['total']['total'].backward()

        self.optimizers['alphas'].step()


    def backward_network(self, lambdas):
        self.optimizers['weights'].zero_grad()
        loss = 0
        
        for t_id, task in enumerate(self.tasks):
            loss += lambdas[t_id] * self.losses[task]['total']
            print(f"      backward_network - task {task}  loss is :', {lambdas[t_id]} * {self.losses[task]['total']:.4f}"
                  f" = {lambdas[t_id] * self.losses[task]['total']:.4f}" )
        
        print(f"      backward_network - Total loss is    : {loss:.4f}" )

        self.losses['total'] = {}
        self.losses['total']['total'] = loss
        self.losses['total']['total'].backward()
        
        self.optimizers['weights'].step()
        
        if 'weights' in self.schedulers.keys():
            self.schedulers['weights'].step()


    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        if decay_ratio is None:
            self.temp *= self._tem_decay
        else:
            self.temp *= decay_ratio
        print("Change temperature from %.5f to %.5f" % (tmp, self.temp))


    def sample_policy(self, hard_sampling):
        # dist1, dist2 = self.get_policy_prob()
        # print(np.concatenate((dist1, dist2), axis=-1))
        policys = self.networks['mtl-net'].test_sample_policy(hard_sampling)
        for t_id, p in enumerate(policys):
            setattr(self, 'policy%d' % (t_id+1), p)            
            

    def get_policy_prob(self):
        distributions = []
        if self.opt['policy_model'] == 'task-specific':
            for t_id in range(self.num_tasks):
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_id + 1)).detach().cpu().numpy()

                else:
                    logits = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_id + 1)).detach().cpu().numpy()
            distributions.append(softmax(logits, axis=-1))

        elif self.opt['policy_model'] == 'instance-specific':
            for t_id in range(self.num_tasks):
                logit = getattr(self, 'logit%d' % (t_id+1)).detach().cpu().numpy()
                distributions.append(logit.mean(axis=0))
        else:
            raise ValueError('policy mode = %s is not supported' % self.opt['policy_model']  )
        return distributions


    def get_current_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            print(f'get policy{t_id+1}')
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.detach().cpu().numpy()
            policys.append(policy)

        return policys


    def get_current_state(self, current_iter):
        # ##################### change the state of each module ####################################
        current_state = super(BlockDropEnv_Dev, self).get_current_state(current_iter)
        current_state['temp'] = self.temp
        return current_state


    def save_policy(self, label):
        policy = {}
        for t_id in range(self.num_tasks):
            tmp = getattr(self, 'policy%d' % (t_id + 1))
            policy['task%d_policy' % (t_id + 1)] = tmp.cpu().data
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        with open(save_path, 'wb') as handle:
            pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        with open(save_path, 'rb') as handle:
            policy = pickle.load(handle)

        for t_id in range(self.num_tasks):
            print(f"setting policy{t_id+1} attribute ....")
            setattr(self, 'policy%d' % (t_id + 1), policy['task%d_policy' % (t_id+1)])
            print(getattr(self, 'policy%d' % (t_id + 1)))


    def check_exist_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        return os.path.exists(save_path)


    def load_snapshot(self, snapshot):
        super(BlockDropEnv_Dev, self).load_snapshot(snapshot)
        self.temp = snapshot['temp']
        return snapshot['iter']


    def fix_w(self):
        if self.opt['backbone'] == 'WRN':
            network_params = self.get_network_parameters()
            for param in network_params:
                param.requires_grad = False
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for param in self.networks['mtl-net'].module.backbone.parameters():
                    param.requires_grad = False

            else:
                for param in self.networks['mtl-net'].backbone.parameters():
                    param.requires_grad = False

            task_specific_parameters = self.get_task_specific_parameters()
            for param in task_specific_parameters:
                param.requires_grad = False


    def free_w(self, fix_BN):
        if self.opt['backbone'] == 'WRN':
            network_params = self.get_network_parameters()
            for param in network_params:
                param.requires_grad = True
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for name, param in self.networks['mtl-net'].module.backbone.named_parameters():
                    param.requires_grad = True

                    if fix_BN and 'bn' in name:
                        param.requires_grad = False
            else:
                for name, param in self.networks['mtl-net'].backbone.named_parameters():
                    param.requires_grad = True
                    if fix_BN and 'bn' in name:
                        param.requires_grad = False

            task_specific_parameters = self.get_task_specific_parameters()
            for param in task_specific_parameters:
                param.requires_grad = True


    def fix_alpha(self):
        """Fix architecture parameters - disable gradient flow through alpha computation graph"""
        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = False


    def free_alpha(self):
        """Fix architecture parameters - allow gradient flow through alpha computation graph"""
        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = True

    # ##################### change the state of each module ####################################
    def cuda(self, gpu_ids):
        super(BlockDropEnv_Dev, self).cuda(gpu_ids)
        policys = []

        for t_id in range(self.num_tasks):
            if not hasattr(self, 'policy%d' % (t_id+1)):
                return
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.to(self.device)
            policys.append(policy)

        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            setattr(self.networks['mtl-net'].module, 'policys', policys)
        else:
            setattr(self.networks['mtl-net'], 'policys', policys)


    def cpu(self):
        super(BlockDropEnv_Dev, self).cpu()
        print(f'blockdrop_env.cpu()')
        policys = []

        for t_id in range(self.num_tasks):
            if not hasattr(self, 'policy%d' % (t_id+1)):
                return
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.to(self.device)
            policys.append(policy)

        ## nn.DataParallel only applies to GPU configurations
        
        # if isinstance(self.networks['mtl-net'], nn.DataParallel):
        #     setattr(self.networks['mtl-net'].module, 'policys', policys)
        # else:
        setattr(self.networks['mtl-net'], 'policys', policys)
        print(f'environ.cpu policy: {policys}')


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


    def name(self):
        return 'BlockDropEnvDev'


    def set_inputs(self, batch):
        """
        :param batch: {'images': a tensor [batch_size, c, video_len, h, w], 'categories': np.ndarray [batch_size,]}
        """
        self.img = batch['img']
        if torch.cuda.is_available():
            self.img = self.img.to(self.device)

        if 'depth' in self.tasks:
            self.depth = batch['depth']
            if torch.cuda.is_available():
                self.depth = self.depth.to(self.device)
            if 'depth_mask' in batch.keys():
                self.depth_mask = batch['depth_mask']
                if torch.cuda.is_available():
                    self.depth_mask = self.depth_mask.to(self.device)
            if 'depth_policy' in batch.keys():
                self.depth_policy = batch['depth_policy']
                if torch.cuda.is_available():
                    self.depth_policy = self.depth_policy.to(self.device)

        if 'sn' in self.tasks:
            self.normal = batch['normal']
            if torch.cuda.is_available():
                self.normal = self.normal.to(self.device)
            if 'normal_mask' in batch.keys():
                self.sn_mask = batch['normal_mask']
                if torch.cuda.is_available():
                    self.sn_mask = self.sn_mask.to(self.device)
            if 'sn_policy' in batch.keys():
                self.sn_policy = batch['sn_policy']
                if torch.cuda.is_available():
                    self.sn_policy = self.sn_policy.to(self.device)

        if 'seg' in self.tasks:
            self.seg = batch['seg']
            if torch.cuda.is_available():
                self.seg = self.seg.to(self.device)
            if 'seg_mask' in batch.keys():
                self.seg_mask = batch['seg_mask']
                if torch.cuda.is_available():
                    self.seg_mask = self.seg_mask.to(self.device)
            if 'seg_policy' in batch.keys():
                self.seg_policy = batch['seg_policy']
                if torch.cuda.is_available():
                    self.seg_policy = self.seg_policy.to(self.device)

        if 'keypoint' in self.tasks:
            self.keypoint = batch['keypoint']
            if torch.cuda.is_available():
                self.keypoint = self.keypoint.to(self.device)
            if 'keypoint_policy' in batch.keys():
                self.keypoint_policy = batch['keypoint_policy']
                if torch.cuda.is_available():
                    self.keypoint_policy = self.keypoint_policy.to(self.device)

        if 'edge' in self.tasks:
            self.edge = batch['edge']
            if torch.cuda.is_available():
                self.edge = self.edge.to(self.device)
            if 'edge_policy' in batch.keys():
                self.edge_policy = batch['edge_policy']
                if torch.cuda.is_available():
                    self.edge_policy = self.edge_policy.to(self.device)


    def resize_results(self):

        new_shape = self.img.shape[-2:]
        if 'seg' in self.tasks:
            print(f"resize results  {self.seg_pred.shape} to {new_shape}")
            self.seg_output = F.interpolate(self.seg_pred, size=new_shape)

        if 'sn' in self.tasks:
            print(f"resize results  {self.sn_pred.shape} to {new_shape}")
            self.sn_output = F.interpolate(self.sn_pred, size=new_shape)

        if 'depth' in self.tasks:
            print(f"resize results  {self.depth_pred.shape} to {new_shape}")
            self.depth_output = F.interpolate(self.depth_pred, size=new_shape)

        if 'keypoint' in self.tasks:
            print(f"resize results  {self.keypoint_pred.shape} to {new_shape}")
            self.keypoint_output = F.interpolate(self.keypoint_pred, size=new_shape)

        if 'edge' in self.tasks:
            print(f"resize results  {self.edge_pred.shape} to {new_shape}")
            self.edge_output = F.interpolate(self.edge_pred, size=new_shape)


    def get_seg_loss(self, seg_num_class, instance=False):
        """
        seg_pred: semantic segmentation predictions. shape: (BatchSize, NumClasses, W, H)
        """
        self.losses['seg'] = {}
        ## convert to (Batchsize x W x H, NumClasses )
        prediction = self.seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        print(f"       get_seg_loss:  self.seg_pred:{self.seg_pred.shape}   prediction: {prediction.shape}")
        
        ## build ground truth 
        batch_size = self.seg_pred.shape[0]
        new_shape  = self.seg_pred.shape[-2:]
        seg_resize = F.interpolate(self.seg.float(), size=new_shape)
        gt = seg_resize.permute(0, 2, 3, 1).contiguous().view(-1)
        print(f"       seg:{seg_resize.shape}   seg_resize:{seg_resize.shape}    gt: {gt.shape}")
        
        # max_label_num = prediction.argmax(dim=-1).max()
        # print(max_label_num)
        
        loss = self.cross_entropy(prediction, gt.long())
        self.losses['seg']['total'] = loss
        
        if instance:
            instance_loss = self.cross_entropy2(prediction, gt.long()).view(batch_size, -1).mean(dim=-1)
            return instance_loss
        else:
            return None


    def get_sn_loss(self, instance=False):
        """
        define the surface normal loss 
        """
        self.losses['sn'] = {}

        ## convert to (Batchsize x W x H, 3 )
        prediction = self.sn_pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        print(f"       get_seg_loss:  self.sn_pred:{self.sn_pred.shape}   prediction: {prediction.shape}")
        
        ## build ground truth 
        new_shape = self.sn_pred.shape[-2:]
        sn_resize = F.interpolate(self.normal.float(), size=new_shape)
        gt = sn_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        print(f"       normal:{self.normal.shape}   seg_resize:{sn_resize.shape}    gt: {gt.shape}")
        
        labels = (gt.max(dim=1)[0] < 255)
        if hasattr(self, 'normal_mask'):
            normal_mask_resize = F.interpolate(self.normal_mask.float(), size=new_shape)
            gt_mask = normal_mask_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels and gt_mask.int() == 1

        prediction = prediction[labels]
        gt = gt[labels]

        prediction = F.normalize(prediction)
        gt = F.normalize(gt)

        self.losses['sn']['total'] = 1 - self.cosine_similiarity(prediction, gt).mean()

        if instance:
            batch_size = self.sn_pred.shape[0]
            instance_stats = labels.view(batch_size, -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = 1 - self.cosine_similiarity(prediction, gt)
            cuda_device = self.sn_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx - 1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None


    def get_depth_loss(self, instance=False):
        """
        define the depth loss 
        """
        self.losses['depth'] = {}
        new_shape = self.depth_pred.shape[-2:]
        depth_resize = F.interpolate(self.depth.float(), size=new_shape)
        if hasattr(self, 'depth_mask'):
            depth_mask_resize = F.interpolate(self.depth_mask.float(), size=new_shape)

        if self.dataset in ['NYU_v2', 'CityScapes']:
            binary_mask = (torch.sum(depth_resize, dim=1) > 3 * 1e-5).unsqueeze(1).to(self.device)
        elif self.dataset == 'Taskonomy' and hasattr(self, 'depth_mask'):
            binary_mask = (depth_resize != 255) * (depth_mask_resize.int() == 1)
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        depth_output = self.depth_pred.masked_select(binary_mask)
        depth_gt = depth_resize.masked_select(binary_mask)

        # torch.sum(torch.abs(self.depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        self.losses['depth']['total'] = self.l1_loss(depth_output, depth_gt)
        if instance:
            instance_stats = binary_mask.view(binary_mask.shape[0], -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = self.l1_loss2(depth_output, depth_gt)
            batch_size = self.depth_pred.shape[0]
            cuda_device = self.depth_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx-1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None


    def get_keypoint_loss(self, instance=False):
        self.losses['keypoint'] = {}
        new_shape = self.keypoint_pred.shape[-2:]
        keypoint_resize = F.interpolate(self.keypoint.float(), size=new_shape)
        if self.dataset == 'Taskonomy':
            binary_mask = keypoint_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        keypoint_output = self.keypoint_pred.masked_select(binary_mask)
        keypoint_gt = keypoint_resize.masked_select(binary_mask)
        # torch.sum(torch.abs(self.depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        self.losses['keypoint']['total'] = self.l1_loss(keypoint_output, keypoint_gt)
        if instance:
            instance_stats = binary_mask.view(binary_mask.shape[0], -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = self.l1_loss2(keypoint_output, keypoint_gt)
            batch_size = self.keypoint_pred.shape[0]
            cuda_device = self.keypoint_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx - 1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None


    def get_edge_loss(self, instance=False):
        self.losses['edge'] = {}
        new_shape = self.edge_pred.shape[-2:]
        edge_resize = F.interpolate(self.edge.float(), size=new_shape)
        if self.dataset == 'Taskonomy':
            binary_mask = edge_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        edge_output = self.edge_pred.masked_select(binary_mask)
        edge_gt = edge_resize.masked_select(binary_mask)
        # torch.sum(torch.abs(self.depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        self.losses['edge']['total'] = self.l1_loss(edge_output, edge_gt)
        if instance:
            instance_stats = binary_mask.view(binary_mask.shape[0], -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = self.l1_loss2(edge_output, edge_gt)
            batch_size = self.edge_pred.shape[0]
            cuda_device = self.edge_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx - 1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None


    def seg_error(self, seg_num_class):
        """
        Returns:
        
        predictions: Predictions (y_hat)
        gt:          Ground truth values
        pixelAcc:    Pixel accuracy 
        err:         Cross Entropy loss
        """
        gt = self.seg.view(-1)
        labels = gt < seg_num_class
        gt = gt[labels].int()

        logits = self.seg_output.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        logits = logits[labels]
        err = self.cross_entropy(logits, gt.long())

        prediction = torch.argmax(self.seg_output, dim=1)
        prediction = prediction.unsqueeze(1)

        # pixel acc
        prediction = prediction.view(-1)
        prediction = prediction[labels].int()
        pixelAcc = (gt == prediction).float().mean()

        return prediction.cpu().numpy(), gt.cpu().numpy(), pixelAcc, err.cpu().numpy()


    def normal_error(self):
        # normalized, ignored gt and prediction
        prediction = self.sn_output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt = self.normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        labels = gt.max(dim=1)[0] != 255
        if hasattr(self, 'normal_mask'):
            gt_mask = self.normal_mask.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels and gt_mask.int() == 1

        gt = gt[labels]
        prediction = prediction[labels]

        gt = F.normalize(gt.float(), dim=1)
        prediction = F.normalize(prediction, dim=1)

        cos_similarity = self.cosine_similiarity(gt, prediction)

        return cos_similarity.cpu().numpy()


    def depth_error(self):
        if self.dataset in ['NYU_v2', 'CityScapes']:
            binary_mask = (torch.sum(self.depth, dim=1) > 3 * 1e-5).unsqueeze(1).to(self.device)
        elif self.dataset == 'Taskonomy' and hasattr(self, 'depth_mask'):
            binary_mask = (self.depth != 255) * (self.depth_mask.int() == 1)
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        depth_output_true = self.depth_output.masked_select(binary_mask)
        depth_gt_true = self.depth.masked_select(binary_mask)
        abs_err = torch.abs(depth_output_true - depth_gt_true)
        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
        sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
        abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)
        rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)
        sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)
        # calcuate the sigma
        term1 = depth_output_true / depth_gt_true
        term2 = depth_gt_true / depth_output_true
        ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
        # calcualte rms
        rms = torch.pow(depth_output_true - depth_gt_true, 2)
        rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)

        return abs_err.cpu().numpy(), rel_err.cpu().numpy(), sq_rel_err.cpu().numpy(), ratio[0].cpu().numpy(), \
               rms.cpu().numpy(), rms_log.cpu().numpy()


    def keypoint_error(self):
        binary_mask = (self.keypoint != 255)
        keypoint_output_true = self.keypoint_output.masked_select(binary_mask)
        keypoint_gt_true = self.keypoint.masked_select(binary_mask)
        abs_err = torch.abs(keypoint_output_true - keypoint_gt_true).mean()
        return abs_err.cpu().numpy()


    def edge_error(self):
        binary_mask = (self.edge != 255)
        edge_output_true = self.edge_output.masked_select(binary_mask)
        edge_gt_true = self.edge.masked_select(binary_mask)
        abs_err = torch.abs(edge_output_true - edge_gt_true).mean()
        return abs_err.cpu().numpy()