
from deeplab_resnet import * 

class MTL2(nn.Module):
    """
    Create the architecture based on the Deep lab ResNet backbone
    """
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0):
        super(MTL2, self).__init__()

        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)

        ## define task specific layers 
        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        ## Initialize policies - there is one policy for each task
        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'logits' in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params

    def network_parameters(self):
        """
        return backbone + task_specific parameters
        """
        params = []
        for name, param in self.named_parameters():
            if not ('task' in name and 'logits' in name):
                params.append(param)
        return params


    def train_sample_policy(self, temperature, hard_sampling):
        print(f' MTL2 TRAIN SAMPLE POLICY')
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(getattr(self, 'task%d_logits' % (t_id + 1)), temperature, hard=hard_sampling)
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        print(f' MTL2 TEST SAMPLE POLICY')
        self.policys = []
        
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to('cuda:%d' % cuda_device)
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to('cuda:%d' % cuda_device)
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, 'task%d_logits' % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                # setattr(self, 'policy%d' % t_id, policy)
                self.policys.append(policy)

        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == 'all_chosen':
                assert(self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer, 2)
                task_logits[:, 0] = 0
            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randn(num_layers-self.skip_layer, 2)
            elif self.init_method == 'equal':
                task_logits = 0.5 * torch.ones(num_layers-self.skip_layer, 2)
            else:
                raise NotImplementedError('Init Method %s is not implemented' % self.init_method)

            self._arch_parameters = []
            self.register_parameter('task%d_logits' % (t_id + 1), nn.Parameter(task_logits, requires_grad=True))
            self._arch_parameters.append(getattr(self, 'task%d_logits' % (t_id + 1)))

    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):
        # print('** MTL2 num_train_layers in mtl forward = ', num_train_layers, 'is_policy: ', is_policy)

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer
            # print(f'self.layers: {sum(self.layers)}   self.skip_layer: {self.skip_layer}    num_train_layers: {num_train_layers}')

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # print(f'min: {sum(self.layers) - self.skip_layer}    num_train_layers: {num_train_layers}')
        
        # Generate features
        cuda_device = img.get_device()
        # print(f'Cuda Device is : {cuda_device}')

        if is_policy:
            if mode == 'train':
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == 'eval':
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == 'fix_policy':
                self.policys = self.test_sample_policy(hard_sampling)
                # pass
                # for p in self.policys:
                #     assert(p is not None)
            else:
                raise NotImplementedError('mode %s is not implemented' % mode)

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            # print(f' forward  - skip layer: {skip_layer}')
            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            padding[:, 1] = 0

            padding_policys = []
            feats = []
            for t_id in range(self.num_tasks):
                padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()), dim=0)
                padding_policys.append(padding_policy)

                feats.append(self.backbone(img, padding_policy))
        else:
            feats = [self.backbone(img)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats[t_id])
            outputs.append(output)

        return outputs, self.policys, [None] * self.num_tasks
