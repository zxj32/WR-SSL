import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
from torch.nn import init
import math


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class MetaWeightBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, input, w):
        if self.training:
            w_exd = w.view([w.shape[0], 1, 1, 1]).expand_as(input)
            mean = (w_exd * input).mean([0, 2, 3])
            mean = mean * input.shape[0] / w.sum()
            var_, var = get_var2d(input, mean, w)

            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)

            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    var_.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - self.running_mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

class MetaWeightBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, input, w):
        if self.training:
            w_exd = w.view([w.shape[0], 1]).expand_as(input)
            mean = (w_exd * input).mean([0])
            mean = mean * input.shape[0] / w.sum()
            var_, var = get_var1d(input, mean, w)

            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)

            return self.weight.view([1, self.num_features]).expand_as(input) * (input - mean.view([1, self.num_features]).expand_as(input)) / (
                    var_.view([1, self.num_features]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features]).expand_as(input) * (input - self.running_mean.view([1, self.num_features]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features]).expand_as(input)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

def get_var2d(x, new_mean, w):
    scale = x.size(0) * x.size(2) * x.size(3) * w.sum() / x.size(0)
    new_mean = new_mean.view([1, new_mean.shape[0], 1, 1]).expand_as(x)
    tmp = torch.pow((x - new_mean), 2)
    tem_v = tmp * w.view([w.shape[0], 1, 1, 1]).expand_as(tmp)
    new_var = tem_v.mean([0, 2, 3]) * x.size(0) / w.sum()
    runing_temp_var = new_var * scale / (scale - 1)
    return new_var, runing_temp_var

def get_var1d(x, new_mean, w):
    scale = x.size(0) * w.sum() / x.size(0)
    new_mean = new_mean.view([1, new_mean.shape[0]]).expand_as(x)
    tmp = torch.pow((x - new_mean), 2)
    tem_v = tmp * w.view([w.shape[0], 1]).expand_as(tmp)
    new_var = tem_v.mean([0]) * x.size(0) / w.sum()
    runing_temp_var = new_var * scale / (scale - 1)
    return new_var, runing_temp_var



class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d_new(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats
        self.update_batch_stats = True

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        if self.update_batch_stats:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                self.training or not self.track_running_stats, self.momentum, self.eps)
        else:
            return F.batch_norm(x, None, None, self.weight, self.bias,
                                self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MLPNet(MetaModule):
    def __init__(self, input, hidden1, hidden2, hidden3, output, p):
        super(MLPNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, hidden2)
        self.linear3 = MetaLinear(hidden2, hidden3)
        self.linear4 = MetaLinear(hidden3, output)
        # self.drop_layer = nn.Dropout(p=p)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x= self.drop_layer(x)
        x = self.linear2(x)
        x = self.relu(x)
        # x= self.drop_layer(x)
        x = self.linear3(x)
        x = self.relu(x)
        # x= self.drop_layer(x)
        out = self.linear4(x)
        return out.squeeze()
    def reset_param(self):
        for m in self.modules():
            if isinstance(m, MetaLinear):
                m.reset_parameters()

class MLPNet_Batch(MetaModule):
    def __init__(self, input, hidden1, hidden2, hidden3, output, p=0.5):
        super(MLPNet_Batch, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, hidden2)
        self.linear3 = MetaLinear(hidden2, hidden3)
        self.linear4 = MetaLinear(hidden3, output)
        self.batch1 = MetaBatchNorm2d_new(hidden1)
        self.batch2 = MetaBatchNorm2d_new(hidden2)
        self.batch3 = MetaBatchNorm2d_new(hidden3)
        # self.drop_layer = nn.Dropout(p=p)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch1(x)
        x = self.relu(x)
        # x= self.drop_layer(x)
        x = self.batch2(x)
        x = self.linear2(x)
        x = self.relu(x)
        # x= self.drop_layer(x)
        x = self.linear3(x)
        x = self.batch3(x)
        x = self.relu(x)
        # x= self.drop_layer(x)
        out = self.linear4(x)
        return out.squeeze()

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm2d_new):
                m.update_batch_stats = flag


class MLPNet_Batch_weight(MetaModule):
    def __init__(self, input, hidden1, hidden2, hidden3, output, p=0.5):
        super(MLPNet_Batch_weight, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, hidden2)
        self.linear3 = MetaLinear(hidden2, hidden3)
        self.linear4 = MetaLinear(hidden3, output)
        self.batch1 = MetaWeightBatchNorm1d(hidden1)
        self.batch2 = MetaWeightBatchNorm1d(hidden2)
        self.batch3 = MetaWeightBatchNorm1d(hidden3)

    def forward(self, x, w):
        x = self.linear1(x)
        x = self.batch1(x, w)
        x = self.relu(x)
        x = self.batch2(x, w)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.batch3(x, w)
        x = self.relu(x)
        out = self.linear4(x)
        return out.squeeze()

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, MetaWeightBatchNorm1d):
                m.update_batch_stats = flag

    def reset_batch_param(self):
        for m in self.modules():
            if isinstance(m, MetaWeightBatchNorm1d):
                m.reset_running_stats()

    def reset_param(self):
        for m in self.modules():
            if isinstance(m, MetaWeightBatchNorm1d):
                m.reset_running_stats()
            elif isinstance(m, MetaLinear):
                m.reset_parameters()

