import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SumTree(object):

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity
        self.tree_capacity = 2 * buffer_capacity - 1
        self.tree = np.zeros(self.tree_capacity)

    def update_priority(self, data_index, priority):
        tree_index = data_index + self.buffer_capacity - 1
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # then propagate the change through the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def prioritized_sample(self, N, batch_size, beta):
        batch_index = np.zeros(batch_size, dtype=np.uint32)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            buffer_index, priority = self._get_index(v)
            batch_index[i] = buffer_index
            prob = priority / self.priority_sum
            IS_weight[i] = (N * prob) ** (-beta)
        Normed_IS_weight = IS_weight / IS_weight.max()  # normalization

        return batch_index, Normed_IS_weight

    def _get_index(self, v):
        ''' sample a index '''
        parent_idx = 0
        while True:
            child_left_idx = 2 * parent_idx + 1
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:
                tree_index = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # tree_index->data_index
        return data_index, self.tree[tree_index]

    @property
    def priority_sum(self):
        return self.tree[0]

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()


def evaluate_policy(env, model, turns = 3):
    scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a = model.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a) # dw: terminated; tr: truncated
            done = dw + tr
            scores += r
            s = s_next
    return int(scores/turns)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, initial_p, final_p):
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
    

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('epsilon_weight', torch.Tensor(out_features, in_features))

        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('epsilon_bias', torch.Tensor(out_features))
        else:
            self.sigma_bias = None
            self.epsilon_bias = None

        self.init_noise_parameters()

    def init_noise_parameters(self):
        """Инициализация шумовых параметров (sigma_weight, sigma_bias)"""
        self.sigma_weight.data.fill_(self.sigma_init)
        if self.bias is not None:
            self.sigma_bias.data.fill_(self.sigma_init)

    def reset_parameters(self):
        """Инициализация весов и bias (вызывается из родителя)"""
        std = 1 / math.sqrt(self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)
    

    def forward(self, input):
        def scale_noise(size):
            x = torch.randn(size)
            return x.sign().mul(x.abs().sqrt())

        epsilon_in = scale_noise(self.in_features)
        epsilon_out = scale_noise(self.out_features)
        device = self.weight.device
        weight_epsilon = epsilon_out.ger(epsilon_in).to(device)
        bias_epsilon = epsilon_out.to(device) if self.bias is not None else None

        if self.training:
            weight = self.weight + self.sigma_weight * weight_epsilon
            bias = self.bias + self.sigma_bias * bias_epsilon if self.bias is not None else None
        else:
            weight = self.weight
            bias = self.bias

        return F.linear(input, weight, bias)

    def reset_noise(self):
        """Сбрасывает шум после каждого шага обучения"""
        device = self.weight.device
        epsilon_in = torch.randn(self.in_features, device=device)
        epsilon_out = torch.randn(self.out_features, device=device)
        self.epsilon_weight = epsilon_out.ger(epsilon_in)
        if self.bias is not None:
            self.epsilon_bias = epsilon_out
