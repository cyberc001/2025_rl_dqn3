import copy
import numpy as np
import torch
import os
from dqn_agent import DQNAgent, QNet
from utils import LinearSchedule, SumTree

class PRAgent(DQNAgent):
    def __init__(self, device, opt):
        self.device = device
        self.qnet = QNet(opt.state_dim, opt.action_dim, (opt.hidden_width, opt.hidden_width)).to(device)
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr = opt.lr)
        self.qtarget = copy.deepcopy(self.qnet)
        for p in self.qtarget.parameters():
            p.requires_grad = False

        self.gamma = opt.gamma
        self.tau = 0.005
        self.batch_size = opt.batch_size
        self.noise = opt.noise_init
        self.action_dim = opt.action_dim
        self.replay_buffer = opt.replay_buffer

        self.noise_scheduler = LinearSchedule(opt.noise_decay_epochs, opt.noise_init, opt.noise_final)
        self.beta_scheduler = LinearSchedule(opt.pr_beta_gain_steps, opt.pr_beta, 1.0)

    def select_action(self, state, deterministic = True):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if deterministic:
                return self.qnet(state).argmax().item()
            else:
                if np.random.rand() < self.noise:
                    return np.random.randint(0, self.action_dim)
                else:
                    return self.qnet(state).argmax().item()

    def pre_update(self, total_steps):
        self.noise = self.noise_scheduler.value(total_steps)
        self.replay_buffer.beta = self.beta_scheduler.value(total_steps)

    def train(self):
        s, a, r, s_prime, dw_mask, ind, norm_is_w = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            max_q_prime = self.qtarget(s_prime).max(1)[0].unsqueeze(1)
            targetq = r + (1 - dw_mask) * self.gamma * max_q_prime

        current_q_a = self.qnet(s).gather(1, a)
        td_errors = (current_q_a - targetq).squeeze(-1)
        loss = (norm_is_w * (td_errors ** 2)).mean()

        self.replay_buffer.update_batch_priorities(ind, td_errors.detach().cpu().numpy())

        self.qnet_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.qnet_optimizer.step()

        for param, target_param in zip(self.qnet.parameters(), self.qtarget.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, env_name, steps):
        if not os.path.exists('./pr_agent'):
            os.mkdir('./pr_agent')
        torch.save(self.qnet.state_dict(), './pr_agent/{}_{}.pth'.format(env_name, steps))

    def load(self, env_name, steps):
        state = torch.load('./pr_agent/{}_{}.pth').format(env_name, steps)
        self.qnet.load_state_dict(state, map_location = self.device)
        self.qtarget.load_state_dict(state, map_location = self.device)
        
class PrioritizedReplayBuffer():
    def __init__(self, device, opt):
        self.cur_idx = 0
        self.size = 0

        max_size = opt.buffer_size
        self.state = np.zeros((max_size, opt.state_dim))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, opt.state_dim))
        self.dw = np.zeros((max_size, 1))
        self.max_size = max_size

        self.sum_tree = SumTree(max_size)
        self.alpha = opt.pr_alpha
        self.beta = opt.pr_beta

        self.device = device

    def add(self, state, action, reward, next_state, dw):
        self.state[self.cur_idx] = state
        self.action[self.cur_idx] = action
        self.reward[self.cur_idx] = reward
        self.next_state[self.cur_idx] = next_state
        self.dw[self.cur_idx] = dw

        self.sum_tree.update_priority(data_index = self.cur_idx, priority = 1.0 if self.size == 0 else self.sum_tree.priority_max)

        self.cur_idx = self.cur_idx + 1
        if self.cur_idx >= self.max_size:
            self.cur_idx = 0
        if self.size < self.max_size:
            self.size += 1

    def sample(self, batch_size):
        ind, norm_is_w = self.sum_tree.prioritized_sample(N = self.size, batch_size = batch_size, beta = self.beta)
        return (torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
                torch.tensor(self.action[ind], dtype=torch.long).to(self.device),
                torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
                torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
                torch.tensor(self.dw[ind], dtype=torch.float32).to(self.device),
                ind,
                norm_is_w.to(self.device))

    def update_batch_priorities(self, batch_index, td_errors):
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for ind, priority in zip(batch_index, priorities):
            self.sum_tree.update_priority(data_index = ind, priority = priority)
