import numpy as np
import torch

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