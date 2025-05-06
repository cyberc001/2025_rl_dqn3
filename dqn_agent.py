import torch.nn as nn
from utils import NoisyLinear

class DuelingQNet(nn.Module):
    '''
        state_dim - размерность вектора состояния (входного слоя)
        action_dim - размерность пространства действий (количество возможных действий) (выходного слоя)
        hidden_shape - список размеров скрытых слоёв
    '''
    def __init__(self, state_dim, action_dim, hidden_shape, use_noisy=False):
        super().__init__()
        self.use_noisy = use_noisy
        self.action_dim = action_dim
        
        # Общие слои
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_shape[0]),
            nn.ReLU(),
            nn.Linear(hidden_shape[0], hidden_shape[1]),
            nn.ReLU()
        )
        
        # Ветви Value и Advantage
        if use_noisy:
            self.value_stream = NoisyLinear(hidden_shape[1], 1)
            self.advantage_stream = NoisyLinear(hidden_shape[1], action_dim)
        else:
            self.value_stream = nn.Linear(hidden_shape[1], 1)
            self.advantage_stream = nn.Linear(hidden_shape[1], action_dim)

    def forward(self, s):
        features = self.feature_layers(s)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Формула Dueling DQN
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear): 
                    module.reset_noise()


class DQNAgent():
    def __init__(self, opt):
        pass

    def select_action(self, state, deterministic=True):
        pass
        
    def pre_update(self, total_steps):
        pass
        
    def train(self):
        pass

    def save(self, env_name, steps):
        pass
        
    def load(self, env_name, steps):
        pass
