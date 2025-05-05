import torch.nn as nn
from utils import NoisyLinear

class DuelingQNet(nn.Module):
    '''
        state_dim - размерность вектора состояния (входного слоя)
        action_dim - размерность пространства действий (количество возможных действий) (выходного слоя)
        hidden_shape - список размеров скрытых слоёв
    '''
    def __init__(self, state_dim, action_dim, hidden_shape, use_noisy=False):
        self.use_noisy = use_noisy
        super(QNet, self).__init__()
	self.use_noisy = use_noisy
        self.action_dim = action_dim

	#слои V/A
	self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_shape[0]),
            nn.ReLU(),
            nn.Linear(hidden_shape[0], hidden_shape[1]),
            nn.ReLU()
        )

	# ветка value (v(s))
	self.value_stream = nn.Sequential(
            NoisyLinear(hidden_shape[1], 1, sigma_init=0.5) if use_noisy else nn.Linear(hidden_shape[1], 1),
            nn.Identity()  # выход скаляра (v(s))
        )

	# ветка advantage (A(s,a))
	self.value_stream = nn.Sequential(
            NoisyLinear(hidden_shape[1], 1, sigma_init=0.5) if use_noisy else nn.Linear(hidden_shape[1], 1),
            nn.Identity()  
        )
    def forward(self, s):
	features = self.feature_layers(s)
        value = self.value_stream(features)  
        advantage = self.advantage_stream(features)  
	
	# формула dueling dqn
	q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
        #return self.model(s)
    
    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear): 
                    module.reset_noise()

''' Базовый класс для алгоритмов DQN '''
class DQNAgent():
    def __init__(self, opt):
        pass

    def select_action(self, state, deterministic = True):
        pass
    def pre_update(self, total_steps):
        pass
    def train(self):
        pass

    def save(self, env_name, steps):
        pass
    def load(self, env_name, steps):
        pass
