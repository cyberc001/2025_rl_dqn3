import torch.nn as nn
from utils import NoisyLinear

class QNet(nn.Module):
    '''
        state_dim - размерность вектора состояния (входного слоя)
        action_dim - размерность пространства действий (количество возможных действий) (выходного слоя)
        hidden_shape - список размеров скрытых слоёв
    '''
    def __init__(self, state_dim, action_dim, hidden_shape, use_noisy=False):
        self.use_noisy = use_noisy
        super(QNet, self).__init__()
        layers = [state_dim] + list(hidden_shape) + [action_dim]
        model_layers = []
        for i in range(len(layers) - 1):
            in_f, out_f = layers[i], layers[i+1]
            if use_noisy and i < len(layers)-2:  # Только скрытые слои
                model_layers += [NoisyLinear(in_f, out_f), nn.ReLU()]
            else:
                model_layers += [nn.Linear(in_f, out_f), nn.ReLU() if i < len(layers) - 2 else nn.Identity()]
        self.model = nn.Sequential(*model_layers)

    def forward(self, s):
        return self.model(s)
    
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
