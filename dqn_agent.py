import torch.nn as nn

class QNet(nn.Module):
    '''
        state_dim - размерность вектора состояния (входного слоя)
        action_dim - размерность пространства действий (количество возможных действий) (выходного слоя)
        hidden_shape - список размеров скрытых слоёв
    '''
    def __init__(self, state_dim, action_dim, hidden_shape):
        super(QNet, self).__init__()
        layers = [state_dim] + list(hidden_shape) + [action_dim]

        # создание модели
        model_layers = []
        for i in range(len(layers) - 1):
            model_layers += [nn.Linear(layers[i], layers[i + 1]), nn.ReLU() if i < len(layers) - 2 else nn.Identity()]
        self.model = nn.Sequential(*model_layers)

    def forward(self, s):
        return self.model(s)

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
