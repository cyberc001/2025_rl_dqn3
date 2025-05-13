import numpy as np
import gymnasium as gym
import argparse
import torch
import signal
import sys
from pr_agent import PRAgent, PrioritizedReplayBuffer
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type = str, default = 'prio_replay', help = 'Policy update algorithm: "prio_replay"')
parser.add_argument('--seed', type = int, default = 0, help = 'RNG seed, or 0 for auto')
parser.add_argument('--max-epochs', type = int, default = 100000, help = 'Number of epochs after which training terminates')
parser.add_argument('--gamma', type = float, default = 0.99, help = 'Discount factor')
parser.add_argument('--hidden_width', type = int, default = 256, help = 'Hidden network width')
parser.add_argument('--lr', type = float, default = 0.0001, help = 'Learning rate')
parser.add_argument('--batch_size', type = int, default = 256, help = 'Mini-batch size')
parser.add_argument('--buffer_size', type = int, default = 100000, help = 'Replay buffer size')
parser.add_argument('--render', type = bool, default = False, help = 'Render gymnasium environment')

parser.add_argument('--use-noisy', action='store_true', help='Enable Noisy Nets')
parser.add_argument('--dueling', action='store_true', help='Use DuelingDQN')
# ??? не знаю относятся к алгоритму или нет
# Я так понимаю, что онтносится, влияет на эпсилон. Эпсилон изменяется линейно от 0.5 к 0.1, 
# влияет на то насколько "ошибается" модель в выборе действия,
# выступает в качестве решения проблемы разведка/эксплуатация

parser.add_argument('--noise_init', type = float, default = 0.5, help = 'Initial exploration noise')
parser.add_argument('--noise_final', type = float, default = 0.1, help = 'Final exploration noise')
parser.add_argument('--noise_decay_epochs', type = float, default = 100000, help = 'Amount of epochs to read noise_final')

parser.add_argument('--warmup-epochs', type = int, default = 1000, help = '[Prioritized Replay] Number of epochs after which Q-Net weights are used for action sampling')
parser.add_argument('--pr_alpha', type = float, default = 0.6, help = '[Prioritized Replay] Alpha')
parser.add_argument('--pr_beta', type = float, default = 0.6, help = '[Prioritized Replay] Initial beta')
parser.add_argument('--pr_beta_gain_steps', type = float, default = 0.6, help = '[Prioritized Replay] Number of epochs to increment beta from pr_beta to 1.0')

opt = parser.parse_args()
plot_title = ''
if opt.use_noisy:
    plot_title = 'noisy nets'
elif opt.dueling:
    plot_title = 'Dueling DQN'
else:
    plot_title = 'DQN'
print('Starting ' + plot_title)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using CUDA device', device)

total_reward = 0
plot_x = []
plot_y = []
def sigint_hndl(sig, frame):
    plt.plot(plot_x, plot_y)
    plt.suptitle(plot_title)
    plt.xlabel('Количество эпох')
    plt.ylabel('Средняя награда')
    plt.show()
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_hndl)


if __name__ == '__main__':
    env = gym.make('LunarLander-v3', render_mode = 'human' if opt.render else None)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_ep_steps = env._max_episode_steps

    if opt.seed != 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

    torch.backends.cudnn.deterministic = True

    if opt.algo == 'prio_replay' or opt.dueling:
        opt.replay_buffer = PrioritizedReplayBuffer(device, opt)
        agent = PRAgent(device, opt, use_dueling=opt.dueling)

    total_steps = 0
    while total_steps < opt.max_epochs:
        s, info = env.reset()
        done, ep_step = False, 0
        while not done:
            ep_step += 1

            if agent.replay_buffer.size < opt.warmup_epochs:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s, deterministic = False)

            s_next, r, dw, tr, info = env.step(a)
            print(total_steps, '. reward:', r)
            if r <= -100: # ???
                r = -10

            # обновить график
            total_reward += r
            if total_steps >= opt.warmup_epochs and total_steps % 100 == 0:
                plot_x.append(total_steps)
                plot_y.append(total_reward / total_steps)

            agent.replay_buffer.add(s, a, r, s_next, dw)
            done = dw or tr
            s = s_next
            
            agent.pre_update(total_steps)

            if total_steps >= opt.warmup_epochs and total_steps % 50 == 0:
                for i in range(50):
                    agent.train()

            total_steps += 1

            if(total_steps) % 10000 == 0:
                print('Saving model...')
                agent.save('LunarLander-v2', total_steps)

    env.close()
