import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from env import WorldEnv, ReplayBuffer_Trajectory, Trajectory
from agent import DDPG

def HER(her=False):
    actor_lr = 1e-3
    critic_lr = 1e-3
    hidden_dim = 128
    state_dim = 4
    action_dim = 2
    action_bound = 1
    sigma = 0.1
    tau = 0.005
    gamma = 0.98
    num_episodes = 2000
    n_train = 20
    batch_size = 256
    minimal_episodes = 200
    buffer_size = 10000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                 critic_lr, sigma, tau, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= minimal_episodes:
                    for _ in range(n_train):
                        transition_dict = replay_buffer.sample(batch_size, her)
                        agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    if her:
        plt.title('DDPG with HER on {}'.format('GridWorld'))
    else:
        plt.title('DDPG without HER on {}'.format('GridWorld'))
    plt.show()

if __name__ == '__main__':
    HER(True)
    HER(False)