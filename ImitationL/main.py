import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import utils
from agent import BehaviorClone, GAIL
from PPO import PPO

def test_agent(agent, env, n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)
def bc():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    lr = 1e-3
    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr)
    n_iterations = 1000
    batch_size = 64
    test_returns = []
    expert_s = np.loadtxt("expert_s")
    expert_a = np.loadtxt("expert_a").astype(np.int64)
    with tqdm(total=n_iterations, desc="进度条") as pbar:
        for i in range(n_iterations):
            sample_indices = np.random.randint(low=0,
                                               high=expert_s.shape[0],
                                               size=batch_size)
            bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
            current_return = test_agent(bc_agent, env, 5)
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(test_returns)))
    plt.plot(iteration_list, test_returns)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('BC on {}'.format(env_name))
    plt.show()

def GAIL_f():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    lr_d = 1e-3
    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d)
    n_episode = 500
    return_list = []
    expert_s = np.loadtxt("expert_s")
    expert_a = np.loadtxt("expert_a").astype(np.int64)

    with tqdm(total=n_episode, desc="进度条") as pbar:
        for i in range(n_episode):
            episode_return = 0
            state = env.reset()[0]
            done = False
            state_list = []
            action_list = []
            next_state_list = []
            done_list = []
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list,
                       next_state_list, done_list)

            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(return_list)))
    plt.plot(iteration_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GAIL on {}'.format(env_name))
    plt.show()

if __name__ == '__main__':
   GAIL_f()