import random
import gym
import torch
import utils
import matplotlib.pyplot as plt
import numpy as np

from agent import PPO

def discrete():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    return_list = utils.train_on_policy_agent(env, agent, num_episodes)

    plot(return_list, env_name)

    def sample_expert_data(n_episode):
        states = []
        actions = []
        for episode in range(n_episode):
            state = env.reset()[0]
            done = False
            while not done:
                action = agent.take_action(state)
                states.append(state)
                actions.append(action)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
        return np.array(states), np.array(actions)

    env.reset(seed=0)
    torch.manual_seed(0)
    random.seed(0)
    expert_s, expert_a = sample_expert_data(n_episode=1)
    n_samples = int(expert_s.shape[0]/2)
    random_index = random.sample(range(expert_s.shape[0]), n_samples)
    expert_s = expert_s[random_index]
    expert_a = expert_a[random_index]

    np.savetxt("expert_s", expert_s)
    np.savetxt("expert_a", expert_a)



def continuous():
    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                          lmbda, epochs, eps, gamma, device, dist="continuous")

    return_list = utils.train_on_policy_agent(env, agent, num_episodes)

    plot(return_list, env_name)

def plot(return_list, env_name):
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = utils.moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

if __name__ == '__main__':
    discrete()