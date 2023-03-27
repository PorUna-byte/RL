import torch
import gym
import matplotlib.pyplot as plt
import utils
from agent import TRPO
if __name__ == '__main__':
    num_episodes = 5000
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    critic_lr = 1e-2
    kl_constraint = 0.0005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'LunarLander'
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device)
    return_list = utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

    mv_return = utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()