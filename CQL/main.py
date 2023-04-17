import numpy as np
import gym
from tqdm import tqdm
import random
import utils
import torch
import matplotlib.pyplot as plt
from agent import CQL, SACContinuous

if __name__ == '__main__':
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    replay_buffer = utils.ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device)

    return_list = utils.train_off_policy_agent(env, agent, num_episodes,
                                               replay_buffer, minimal_size,
                                               batch_size)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()



    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)
    beta = 5.0
    num_random = 5
    num_epochs = 100
    num_trains_per_epoch = 500

    agent = CQL(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                critic_lr, alpha_lr, target_entropy, tau, gamma, device, beta,
                num_random)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_epochs / 10), desc='Iteration %d' % i) as pbar:
            for i_epoch in range(int(num_epochs / 10)):
                # 此处与环境交互只是为了评估策略,最后作图用,不会用于训练
                epoch_return = 0
                state = env.reset()[0]
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    state = next_state
                    epoch_return += reward
                return_list.append(epoch_return)

                for _ in range(num_trains_per_epoch):
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

                if (i_epoch + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epoch':
                        '%d' % (num_epochs / 10 * i + i_epoch + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    epochs_list = list(range(len(return_list)))
    plt.plot(epochs_list, return_list)
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.title('CQL on {}'.format(env_name))
    plt.show()

    mv_return = utils.moving_average(return_list, 9)
    plt.plot(epochs_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('CQL on {}'.format(env_name))
    plt.show()