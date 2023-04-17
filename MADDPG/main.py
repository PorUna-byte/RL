import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from utils import make_env
from agent import MADDPG
import sys

if __name__ == '__main__':
    sys.path.append("multiagen_particle_envs")
    num_episodes = 5000
    episode_length = 25  # 每条序列的最大长度
    buffer_size = 100000
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 100
    minimal_size = 4000

    env_id = "simple_adversary"
    env = make_env(env_id)
    replay_buffer = utils.ReplayBuffer(buffer_size)

    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)

    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau)


    def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
        # 对学习的策略进行评估,此时不会进行探索
        env = make_env(env_id)
        returns = np.zeros(len(env.agents))
        for _ in range(n_episode):
            obs = env.reset()
            for t_i in range(episode_length):
                actions = maddpg.take_action(obs, explore=False)
                obs, rew, done, info = env.step(actions)
                rew = np.array(rew)
                returns += rew / n_episode
        return returns.tolist()


    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()
        # ep_returns = np.zeros(len(env.agents))
        for e_i in range(episode_length):
            actions = maddpg.take_action(state, explore=True)
            next_state, reward, done, _ = env.step(actions)
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            total_step += 1
            if replay_buffer.size(
            ) >= minimal_size and total_step % update_interval == 0:
                sample = replay_buffer.sample(batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                  for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(device)
                        for aa in rearranged
                    ]

                sample = [stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode + 1}, {ep_returns}")

    return_array = np.array(return_list)
    for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
        plt.figure()
        plt.plot(
            np.arange(return_array.shape[0]) * 100,
            utils.moving_average(return_array[:, i], 9))
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(f"{agent_name} by MADDPG")