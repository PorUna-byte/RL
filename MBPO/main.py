import gym
import matplotlib.pyplot as plt
from agent import SAC, EnsembleDynamicsModel, FakeEnv, ReplayBuffer, MBPO

if __name__ == '__main__':
    real_ratio = 0.5
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    num_episodes = 20
    actor_lr = 5e-4
    critic_lr = 5e-3
    alpha_lr = 1e-3
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    target_entropy = -1
    model_alpha = 0.01  # 模型损失函数中的加权权重
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值

    rollout_batch_size = 1000
    rollout_length = 1  # 推演长度k,推荐更多尝试
    model_pool_size = rollout_batch_size * rollout_length

    agent = SAC(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                critic_lr, alpha_lr, target_entropy, tau, gamma)
    model = EnsembleDynamicsModel(state_dim, action_dim, model_alpha)
    fake_env = FakeEnv(model)
    env_pool = ReplayBuffer(buffer_size)
    model_pool = ReplayBuffer(model_pool_size)
    mbpo = MBPO(env, agent, fake_env, env_pool, model_pool, rollout_length,
                rollout_batch_size, real_ratio, num_episodes)

    return_list = mbpo.train()

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('MBPO on {}'.format(env_name))
    plt.show()
