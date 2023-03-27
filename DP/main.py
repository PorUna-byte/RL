import agent as agent_pkg
import env as env_pkg
import util as util_pkg
def policy_iteration():
    env = env_pkg.CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = agent_pkg.PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    util_pkg.print_agent(agent, action_meaning, list(range(37, 47)), [47])

def value_iteration():
    env = env_pkg.CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = agent_pkg.ValueIteration(env, theta, gamma)
    agent.value_iteration()
    util_pkg.print_agent(agent, action_meaning, list(range(37, 47)), [47])
if __name__ == '__main__':
    value_iteration()
