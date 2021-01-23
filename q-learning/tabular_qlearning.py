""" Tabular Q-Learning.
1. Start with an empty values table.
2. Obtain a tuple (source, action, reward, target) from the env.
3. Make a Bellman update of Q(s, a) with learning rate alpha.
4. Check convergence conditions. If not met, go back to step 2.
"""


import gym
import collections
from copy import deepcopy


class Agent:
  def __init__(self, env_name, gamma, alpha):
    self.env, self.test_env = gym.make(env_name), gym.make(env_name)
    self.state = self.env.reset()
    self.values = collections.defaultdict(float)
    self.gamma = gamma
    self.alpha = alpha

  def sample_env(self):
    """ Perform a random action in the environment and return a tuple
    (source, action, reward, target). """
    action = self.env.action_space.sample()
    source = self.state
    self.state, reward, is_done, _ = self.env.step(action)
    if is_done:
      self.state = self.env.reset()
    return source, action, reward, self.state

  def select_action(self, state):
    """ Return a tuple (value, action). The value should be the largest value
    for an action with source 'state'."""
    best_action, best_value = None, -1
    for action in range(self.env.action_space.n):
      if self.values[(state, action)] > best_value:
        best_action = action
        best_value = self.values[(state, action)]
    return best_value, best_action

  def update_value(self, source, action, reward, target):
    """ Update the value of Q(source, action). """
    best_value, _ = self.select_action(target)
    new = (reward + self.gamma * best_value)
    old = self.values[(source, action)]
    self.values[(source, action)] = old * (1-self.alpha) + new * self.alpha

  def run(self):
    """ Play an episode without updating the Q-values. """
    total_reward = 0
    state = self.test_env.reset()
    is_done = False
    while not is_done:
      _, action = self.select_action(state)
      state, reward, is_done, _ = self.test_env.step(action)
      total_reward += reward
    return total_reward


def train(agent, desired_reward, test_cnt):
  i = 0
  avg_reward = -1
  while avg_reward < desired_reward:
    agent.update_value(*agent.sample_env())
    rewards = [agent.run() for _ in range(test_cnt)]
    avg_reward = sum(rewards) / len(rewards)
    print(f'{i}: {avg_reward}')
    i += 1


if __name__ == "__main__":
  agent = Agent('FrozenLake-v0', .9, .2)
  train(agent, .9, 20)
