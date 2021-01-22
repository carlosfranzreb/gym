"""
1. Play 100 games to populate the reward and transition tables.
  a. Reward table: (source, action, target) -> reward.
  b. Transition table: (source, action) -> count(target) for all targets.
2. Perform a value iteration loop to update the value table.
  a. Value table: state -> value.
  b. Value iteration loop (for every state): calculate the values for all 
    reachable states and update the value with the max. one.
3. Play several episodes. If the desired avg. reward was reached, stop. 
  Else, go back to step 1.

Value iteration loop:
1. 
"""


import collections
import gym
import json


class Agent:
  def __init__(self, env, gamma):
    self.env = env
    self.values = [0 for _ in range(env.observation_space.n)]
    self.rewards = {}
    self.transitions = collections.defaultdict(collections.Counter)
    self.gamma = gamma  # discount factor

  def update_tables(self, n_steps):
    """ Play num random steps to populate the reward and transition tables.
    Then update the values. """
    source = self.env.reset()
    for _ in range(n_steps):
      action = self.env.action_space.sample()
      target, reward, is_done, _ = self.env.step(action)
      self.rewards[(source, action, target)] = reward
      self.transitions[(source, action)][target] += 1
      source = target if not is_done else self.env.reset()

  def value_iteration(self):
    """ For each state s calculate the value of all states reachable from
    s. The new value of s is the max. of those values. """
    for state in range(self.env.observation_space.n):
      state_values = [self.calculate_value(state, action)
        for action in range(self.env.action_space.n)]
      self.values[state] = max(state_values)
    
  def calculate_value(self, source, action):
    """ Return the value for the given state and action by calculating
    the probabilities of reaching each state and multiplying it with 
    the reward and the value of that state. """
    value = 0
    total = sum(self.transitions[(source, action)].values())
    for target, count in self.transitions[(source, action)].items():
      p = count / total
      target_value = self.gamma * self.values[target]
      value += p * (self.rewards[(source, action, target)] + target_value)
    return value

  def run(self):
    """ Run an episode. """
    total_reward = 0
    source = self.env.reset()
    is_done = False
    while not is_done:
      action = self.select_action(source)
      target, reward, is_done, _ = self.env.step(action)
      self.rewards[(source, action, target)] = reward
      self.transitions[(source, action)][target] += 1
      total_reward += reward
      source = target
    return total_reward

  def select_action(self, state):
    """ Return the optimal action w.r.t to the current values. """
    best_action, best_value = None, -1
    for action in range(self.env.action_space.n):
      action_value = self.calculate_value(state, action)
      if action_value > best_value:
        best_value = action_value
        best_action = action
    return best_action


def train(agent, desired_reward):
  avg_reward = -1
  i = 0
  while avg_reward < desired_reward:
    agent.update_tables(100)
    agent.value_iteration()
    rewards = [agent.run() for _ in range(30)]
    avg_reward = sum(rewards) / len(rewards)
    print(f"{i}: {avg_reward}")
    i += 1
  return agent.values


def show_game(agent):
  """ Play one game and render it. """
  is_done = False
  state = agent.env.reset()
  while not is_done:
    agent.env.render()
    state, _, is_done, _ = agent.env.step(agent.select_action(state))


if __name__ == "__main__":
  env = gym.make('FrozenLake-v0')
  agent = Agent(env, .9)
  values = train(agent, .9)
  # json.dump(values, open('values.json', 'w'))
  show_game(agent)

