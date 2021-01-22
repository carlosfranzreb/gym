""" This example shows us the limitations of the cross-entropy method:
1. For training, our episodes have to be finite and, preferably, short.
2. The total reward for the episodes should have enough variability to
  separate good episodes from bad ones.
3. There is no intermediate indication about whether the agent has
  succeeded or failed.

We can tackle this issues by implementing the following:
1. Increase the batch size: to find a decent amount of succesful episodes.
2. Discount factor: to prioritize shorter episodes.
3. Keep elite episodes: they are harder to find.
4. Decrease learning rate and increase training time.

The non-slippery version, where randomness is excluded from action
selection, is much easier to learn.
"""


import torch
import torch.nn as nn
import gym
import numpy as np
from matplotlib import pyplot as plt


class Net(nn.Module):
  def __init__(self, n_observations, hidden_size, n_actions):
    super(Net, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(n_observations, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, n_actions),
    )

  def forward(self, x):
    return self.net(x)


def run(env, agent):
  """ Simulate a game with the agent and return a list with all 
  observations and actions. """
  observations, actions = [env.reset()], []
  softmax = nn.Softmax(dim=0)
  while True:
    out = softmax(agent(torch.tensor(observations[-1], dtype=torch.float)))
    probabilities = normalize(out)
    action = np.random.choice(out.size(0), 1, p=probabilities)[0]
    actions.append(action)
    observation, reward, is_done, _ = env.step(action)
    if not is_done:
      observations.append(observation)
    else:
      return observations, actions, reward


def collect_episodes(env, agent, num, pctg, gamma, elite_batch):
  """ Collect num episodes of the agent and return the best pctg% of them. """
  obs, acts, rewards, best_obs, best_acts = [], [], [], [], []
  for _ in range(num):
    observation, action, reward = run(env, agent)
    obs.append(observation)
    acts.append(action)
    rewards.append(reward * gamma**len(action))
  for batch in elite_batch:
    obs.append(batch[0])
    acts.append(batch[1])
    rewards.append(batch[2])
  percentile = np.percentile(rewards, pctg)
  for i in range(len(rewards)):
    if rewards[i] >= percentile:
      best_obs += obs[i]
      best_acts += acts[i]
      elite_batch.append((obs[i], acts[i], rewards[i]))
  return torch.tensor(best_obs, dtype=torch.float), \
    torch.tensor(best_acts, dtype=torch.long)


def train(env, batch_len, batch_cnt, epochs, agent, optimizer, loss_fn, pctl, gamma):
  validations, elite_batch = [], []
  for epoch in range(epochs):
    for _ in range(batch_cnt):
      observations, actions = collect_episodes(
        env, agent, batch_len, pctl, gamma, elite_batch
      )
      out = agent(observations)
      loss = loss_fn(out, actions)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if len(elite_batch) > 1000:
        elite_batch = elite_batch[-500:]
    lengths = []
    for _ in range(batch_len):
      _, action, reward = run(env, agent)
      if reward == 1:
        lengths.append(len(action))
    pctg = len(lengths) / batch_len
    avg_len = sum(lengths) / len(lengths) if len(lengths) > 0 else 0
    validations.append((agent, pctg, avg_len))
    print(f"Agent survived {pctg}% of the time taking {avg_len} steps in epoch {epoch}")
  return validations, agent


def normalize(tensor):
  arr = tensor.tolist()
  total = sum(arr)
  return [elem / total for elem in arr]


def train_frozenlake(f):
  """ Train an agent in cart pole and save it in f. """
  env = OHWrapper(gym.make('FrozenLake-v0', is_slippery=False))
  n_observations = env.observation_space.shape[0]
  n_actions = env.action_space.n
  agent = Net(n_observations, 16, n_actions)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
  train(env, 128, 100, 50, agent, optimizer, loss_fn, 70, .95)
  torch.save(agent.state_dict(), f)


class OHWrapper(gym.ObservationWrapper):
  """ Transform the position, which is given as a integer between 0 and 15,
  to a one-hot encoded vector of size 16."""
  def __init__(self, env):
    super(OHWrapper, self).__init__(env)
    self.observation_space = gym.spaces.Box(
      .0, 1., (env.observation_space.n, ), dtype=np.float32
    )

  def observation(self, observation):
      res = np.copy(self.observation_space.low)
      res[observation] = 1.
      return res


if __name__ == "__main__":
  train_frozenlake('agent_frozenlake_nonslippery.pt')

