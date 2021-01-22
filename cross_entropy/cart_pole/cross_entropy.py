""" Implements the cross entropy method for the cart pole problem.

The CE method is model-free, policy-based and on-policy.
Steps: 
  1. Play N number of episodes using our current model and environment.
  2. Calculate the total reward for every episode and decide on a reward
    boundary. Usually, we use some percentile of all rewards, such as
    50th or 70th.
  3. Throw away all episodes with a reward below the boundary.
  4. Train on the remaining "elite" episodes using observations as the input
    and issued actions as the desired output.
  5. Repeat from step 1 until we become satisfied with the result.
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
    observation, _, is_done, _ = env.step(action)
    if not is_done:
      observations.append(observation)
    else:
      return observations, actions


def collect_episodes(env, agent, num, pctg):
  """ Collect num episodes of the agent and return the best pctg% of them. """
  obs, acts, rewards, best_obs, best_acts = [], [], [], [], []
  for _ in range(num):
    observation, action = run(env, agent)
    obs.append(observation)
    acts.append(action)
    rewards.append(len(action))
  percentile = np.percentile(rewards, pctg)
  for i in range(len(rewards)):
    if rewards[i] >= percentile:
      best_obs += obs[i]
      best_acts += acts[i]  
  return torch.tensor(best_obs, dtype=torch.float), \
    torch.tensor(best_acts, dtype=torch.long)


def train(env, batch_len, batch_cnt, epochs, agent, optimizer, loss_fn, pctl):
  validations = list()
  for epoch in range(epochs):
    for _ in range(batch_cnt):
      observations, actions = collect_episodes(env, agent, batch_len, pctl)
      out = agent(observations)
      loss = loss_fn(out, actions)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    times = [len(run(env, agent)[1]) for _ in range(batch_len)]
    avg_time = sum(times) / len(times)
    validations.append((agent, avg_time))
    print(f"Agent survived {round(avg_time, 2)} s in epoch {epoch}")
  return validations, agent


def normalize(tensor):
  arr = tensor.tolist()
  total = sum(arr)
  return [elem / total for elem in arr]


def train_cartpole(f):
  """ Train an agent in cart pole and save it in f. """
  env = gym.make('CartPole-v0')
  n_observations = env.observation_space.shape[0]
  n_actions = env.action_space.n
  agent = Net(n_observations, 16, n_actions)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)
  train(env, 16, 30, 7, agent, optimizer, loss_fn, 70)
  torch.save(agent.state_dict(), f)


if __name__ == "__main__":
  train_cartpole('second_agent.pt')

