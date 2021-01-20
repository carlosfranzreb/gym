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


class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.pipe = nn.Sequential(
      nn.Linear(4, 16),
      nn.ReLU(),
      nn.Linear(16, 2),
    )

  def forward(self, observation):
    tensor = torch.from_numpy(observation).float()
    return self.pipe(tensor)


def run(agent):
  """ Simulate a game with the agent and return a list with all 
  observations and actions. """
  env = gym.make('CartPole-v0')
  observations, actions = [env.reset()], []
  softmax = nn.Softmax(dim=1)
  while True:
    out = softmax(agent(observations[-1]))
    action = np.random.choice([0, 1], 1, p=out.tolist())[0]
    actions.append(action)
    observation, _, is_done, _ = env.step(action)
    if not is_done:
      observations.append(observation)
    else:
      env.close()
      return observations, actions


def collect_episodes(agent, num, pctg):
  """ Collect num episodes of the agent and return the best pctg% of them. """
  obs, acts, rewards, best_obs, best_acts = [], [], [], [], []
  for _ in range(num):
    observation, action = run(agent)
    obs.append(observation)
    acts.append(action)
    rewards.append(len(action))
  percentile = np.percentile(rewards, pctg)
  for i in range(len(rewards)):
    if rewards[i] >= percentile:
      best_obs += obs[i]
      best_acts += acts[i]  
  return np.array(best_obs), torch.tensor(best_acts, dtype=torch.long)


def train(batch_len, batch_cnt, epochs, agent, optimizer, loss_fn):
  validations = list()
  for epoch in range(epochs):
    for _ in range(batch_cnt):
      observations, actions = collect_episodes(agent, batch_len, 70)
      out = agent(observations)
      loss = loss_fn(out, actions)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(list(agent.parameters()))
    times = [len(run(agent)[1]) for _ in range(batch_len)]
    avg_time = sum(times) / len(times)
    validations.append((agent, avg_time))
    print(f"Agent survived {round(avg_time, 2)} s in epoch {epoch}")
  return validations


if __name__ == "__main__":
  agent = NN()
  loss_fn = torch.nn.NLLLoss()
  optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)
  validations = train(32, 100, 100, agent, optimizer, loss_fn)
  torch.save(agent.paremeters(), 'agent.pt')