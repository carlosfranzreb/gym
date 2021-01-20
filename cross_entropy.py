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
      nn.Linear(4, 1),
      nn.Softmax(dim=0)
    ).double()

  def forward(self, observations):
    tensor = torch.from_numpy(observations).double()
    return self.pipe(tensor)


def run(agent):
  """ Simulate a game with the agent and return the time he survived. """
  env = gym.make('CartPole-v0')
  env.reset()
  is_done = False
  time = 0
  outputs = env.step(env.action_space.sample())
  while not is_done:
    time += 1
    observations, _, is_done, _ = outputs
    out = agent(observations)
    action = np.random.choice([0, 1], 1, p=[1-out, out])
    outputs = env.step(action[0])
  return time


def collect_outputs(num, agent):
  """ Collect num outputs from the cart pole env while using the agent
  for action selection. """
  env = gym.make('CartPole-v0')
  env.reset()
  observations = env.step(env.action_space.sample())[0]
  outputs = [[observations]]
  for _ in range(num):
    out = agent(outputs[-1][0])
    action = np.random.choice([0, 1], 1, p=[1-out, out])
    observations, _, is_done, _ = env.step(action[0])
    outputs[-1].append(
      torch.tensor([is_done]).double()
    )  # reward for previous time step
    outputs.append([observations])  # new set of observations
    if is_done:
      env.reset()
  env.close()
  return outputs[:-1]  # remove observation without reward


def train(batch_len, batch_cnt, epochs, agent, optimizer, loss_fn):
  validations = list()
  for epoch in range(epochs):
    for _ in range(batch_cnt):
      outputs = collect_outputs(batch_len, agent)
      for observations, reward in outputs:
        out = agent(observations)
        loss = loss_fn(out, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    validations.append((agent, run(agent)))
    print(f"Agent survived {validations[-1][1]} s in epoch {epoch}")
  return validations


if __name__ == "__main__":
  agent = NN()
  loss_fn = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)
  validations = train(30, 100, 500, agent, optimizer, loss_fn)
  times = [v[1] for v in validations]
  plt.plot(times)