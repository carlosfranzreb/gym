""" Deep Q-learning Network (DQN)
1. Init. the values table randomly and set epsilon to 1.
2. With probability epison select a random action. Else, select one
  according to the values table.
3. Execute the action and add the resulting tuple (source, action,
  reward, target) to the replay buffer.
4. Sample a random batch from the replay buffer. For every transition
  in the buffer compute y. y is the reward if the episode has ended. If not,
  y equals the new value (reward + max. reachable Q-value * gamma).
5. Update Q(s, a) using SGD with loss L = (Q(s, a) - y)**2
6. Every N steps, copy weights from Q to Q_t (the target network).
7. Repeat from step 2 until convergence is achieved.
"""


import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
