# Q-Learning

## Deep Q-learning Network (DQN)

DQNs learn Q-values by using neural networks. They address four issues, which are presented here.

To address the exploration vs. exploitation dilemma, DQNs implement the *epsilon-greedy method*. Epsilon determines the probability of a random action being chosen. It is gradually decreased throughout the training, so the agent explores the environment at the beginning and then sticks to the policy once it improves.

DQNs use *Stochastic Gradient Descent (SGD)* to train the network. One of the fundamental requirements of SGDs is that the training data is identically and independently distributed (i.i.d.). This is often not the case in games. Samples in batches are not independent, as they belong to the same episode. Also, the distribution of our training data won't be identical to the data that arises from using the optimal policy. To adress this issue, DQNs implement a *replay buffer*, which is a queue where samples are stored. Every time new samples are added, the oldest ones are thrown away. This method increases the independence of the data.

Samples will also be highly correlated, as they are only seperated by one time step. Networks are unable to distinguish between them. Therefore, when optimizing for a certain Q-value, we can indirectly alter the value produced for the preceding Q-value. This makes training unstable: optimizing one of them worsens the other. This issue is solved by using the *target network* trick, where a copy of the network is frozen and used in the Bellman equation, which update the original network. The copy, i.e. the target network, is updated every 1k-10k steps.

Games usually don't fufill the Markov property: from a single screenshot of a Pong game it is impossible to know in which direction the ball is moving. Such games are described by partially observable MDPs, which violate the Markov property. They can be turned into MDPs by stacking frames together and using them as the observation in every state.