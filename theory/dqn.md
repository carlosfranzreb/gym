# Deep Q-learning Network (DQN)

DQNs learn Q-values by using neural networks. They address four issues, which are presented here.

## Solving four issues of games

### Epsilon-greedy method

To address the exploration vs. exploitation dilemma, DQNs implement the *epsilon-greedy method*. Epsilon determines the probability of a random action being chosen. It is gradually decreased throughout the training, so the agent explores the environment at the beginning and then sticks to the policy once it improves.

### Stochastic Gradient Descent

DQNs use *Stochastic Gradient Descent (SGD)* to train the network. One of the fundamental requirements of SGDs is that the training data is identically and independently distributed (i.i.d.). This is often not the case in games. Samples in batches are not independent, as they belong to the same episode. Also, the distribution of our training data won't be identical to the data that arises from using the optimal policy. To adress this issue, DQNs implement a *replay buffer*, which is a queue where samples are stored. Every time new samples are added, the oldest ones are thrown away. This method increases the independence of the data.

### Target Networks

Samples will also be highly correlated, as they are only seperated by one time step. Networks are unable to distinguish between them. Therefore, when optimizing for a certain Q-value, we can indirectly alter the value produced for the preceding Q-value. This makes training unstable: optimizing one of them worsens the other. This issue is solved by using the *target network* trick, where a copy of the network is frozen and used in the Bellman equation, which update the original network. The copy, i.e. the target network, is updated every 1k-10k steps.

### Stacking Frames

Games usually don't fufill the Markov property: from a single screenshot of a Pong game it is impossible to know in which direction the ball is moving. Such games are described by partially observable MDPs, which violate the Markov property. They can be turned into MDPs by stacking frames together and using them as the observation in every state.

## DQN Extensions

After the original paper presenting DQNs was published, further work improved their performance.

### N-step DQN

This improvement consists on unrolling the Bellman update. Multiple steps improve the propagation speed of values, which improves convergence.

The N-step DQN is an on-policy method and thus requires fresh data. On-policy methods try to improve the current policy and thus require data that was samples using the current policy. Data sampled by older policies is of no use for such methods.

The simple DQN is off-policy and thus can use old data. That's because it updates the current Q-value with immediate reward plus the discounted current approximation of the best action's value. This allows for larger experience buffers, which bring the data closer to i.i.d. However, they usually converge slower than on-policy methods.

With that being said, n-step DQNs usually speed up convergence for small step values (2 or 3).

### Double DQN

Researchers at DeepMind demonstrated that the basic DQN tends to overestimate Q-values because of the max operation in the Bellman equation. To address this issue, the authors proposed choosing actions for the next state using the trained network but taking values of Q from the target network.

### Noisy DQN

Adding noise to the weights of the network allows the network to learn exploration characteristics. Simple DQNs achieve exploration with the epsilon-greedy method, which requires tuning. This method inserts the exploration in the training procedure, allowing the agent to decide how much exploration it requires.

There are two ways of adding the noise:

1. *Independent Gaussian noise*: for every weight in a layer we have a random value drawn from a normal distribution. The parameters of that distribution are part of that layer and are also trained. The output of the noisy layer is computed in the same way as in a linear layer.
2. *Factorized Gaussian noise*: to minimize the amount of random values to be sampled, only two random vectors are kept. One holds the size of the input and the other the size of the output. A random matrix for the layer is created by calculating the outer product of the vectors.

### Prioritized replay buffer

This method improves sample efficiency by sampling according to training loss, instead of uniformly. How much emphasis is placed on samples with low loss is determined by a hyperparameter alpha, which must also be tuned. Large values for alpha put more stress on samples with low training loss.

This introduces bias in the data distribution because we sample some transitions more frequently than others and must be compensated for SGD to work. To do so, the authors multiplied sample losses with sample weights. How much these weights compensate for the sampling bias is tuned with another hyperparameter, beta. It should start between 0 and 1 and gradually increase to 1 during training. When beta equals 1, the bias is fully compensated.

### Dueling DQN

Q-Values can be computed by adding state values *V(s)* and action advantages *A(s,a)*: *Q(s,a) = V(s) + A(s,a)*. An action advantage *A(s,a)* defines how much extra reward action *a* brings in state *s*.

The simple DQN takes features from the convolution layer and transforms them into a vector of Q-values. Dueling DQNs divide this task in two steps. The convolution features are processed independently by two paths, one responsible for value prediction and the other for advantage prediction. After that, the results are added to obtain Q-values.

For this to work we need to constrain the mean value of all advantages to be zero. This is introduced in the loss function by subtracting the mean from the Q-value equation.

### Categorical DQN

This network learns probability distributions for the Q-values instead of estimating the value directly. Rewards and Q-values are replaced by their distributions in the Bellman update. The loss function is changed to the KL divergence.

The distributions are represented with bins, where each bin accounts for a certain area of the probability range. The network predicts the probability that future discounted value wil fall into each bin's range.
