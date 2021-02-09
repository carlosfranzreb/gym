# Policy Gradients

PG methods learn directly the optimal policy instead of inferring it from the values. PG is a loss function that multiplies the negative Q-value of the state-action pair with the probability score of taking that action given the current state. Thus, the direction of the gradient is defined in terms of the accumulated total reward. Its scale is proportional to the Q-value.

PG aims to increase the the probability of actions that have a large reward and decrease the probability of actions with low reward.

## Why policy?

1. Policy is what we are looking for. Even when computing values, the ultimate goal is to extract the optimal policy for the environment. PG methods avoid the extra work.
2. Better than value methods for environments with a continuous action space. Calculating values for continuous actions is a hard optimization problem, as the value function is usually highly non-linear.
3. Stochasticity. Policy is naturally represented as the probability of actions.

## Policy representation

Policy is represented as a probability distribution over the set of possible actions. This representation encodes exploration into the agent's behavior. It is also a smooth representation, meaning that changing the network weights slightly translates into slightly different probability scores instead of a choosing a new action (for discrete representations).

## REINFORCE method

Similar to the Cross-Entropy method, but looking at single transitions and their values instead of full episodes. This

1. increases the contribution of episodes with larger rewards,
2. increases the probability of good actions in the beginning of the episode and
3. decreases the probability of actions closer to the end of the episode.

The REINFORCE method consists of **six steps**.

1. Initialize the network with random weights.
2. Play N full episodes.
3. For every step of every episode calculate the discounted total reward for subsequent steps.
    * This is done in reverse order. The last step's total reward is just its local reward, the previous step has that reward as its discounted reward, etc.
4. Calculate the loss function for all transitions.
5. Perform SGD update.
6. Repeat from step 2 until convergence.

This method differs from Q-learning in three major ways.

1. **No explicit exploration is needed**, as it is already encoded in the probability distribution.
2. **No replay buffer is used**. PG methods are on-policy and therefore they can't train from data obtained with the old policy.
    * PG methods usually converge faster,
    * but also require much more interaction with the environment than off-policy methods.
3. **No target network is needed**. Q-values are not approximated, as the whole episode is played before calculating them.

## Policy-based vs. value-based methods

1. Policy methods **directly optimize** what is needed, instead of doing so indirectly through values.
2. Policy methods are **on-policy**. They require fresh samples from the environment. Value-methods can benefit from data obtained from older policies or human demonstration.
3. Policy methods are usually **less sample-efficient**. They require more interaction with the environment. On the other hand, value methods can benefit from large replay buffers.
    * This doesn't mean that value methods are computationally more efficient. Very often it's the opposite. For example, DQNs need to access the network twice every step: once for the current state and another for the next state in the Bellman update.

## Issues of the REINFORCE method

1. **Full episodes are required**
    * Reason for this is to compute the Q-values starting from the episode's last step instead of approximating them through the Bellman equation.
    * Problem is that some games have episodes that can last thousands of time steps.
    * Solution: replace the exact value for a discounted reward. This can be done by either estimating the state values or by unrolling the Bellman equation N steps ahead, with N depending on gamma.
2. **High gradients variance**
    * The gradient is proportional to the discounted reward from the given state.
    * Episodes with large rewards will dominate training, even if they are just a product of luck. Episodes with low rewards will be ignored, although they are useful for lowering the probabilities of the actions that led to those situations (negative examples).
    * Solution: subtract a baseline value from the Q-values. It can be the mean or the moving avg. of the discounted rewards or the value of the state.
3. **Exploration**
    * Even with the policy represented as a probability distribution, the agent will often converge to some locally optimal policy.
    * Solution: entropy shows how uncertain the agent is about which action to take. We can subtract the entropy of the policy from the loss function, punishing the agent for being too certain about the action to take.
4. **Correlation between samples**
    * Training samples in one episode are usually heavily correlated, which is bad for SGD training as it assumes i.i.d. data.
    * Solution: use parallel environments.
