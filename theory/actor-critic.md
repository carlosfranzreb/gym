# Actor-Critic Method

Extension of the vanilly Policy Gradient method that improves the stability and the convergence speed.

## Variance Reduction

PG methods increase the probability of good actions and decrease the probability of bad actions. The Q-values determine the scale of the increase or decrease.

To increase the stability of REINFORCE, we subtracted the mean reward from the gradient scale. This makes Q-values below the avg. negative, so the agent tries to avoid those actions. If all the Q-values were positive, the agent would push the policy towards them even if some actions are well below the avg. The same would happen in the opposite case: if all Q-values were negative, the agent would try to avoid all the actions, even if some of them are better than the rest.

## Actor-Critic

The actor critic-method consists of making the baseline state-dependent. If we know the value of a state, we can use it to calculate the PG and update our policy network to increase probabilities for actions with good advantage values and decrease those with bad advantage values. Remember that Q-value can be expressed as the value of the state plus the advantage of executing the action on that state. This was explained when introducing the dueling DQNs.

To do so, the actor-critic method introduces a second network, which is in charge of approximating the values of the states. Similarly to DQNs, we minimize the mean square error after computing the Bellman update to improve the value approximation. This network is called the critic, whereas the policy network is called the actor. The actor tells us what to do and the critic allows us to understand how good our actions were.

### Training steps

1. Initialize network parameters with random values.
2. Play N steps in the environment using the current policy, saving state, action and reward.
3. If the end of the episode is reached, R = 0.
4. Else for each performed time step, starting with the last one:
    a. R = gamma * R + (this time-step's reward).
    b. Accumulate the PG.
    c. Accumulate the value gradients.
5. Update network parameters using the accumulated gradients, moving in the direction of PG and in the opposite direction of the value gradients.
6. Repeat from step 2 until convergence is reached.

### Considerations for training

1. Entropy bonus is usually added to improve exploration.
2. Gradients accumulation is usually implemented as a loss function combining all three components: policy loss, value loss, and entropy loss.
3. To improve stability, it's worth using several environments, providing you with observations concurrently.
