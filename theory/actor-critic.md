# Actor-Critic Method

Extension of the vanilly Policy Gradient method that improves the stability and the convergence speed.

## Variance Reduction

PG methods increase the probability of good actions and decrease the probability of bad actions. The Q-values determine the scale of the increase or decrease.

To increase the stability of REINFORCE, we subtracted the mean reward from the gradient scale. This makes Q-values below the avg. negative, so the agent tries to avoid those actions. If all the Q-values were positive, the agent would push the policy towards them even if some actions are well below the avg. The same would happen in the opposite case: if all Q-values were negative, the agent would try to avoid all the actions, even if some of them are better than the rest.

## Advantage Actor-Critic (A2C)

The actor critic-method consists of making the baseline state-dependent. If we know the value of a state, we can use it to calculate the PG and update our policy network to increase probabilities for actions with good advantage values and decrease those with bad advantage values. Remember that Q-value can be expressed as the value of the state plus the advantage of executing the action on that state. This was explained when introducing the dueling DQNs.

To do so, the actor-critic method introduces a second network, which is in charge of approximating the values of the states. Similarly to DQNs, we minimize the mean square error after computing the Bellman update to improve the value approximation. This network is called the critic, whereas the policy network is called the actor. The actor tells us what to do and the critic allows us to understand how good our actions were.

### Training steps

1. Initialize network parameters with random values.
2. Play N steps in the environment using the current policy, saving state, action and reward.
3. If the end of the episode is reached, R = 0.
4. Else for each performed time step, starting with the last one:

   1. R = gamma \* R + (this time-step's reward).
   2. Accumulate the PG.
   3. Accumulate the value gradients.

5. Update network parameters using the accumulated gradients, moving in the direction of PG and in the opposite direction of the value gradients.
6. Repeat from step 2 until convergence is reached.

### Considerations for training

1. Entropy bonus is usually added to improve exploration.
2. Gradients accumulation is usually implemented as a loss function combining all three components: policy loss, value loss, and entropy loss.
3. To improve stability, it's worth using several environments, providing you with observations concurrently.

## Asynchronous Advantage Actor-Critic (A3C)

The stability of the PG methods can be improved by using multiple environments in parallel, because this reduces the correlation between samples and allows for the i.i.d data assumption of SGD optimization. This method is still sample-inefficient, as all the experience is thrown away after each training step.

There are two approaches to actor-critic parallelization:

1. **Data parallelism**: we have several processes, each of them communicating with one or more environments. The samples are gathered in one process, where the model. Its updated parameters are broadcasted to all other processes.

2. **Gradients parallelism**: we have several processes calculating gradients on their own training samples. Then, the gradients are summed together to perform the SGD update in one process.

The second method is computationally better, as computing the gradients is expensive. The SGD optimization step is quite cheap in comparison. Parallelizing the computation of the gradients eliminates thus eliminates the major bottleneck. On the other hand, parallelizing the data is easier to implement. If only one GPU is available, both methods would perform similarly and thus make the first method better suited.
