# mlprague-2021
Material for MLPrague 2021 workshop



# Elements of Reinforcement Learning

Beyond the agent and the environment, one can identify four main subelements of a
reinforcement learning system: a policy, a reward signal, a value function, and, optionally,
a model of the environment.

*Policy* - defines the learning agent’s way of behaving at a given time. Roughly speaking,
a policy is a mapping from perceived states of the environment to actions to be taken
when in those states. It corresponds to what in psychology would be called a set of
stimulus–response rules or associations. In some cases the policy may be a simple function
or lookup table, whereas in others it may involve extensive computation such as a search
process. The policy is the core of a reinforcement learning agent in the sense that it alone
is sucient to determine behavior. In general, policies may be stochastic, specifying
probabilities for each action.

*Reward signal* - defines the goal of a reinforcement learning problem. On each time
step, the environment sends to the reinforcement learning agent a single number called
the reward. The agent’s sole objective is to maximize the total reward it receives over
the long run. The reward signal thus defines what are the good and bad events for the
agent.

*Value function* - specifies what is good in the long run. Roughly speaking, the value of a state is
the total amount of reward an agent can expect to accumulate over the future, starting
from that state. Whereas rewards determine the immediate, intrinsic desirability of
environmental states, values indicate the long-term desirability of states after taking into
account the states that are likely to follow and the rewards available in those states. For
example, a state might always yield a low immediate reward but still have a high value
because it is regularly followed by other states that yield high rewards.

## Other subelements for RL terminology:
- Actions - we thought of this as "pulling the bandit arm". Action is what agent does. If you have three advertisements (e.g. iPhone, Huawei, Samsung) so that you can possibly show to a user three possible actions. Actions can also be continuous, e.g. how many degrees to rotate a steering wheel.
- States - an online advertising system: State = [age, gender, day, time]; a temperature controller: State = [temperature, humidity].

```math
x: feature representation of state
w: model parameters
$n^2$y
```

- Environment - e.g. Gridworld, what game we are playing
- Punishment -  
- Rewards - references to psychology; RL has been used to model animal behavior. RL agent's goal is in the future. In contract, a supervised model simply tries to get good accuracy / minimaze cost on current input. RL algorithm get feedback signals (rewards) come from the environment (i.e. the agent experiences them). Supervised models instantly know if it is wrong/right, because inputs + target are provided simultaneously. RL is dynamic - if agent solves a maze, it only knows its decisions were corrent if it eventually solves the maze.
- Returns (sum of future rewards)
- Value (Expected sum of future rewards)
- Learning - Reinforcing desired behavior
- Initial state
- Terminal state
- State space - set of all possible states
- Action space - set of all possible actions

# References
[1] [Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto] (http://incompleteideas.net/book/the-book.html)
