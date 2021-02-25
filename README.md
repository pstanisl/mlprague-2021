# MLPrague 2021 - How to Make Data-Driven Decisions: The Case for Contextual Multi-armed Bandits

> Other names: associative reinforcement learning, associative bandits,learning with partial feedback, bandits with side information

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

![first equation](https://latex.codecogs.com/gif.latex?x%3A) *feature representation of state*  
![second equation](https://latex.codecogs.com/gif.latex?w%3A) *model parameters*  
![third equation](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20%3D%20w%5E%7BT%7Dx) *expected reward*  
![fourth equation](https://latex.codecogs.com/gif.latex?y%3A) *true reward*  

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


## Algorithms

### Epsilon-Greedy

The [Epsilon-Greedy](https://github.com/raffg/multi_armed_bandit/blob/master/algorithms/epsilon_greedy.py) algorithm balances exploitation and exploration fairly basically. It takes a parameter, epsilon, between 0 and 1, as the probability of exploring the options (called arms in multi-armed bandit discussions) as opposed to exploiting the current best variant in the test. For example, say epsilon is set at 0.1. Every time a visitor comes to the website being tested, a number between 0 and 1 is randomly drawn. If that number is greater than 0.1, then that visitor will be shown whichever variant (at first, version A) is performing best. If that random number is less than 0.1, then a random arm out of all available options will be chosen and provided to the visitor. The visitor’s reaction will be recorded (a click or no click, a sale or no sale, etc.) and the success rate of that arm will be updated accordingly.

### Thompson Sampling

This algorithm is fully Bayesian. It generates a vector of expected rewards for each arm from a posterior distribution and then updates the distributions.

### Others

* Softmax (Boltzmann) - trying to solve an obvious flaw in epsilon-Greedy is that it explores completely at random. If we have two arms with very similar rewards, we need to explore a lot to learn which is better and so choose a high epsilon. The algorithm (and its annealed counterpart) attempt to solve this problem by selecting each arm in the explore phase roughly in proportion to the currently expected reward.
* Upper Confidence Bound 1
* Upper Confidence Bound 2
* Exp3

# References

## Blogs

* **[Simple Reinforcement Learning with TensorFlow Part 0: Q-Learning with Tables and Neural](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)**
* [Contextual Bandits and Reinforcement Learning](https://towardsdatascience.com/contextual-bandits-and-reinforcement-learning-6bdfeaece72a)
* [Reinforcement Learning 101](https://towardsdatascience.com/reinforcement-learning-101-e24b50e1d292)
* [A/B testing — Is there a better way? An exploration of multi-armed bandits](https://towardsdatascience.com/a-b-testing-is-there-a-better-way-an-exploration-of-multi-armed-bandits-98ca927b357d)
* [Beyond A/B testing: Multi-armed bandit experiments](https://www.dynamicyield.com/blog/contextual-bandit-optimization/)
* [Deep contextual multi-armed bandits: Deep learning for smarter A/B testing on autopilot](https://product.hubspot.com/blog/deep-contextual-multi-armed-bandits-deep-learning-for-smarter-a/b-testing-on-autopilot)
* [An Introduction to Contextual Bandits](https://getstream.io/blog/introduction-contextual-bandits/)
* [Demystifying Deep Reinforcement Learning](https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/)
* [Deep Reinforcement Learning With TensorFlow 2.1](http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/)
* [Better bandit building: Advanced personalization the easy way with AutoML Tables](https://cloud.google.com/blog/products/ai-machine-learning/how-to-build-better-contextual-bandits-machine-learning-models)
* [How Stitch Fix Optimizes Client Engagement With Contextual Bandits](https://www.topbots.com/stitchfix-client-engagement-contextual-bandits/)
* [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)

## Articles

* **[Introduction to Multi-Armed Bandits](https://arxiv.org/pdf/1904.07272.pdf)**
* **[Neural Contextual Bandits with UCB-based Exploration](https://arxiv.org/pdf/1911.04462.pdf)**
* [NeuralUCB: Contextual Bandits with Neural Network-Based Exploration](https://openreview.net/attachment?id=r1xa9TVFvH&name=original_pdf)
* [Multi-armed bandit experiments in the online service economy](Multi-armed%20bandit%20experiments%20in%20the%20online%20service%20economy)
* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
* [Concrete Dropout](https://arxiv.org/abs/1705.07832)
* [Selecting multiple web adverts: A contextual multi-armed bandit with state uncertainty](https://www.tandfonline.com/doi/full/10.1080/01605682.2018.1546650)
* [Microsoft Multi-word testing whitepaper](https://github.com/Microsoft/mwt-ds/raw/master/images/MWT-WhitePaper.pdf)
* [The Epoch-Greedy Algorithm for ContextualMulti-armed Bandits](https://hunch.net/~jl/projects/interactive/sidebandits/bandit.pdf)
* [AutoML for Contextual Bandits](https://arxiv.org/pdf/1909.03212.pdf)
* [Warm-starting Contextual Bandits:Robustly Combining Supervised and Bandit Feedback](https://arxiv.org/pdf/1901.00301.pdf)
* [A Contextual Bandit Bake-off](https://hal.inria.fr/hal-01708310v2/document)
* [Adapting multi-armed bandits policies to contextualbandits scenarios](https://arxiv.org/pdf/1811.04383.pdf)
* [Taming the Monster: A Fast and Simple Algorithm for Contextual Bandits](http://proceedings.mlr.press/v32/agarwalb14.pdf)
* [Contextual-Bandit: New-Article-Recommendation](https://github.com/appurwar/Contextual-Bandit-News-Article-Recommendation)

## Presentations

* [Learning for Contextual Bandits](https://hunch.net/~exploration_learning/main.pdf)

## Repositories and Gists

* **<https://github.com/tensorflow/agents>**
* <https://github.com/raffg/multi_armed_bandit>
* <https://github.com/awjuliani/oreilly-rl-tutorial>
* <https://gitlab.fit.cvut.cz/podszond/mvi-sp>
* <https://github.com/etiennekintzler/bandits_algorithm>
* <https://gist.github.com/tushuhei/0cef4b9f66956d9ce2076f2ecf6feefd>
* <https://github.com/niffler92/Bandit>
* <https://github.com/tensorflow/agents>

## Others

* **[Data-driven evaluation of Contextual Bandit algorithmsand applications to Dynamic Recommendation](https://tel.archives-ouvertes.fr/tel-01297407/document)**
* [SIGIR 2016 Tutorial on Counterfactual Evaluation and Learning for Search, Recommendation and Ad Placement](http://www.cs.cornell.edu/~adith/CfactSIGIR2016/)
* [Learning through Exploration](http://videolectures.net/kdd2010_beygelzimer_langford_lte/)
* [Reinforcement Learning Book](http://incompleteideas.net/book/RLbook2018.pdf)
* [Bandit Algorithms Book](https://tor-lattimore.com/downloads/book/book.pdf)
* [Introduction to Contextual Bandits](http://alekhagarwal.net/bandits_and_rl/cb_intro.pdf)
* [A Beginner's Guide to Deep Reinforcement Learning](https://pathmind.com/wiki/deep-reinforcement-learning)
