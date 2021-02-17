# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.bandits.agents import linear_bandit_agent as lin_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.policies import linalg  # NOTE: Maybe unnecessary
from tf_agents.bandits.policies import linear_bandit_policy
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import tf_policy
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from typing import Optional, Sequence, Text

class LinearBanditAgent(tf_agent.TFAgent):
  """An agent implementing the Linear UCB bandit algorithm.
  Reference:
  "A Contextual Bandit Approach to Personalized News Article Recommendation",
  Lihong Li, Wei Chu, John Langford, Robert Schapire, WWW 2010.
  """

  def __init__(self,
               time_step_spec: types.TimeStep,
               action_spec: types.BoundedTensorSpec,
               policy: tf_policy.TFPolicy,
               variable_collection: tf.Module,
               alpha: float = 1.0,
               gamma: float = 1.0,
               use_eigendecomp: bool = False,
               tikhonov_weight: float = 1.0,
               add_bias: bool = False,
               emit_policy_info: Sequence[Text] = (),
               emit_log_probability: bool = False,
               debug_summaries: bool = False,
               summarize_grads_and_vars: bool = False,
               enable_summaries: bool = True,
               dtype: tf.DType = tf.float32,
               name: Optional[Text] = None):
    """Initialize an instance of `LinearUCBAgent`.
    Args:
      time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
      action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
        describing the number of actions for this agent.
      alpha: (float) positive scalar. This is the exploration parameter that
        multiplies the confidence intervals.
      gamma: a float forgetting factor in [0.0, 1.0]. When set to
        1.0, the algorithm does not forget.
      use_eigendecomp: whether to use eigen-decomposition or not. The default
        solver is Conjugate Gradient.
      tikhonov_weight: (float) tikhonov regularization term.
      add_bias: If true, a bias term will be added to the linear reward
        estimation.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      emit_log_probability: Whether the LinearUCBPolicy emits log-probabilities
        or not. Since the policy is deterministic, the probability is just 1.
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      dtype: The type of the parameters stored and updated by the agent. Should
        be one of `tf.float32` and `tf.float64`. Defaults to `tf.float32`.
      name: a name for this instance of `LinearUCBAgent`.
    Raises:
      ValueError if dtype is not one of `tf.float32` or `tf.float64`.
    """
    tf.Module.__init__(self, name=name)
    common.tf_agents_gauge.get_cell('TFABandit').set(True)
    self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
        action_spec)
    self._num_models = self._num_actions
    self._observation_and_action_constraint_splitter = None
    self._time_step_spec = time_step_spec
    self._add_bias = add_bias
    context_spec = time_step_spec.observation

    (self._global_context_dim,
     self._arm_context_dim) = bandit_spec_utils.get_context_dims_from_spec(
         context_spec, False)
    if self._add_bias:
      # The bias is added via a constant 1 feature.
      self._global_context_dim += 1
    self._overall_context_dim = self._global_context_dim + self._arm_context_dim

    self._alpha = alpha
    # variable_collection = lin_agent.LinearBanditVariableCollection(
    #    context_dim=self._overall_context_dim,
    #    num_models=self._num_models,
    #    use_eigendecomp=use_eigendecomp,
    #    dtype=dtype)

    self._variable_collection = variable_collection
    # print("inside: _variable_collection", self._variable_collection)
    self._cov_matrix_list = variable_collection.cov_matrix_list
    # print("inside: _cov_matrix_list", self._cov_matrix_list)
    self._data_vector_list = variable_collection.data_vector_list
    # print("inside: _data_vector_list", self._data_vector_list)
    self._eig_matrix_list = variable_collection.eig_matrix_list
    # print("inside: _use_eigendecomp", self._use_eigendecomp)
    # print("inside: _eig_matrix_list", self._eig_matrix_list)
    self._eig_vals_list = variable_collection.eig_vals_list
    # print("inside: _eig_vals_list", self._eig_vals_list)
    # We keep track of the number of samples per arm.
    self._num_samples_list = variable_collection.num_samples_list
    self._gamma = gamma
    if self._gamma < 0.0 or self._gamma > 1.0:
      raise ValueError('Forgetting factor `gamma` must be in [0.0, 1.0].')
    self._dtype = dtype
    if dtype not in (tf.float32, tf.float64):
      raise ValueError(
          'Agent dtype should be either `tf.float32 or `tf.float64`.')
    self._use_eigendecomp = use_eigendecomp
    self._tikhonov_weight = tikhonov_weight

    # policy = linear_bandit_policy.LinearBanditPolicy(
    #     action_spec=action_spec,
    #     cov_matrix=self._cov_matrix_list,
    #     data_vector=self._data_vector_list,
    #     num_samples=self._num_samples_list,
    #     time_step_spec=time_step_spec,
    #     alpha=alpha,
    #     eig_vals=self._eig_vals_list if self._use_eigendecomp else (),
    #     eig_matrix=self._eig_matrix_list if self._use_eigendecomp else (),
    #     tikhonov_weight=self._tikhonov_weight,
    #     add_bias=add_bias,
    #     emit_policy_info=emit_policy_info,
    #     emit_log_probability=emit_log_probability)

    training_data_spec = None

    super(LinearBanditAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy=policy,
        collect_policy=policy,
        training_data_spec=training_data_spec,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        train_sequence_length=None,
        validate_args=False)

    self._as_trajectory = data_converter.AsTrajectory(
        self.data_context, sequence_length=None)

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def cov_matrix(self):
    return self._cov_matrix_list

  @property
  def eig_matrix(self):
    return self._eig_matrix_list

  @property
  def eig_vals(self):
    return self._eig_vals_list

  @property
  def data_vector(self):
    return self._data_vector_list

  @property
  def num_samples(self):
    return self._num_samples_list

  @property
  def alpha(self):
    return self._alpha

  def update_alpha(self, alpha):
    return tf.compat.v1.assign(self._alpha, alpha)

  @property
  def theta(self):
    """Returns the matrix of per-arm feature weights.
    The returned matrix has shape (num_actions, context_dim).
    It's equivalent to a stacking of theta vectors from the paper.
    """
    thetas = []
    for k in range(self._num_models):
      thetas.append(
          tf.squeeze(
              linalg.conjugate_gradient_solve(
                  self._cov_matrix_list[k] + self._tikhonov_weight *
                  tf.eye(self._overall_context_dim, dtype=self._dtype),
                  tf.expand_dims(self._data_vector_list[k], axis=-1)),
              axis=-1))

    return tf.stack(thetas, axis=0)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  def compute_summaries(self, loss: types.Tensor):
    if self.summaries_enabled:
      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='loss', data=loss, step=self.train_step_counter)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self.policy.variables():
            var_name = var.name.replace(':', '_')
            tf.compat.v2.summary.histogram(
                name=var_name,
                data=var,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name=var_name + '_value_norm',
                data=tf.linalg.global_norm([var]),
                step=self.train_step_counter)
        if self._add_bias:
          thetas = self.theta
          biases = thetas[:, self._global_context_dim - 1]
          bias_list = tf.unstack(biases, axis=0)
          for i in range(self._num_actions):
            tf.compat.v2.summary.scalar(
                name='bias/action_' + str(i),
                data=bias_list[i],
                step=self.train_step_counter)

  def _process_experience(self, experience):
    """Given an experience, returns reward, action, observation, and batch size."""
    return self._process_experience_global(experience)

  def _process_experience_global(self, experience):
    """Processes the experience in case the agent accepts only global features.
    In the experience coming from the replay buffer, the reward (and all other
    elements) have two batch dimensions `batch_size` and `time_steps`, where
    `time_steps` is the number of driver steps executed in each training loop.
    We flatten the tensors in order to reflect the effective batch size. Then,
    all the necessary processing on the observation is done, including splitting
    the action mask if it is present.
    Args:
      experience: An instance of trajectory. Every element in the trajectory has
      two batch dimensions.
    Returns:
      A tuple of reward, action, observation, and batch_size. All the outputs
        (except `batch_size`) have a single batch dimension of value
        `batch_size`.
    """
    reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.reward, self._time_step_spec.reward)
    observation, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.observation, self.training_data_spec.observation)
    action, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.action, self._action_spec)
    batch_size = tf.cast(
        tf.compat.dimension_value(tf.shape(reward)[0]), dtype=tf.int64)

    if self._observation_and_action_constraint_splitter is not None:
      observation, _ = self._observation_and_action_constraint_splitter(
          observation)
    if self._add_bias:
      # The bias is added via a constant 1 feature.
      observation = tf.concat(
          [observation,
           tf.ones([batch_size, 1], dtype=observation.dtype)],
          axis=1)

    observation = tf.reshape(
        tf.cast(observation, self._dtype), [batch_size, -1])
    reward = tf.cast(reward, self._dtype)

    return reward, action, observation, batch_size

  def _distributed_train_step(self, experience, weights=None):
    """Distributed train fn to be passed as input to run()."""
    del weights  # unused
    reward, action, observation, batch_size = self._process_experience(
        experience)
    self._train_step_counter.assign_add(batch_size)

    for k in range(self._num_models):
      diag_mask = tf.linalg.tensor_diag(
          tf.cast(tf.equal(action, k), self._dtype))
      observations_for_arm = tf.matmul(diag_mask, observation)
      rewards_for_arm = tf.matmul(diag_mask, tf.reshape(reward, [-1, 1]))

      # Compute local updates for the matrix A and b of this arm.
      cov_matrix_local_udpate = tf.matmul(
          observations_for_arm, observations_for_arm, transpose_a=True)
      data_vector_local_update = bandit_utils.sum_reward_weighted_observations(
          rewards_for_arm, observations_for_arm)

      def _merge_fn(strategy, per_replica_cov_matrix_update,
                    per_replica_data_vector_update):
        """Merge the per-replica-updates."""
        # Reduce the per-replica-updates using SUM.
        # pylint: disable=cell-var-from-loop
        updates_and_vars = [
            (per_replica_cov_matrix_update, self._cov_matrix_list[k]),
            (per_replica_data_vector_update, self._data_vector_list[k])
        ]

        reduced_updates = strategy.extended.batch_reduce_to(
            tf.distribute.ReduceOp.SUM, updates_and_vars)

        # Update the model variables.
        self._cov_matrix_list[k].assign_add(reduced_updates[0])
        self._data_vector_list[k].assign_add(reduced_updates[1])

        # Compute the eigendecomposition, if needed.
        if self._use_eigendecomp:
          eig_vals, eig_matrix = tf.linalg.eigh(self._cov_matrix_list[k])
          self._eig_vals_list[k].assign(eig_vals)
          self._eig_matrix_list[k].assign(eig_matrix)

      # Passes the local_updates to the _merge_fn() above that performs custom
      # computation on the per-replica values.
      # All replicas pause their execution until merge_call() is done and then,
      # execution is resumed.
      replica_context = tf.distribute.get_replica_context()
      replica_context.merge_call(
          _merge_fn,
          args=(cov_matrix_local_udpate, data_vector_local_update))

    loss = -1. * tf.reduce_sum(reward)
    return tf_agent.LossInfo(loss=(loss), extra=())

  def _train(self, experience, weights=None):
    """Updates the policy based on the data in `experience`.
    Note that `experience` should only contain data points that this agent has
    not previously seen. If `experience` comes from a replay buffer, this buffer
    should be cleared between each call to `train`.
    Args:
      experience: A batch of experience data in the form of a `Trajectory`.
      weights: Unused.
    Returns:
        A `LossInfo` containing the loss *before* the training step is taken.
        In most cases, if `weights` is provided, the entries of this tuple will
        have been calculated with the weights.  Note that each Agent chooses
        its own method of applying weights.
    """
    experience = self._as_trajectory(experience)

    if tf.distribute.has_strategy():
      return self._distributed_train_step(experience)

    del weights  # unused

    reward, action, observation, batch_size = self._process_experience(
        experience)

    for k in range(self._num_models):
      diag_mask = tf.linalg.tensor_diag(
          tf.cast(tf.equal(action, k), self._dtype))
      observations_for_arm = tf.matmul(diag_mask, observation)
      rewards_for_arm = tf.matmul(diag_mask, tf.reshape(reward, [-1, 1]))

      num_samples_for_arm_current = tf.reduce_sum(diag_mask)
      tf.compat.v1.assign_add(self._num_samples_list[k],
                              num_samples_for_arm_current)
      num_samples_for_arm_total = self._num_samples_list[k].read_value()

      # Update the matrix A and b.
      # pylint: disable=cell-var-from-loop,g-long-lambda
      def update(cov_matrix, data_vector):
        return lin_agent.update_a_and_b_with_forgetting(
            cov_matrix, data_vector, rewards_for_arm, observations_for_arm,
            self._gamma, self._use_eigendecomp)
      a_new, b_new, eig_vals, eig_matrix = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0,
          lambda: update(self._cov_matrix_list[k], self._data_vector_list[k]),
          lambda: (self._cov_matrix_list[k], self._data_vector_list[k],
                   self._eig_vals_list[k], self._eig_matrix_list[k]))

      tf.compat.v1.assign(self._cov_matrix_list[k], a_new)
      tf.compat.v1.assign(self._data_vector_list[k], b_new)
      tf.compat.v1.assign(self._eig_vals_list[k], eig_vals)
      tf.compat.v1.assign(self._eig_matrix_list[k], eig_matrix)

    loss = -1. * tf.reduce_sum(reward)
    self.compute_summaries(loss)

    self._train_step_counter.assign_add(batch_size)

    return tf_agent.LossInfo(loss=(loss), extra=())