"""
Module implements PPO with clipped surrogate function
"""
import copy

import numpy as np
import tensorflow as tf

class AgentPPO:
	"""
	Agent that implements Proximal Policy Optimization algorithm, specifically - clipped surrogate variant
	(and not adaptive KL penalty variant). See https://arxiv.org/abs/1707.06347 (referred to as PPO paper)
	"""
	def __init__(self, policy, old_policy, learning_rate=1e-4, discount_gamma=0.95, clip_value=0.2, w_value=1.0):
		"""
		:param policy: current policy network, which should define `act_probs` tensor of action probabilities
		:param old_policy: old policy network, which should define `act_probs` tensor of action probabilities
		:param learning_rate: optimizer (Adam) learning rate
		:param discount_gamma: reward function discount factor
		:param clip_value: PPO clip epsilon in the L^CLIP
		:param w_value: value loss (critic loss) weight, for policy/value layers reuse
		"""

		self._tf_sess = None

		self.policy = policy
		self.old_policy = old_policy
		self.discount_gamma = discount_gamma

		pi_trainable = self.policy.get_trainable_variables()
		old_pi_trainable = self.old_policy.get_trainable_variables()

		if policy.discrete:
			self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
		else:
			self.actions = tf.placeholder(dtype=tf.float32, shape=[None]+[policy.action_space_size], name='actions')
		self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
		self.advantage = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage')
		self.v_t1_preds = tf.placeholder(dtype=tf.float32, shape=[None], name='v_t1_preds')
		self.w_entropy = tf.placeholder(dtype=tf.float32, shape=[], name='w_entropy')

		# Build graph that updates old policy with the current policy values
		with tf.variable_scope('update_old_policy'):
			self.update_old_policy_ops = []
			for old_variables, variables in zip(old_pi_trainable, pi_trainable):
				self.update_old_policy_ops.append(tf.assign(old_variables, variables))

		with tf.variable_scope('loss'):
			with tf.variable_scope('CLIP'):
				if policy.discrete:
					# Calculate probability to take action a_t using current policy pi(theta)
					pi_a_t_prob = self.policy.act_probs * tf.one_hot(indices=self.actions, depth=self.policy.act_probs.shape[1])
					pi_a_t_prob = tf.reduce_sum(pi_a_t_prob, axis=1)

					# Calculate probability to take action a_t using old policy pi(theta_old)
					pi_old_a_t_prob = self.old_policy.act_probs * tf.one_hot(indices=self.actions, depth=self.old_policy.act_probs.shape[1])
					pi_old_a_t_prob = tf.reduce_sum(pi_old_a_t_prob, axis=1)
				else:
					# Normal distribution
					# To get positive std, logstd is trained - this also allow for potential probability calculation optimizations
					# Alternative to get positive std is to use the softplus (log(exp(value) + 1), see tf.nn.softplus)
					#TODO: optimize keeping logstd in mind - avoid additional exp and divisions
					pi_a_t_prob = 1.0 / (np.sqrt(2*np.pi) * tf.exp(self.policy.act_probs_logstd)) * \
									tf.exp(-tf.square(self.actions - self.policy.act_probs_mean)/(2.0 * tf.square(tf.exp(self.policy.act_probs_logstd))))
					pi_a_t_prob = tf.reduce_sum(pi_a_t_prob, axis=1)

					pi_old_a_t_prob = 1.0 / (np.sqrt(2*np.pi) * tf.exp(self.old_policy.act_probs_logstd)) * \
									tf.exp(-tf.square(self.actions - self.old_policy.act_probs_mean)/(2.0 * tf.square(tf.exp(self.old_policy.act_probs_logstd))))
					pi_old_a_t_prob = tf.reduce_sum(pi_old_a_t_prob, axis=1)

				pi_old_a_t_prob = tf.stop_gradient(pi_old_a_t_prob)

				# PPO pessimistic surrogate L^CLIP:
				#	ratio_t = pi(a_t | s_t) / pi_old(a_t | s_t)
				#	L^CLIP = E[ min(ratio_t * advantage, clip(ratio_t, 1-eps, 1+eps) * advantage) ]
				# See PPO paper, eqn. (7)
				ratio = tf.exp(tf.log(pi_a_t_prob) - tf.log(pi_old_a_t_prob))
				clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
				loss_clip = tf.minimum(self.advantage * ratio, self.advantage * clipped_ratio)
				loss_clip = tf.reduce_mean(loss_clip)
				tf.summary.scalar('L_CLIP', loss_clip)

			with tf.variable_scope('VF'):
				# Value function loss: L^VF = (V_target - V_pred)^2 = ||r+gamma*V' - V||^2
				# Mentioned in PPO paper, right after eqn. (9)
				loss_vf = tf.squared_difference(self.rewards + self.discount_gamma * self.v_t1_preds, self.policy.v_preds)
				loss_vf = tf.reduce_mean(loss_vf)
				tf.summary.scalar('L_VF', loss_vf)

			with tf.variable_scope('entropy'):
				# Calculate Entropy [H = -Sum( pi(x) * log(pi(x)) )] to encourage exploration
				# Mentioned in PPO paper, as S, right after eqn. (9)
				if policy.discrete:
					entropy = -tf.reduce_sum(self.policy.act_probs * tf.log(tf.clip_by_value(self.policy.act_probs, 1e-15, 1.0)), axis=1)
				else:
					entropy = -(pi_a_t_prob * tf.log(tf.clip_by_value(pi_a_t_prob, 1e-15, 1.0)))

				entropy = tf.reduce_mean(entropy, axis=0)
				tf.summary.scalar('L_entropy', entropy)

			# Calculate L^(CLIP+VF+S) = E[L^CLIP - c1*L^VF + c2*H]
			# See PPO paper, eqn. (9)
			loss = -( loss_clip - w_value * loss_vf + self.w_entropy * entropy )
			tf.summary.scalar('loss', loss)

		self.summary_merged = tf.summary.merge_all()
		self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=pi_trainable)

	def train_iteration_start(self):
		self.update_old_policy()

	@property
	def tf_session(self):
		"""
		Get TF session
		"""
		return self._tf_sess

	@tf_session.setter
	def tf_session(self, sess):
		"""
		Set TF session
		"""
		self._tf_sess = sess

	def train(self, observations, actions, rewards, advantage, v_t1_preds, w_entropy):
		"""
		Initiate agent training, TF session must be valid
		"""
		assert self._tf_sess is not None, 'TF session is not valid'
		self._tf_sess.run(
			[self.train_op],
			feed_dict={
				self.policy.observations: observations,
				self.old_policy.observations: observations,
				self.actions: actions,
				self.rewards: rewards,
				self.advantage: advantage,
				self.v_t1_preds: v_t1_preds,
				self.w_entropy: w_entropy
				}
			)

	def get_tf_summary(self, observations, actions, rewards, advantage, v_t1_preds, w_entropy):
		"""
		Get TF tensorboard, TF session must be valid
		"""
		assert self._tf_sess is not None, 'TF session is not valid'
		return self._tf_sess.run(
			[self.summary_merged],
			feed_dict={
				self.policy.observations: observations,
				self.old_policy.observations: observations,
				self.actions: actions,
				self.rewards: rewards,
				self.advantage: advantage,
				self.v_t1_preds: v_t1_preds,
				self.w_entropy: w_entropy
				}
			)

	def update_old_policy(self):
		"""
		Update old policy, should happen before agent training on certain iteration
		"""
		return self._tf_sess.run(self.update_old_policy_ops)
