"""
Module implements FC (Fully Connected, also commonly referred to as Muli-Layer Perceptron) networks
that output action probabilities and stat value prediction
"""

import numpy as np
import tensorflow as tf

class ActorCriticNetFC:
	"""
	Implements Actor-Critic (policy/value) networks using FC layers
	"""
	def __init__(self, name, obs_space_shape, act_space_shape, fc_sizes_policy, fc_sizes_value, bolt_kTemp=0.1, discrete=True):
		"""
		:param name: parent TF variable scope for this policy
		:param obs_space_shape: shape of the observation space
		:param act_space_shape: shape of the action space
		:param fc_sizes_policy: list of Fully Connected network hidden layers sizes for policy network
		:param fc_sizes_value: list of Fully Connected network hidden layers sizes for value network
		:param bolt_kTemp: boltzmann distribution constant (k*temperature)
		:param discrete: whether the action space is discrete or continuous
		"""

		self.discrete = discrete

		self.observation_space_size = obs_space_shape[0]
		self.action_space_size = act_space_shape[0]

		with tf.variable_scope(name):
			# Full TF scope name
			self.scope = tf.get_variable_scope().name

			self.observations = tf.placeholder(dtype=tf.float32, shape=[None] + list(obs_space_shape), name='observations')

			def add_fc_layer(name, input, size, activation_fn):
				"""
				Custom function to add fully-connected layer, suitable for adding parameter noise
				"""
				#return tf.layers.dense(inputs=input, units=size, activation=activation_fn)
				weight = tf.get_variable(name + "_w", [input.get_shape()[-1].value, size])
				bias = tf.get_variable(name + "_b", [size], initializer=tf.constant_initializer(0.0))
				hidden = tf.nn.bias_add(tf.matmul(input, weight), bias)
				if activation_fn:
					hidden = activation_fn(hidden)
				return hidden

			def policy_value_fc_core(name, input, sizes):
				""" Common core for Policy and Value networks """
				prev_layer = add_fc_layer(
					name=name+"_0",
					input=input,
					size=sizes[0],
					activation_fn=tf.tanh
					)
				for idx, fc_size in enumerate(sizes[1:]):
					prev_layer = add_fc_layer(
						name=name+"_%d"%(idx+1),
						input=prev_layer,
						size=fc_size,
						activation_fn=tf.tanh
						)
				return prev_layer

			def policy_fc_head(name, prev_layer, action_space_size, bolt_kTemp, needs_softmax=True):
				""" Policy FC head that outputs action probabilities """
				policy_layer = add_fc_layer(
					name=name+"_0",
					input=prev_layer,
					size=action_space_size,
					activation_fn=tf.tanh
					)
				# Action probabilities calculated via Boltzmann categorical distribution, where input
				#   is negative energy divided by (thermodynamic temperature times Boltzmann constant)
				act_probs = add_fc_layer(
					name=name+"_1",
					input=tf.negative(tf.divide(policy_layer, bolt_kTemp)),
					size=action_space_size,
					activation_fn=tf.nn.softmax if needs_softmax else None
					)
				return act_probs

			def value_fc_head(name, prev_layer):
				""" Value FC head that predicts state value """
				value = add_fc_layer(
					name=name+"_0",
					input=prev_layer,
					size=1,
					activation_fn=None
					)
				return value

			share_weights = False
			if share_weights:
				# Policy and value networks share some of the neural network structure
				with tf.variable_scope('core_fc'):
					prev_layer = policy_value_fc_core(
						name="pvc_shared",
						input=self.observations,
						sizes=fc_sizes_policy
						)

				with tf.variable_scope('policy_fc'):
					if self.discrete:
						self.act_probs = policy_fc_head("ph", prev_layer, self.action_space_size, bolt_kTemp)
					else:
						self.act_probs_mean = policy_fc_head("ph_mean", prev_layer, self.action_space_size, 1.0, False)
						self.act_probs_logstd = policy_fc_head("ph_logstd", prev_layer, self.action_space_size, 1.0, False)

				with tf.variable_scope('value_fc'):
					self.v_preds = value_fc_head("vh", prev_layer)
			else:
				# Policy and value networks are distinct neural networks
				with tf.variable_scope('policy_fc'):
					prev_layer = policy_value_fc_core(
						name="pc",
						input=self.observations,
						sizes=fc_sizes_policy
						)
					if self.discrete:
						self.act_probs = policy_fc_head("ph", prev_layer, self.action_space_size, bolt_kTemp)
					else:
						self.act_probs_mean = policy_fc_head("ph_mean", prev_layer, self.action_space_size, 1.0, False)
						self.act_probs_logstd = policy_fc_head("ph_logstd", prev_layer, self.action_space_size, 1.0, False)
				with tf.variable_scope('value_fc'):
					prev_layer = policy_value_fc_core(
						name="vc",
						input=self.observations,
						sizes=fc_sizes_value
						)
					self.v_preds = value_fc_head("vh", prev_layer)

	def act(self, tf_sess, observations):
		"""
		Retrieve action probabilities (for discrete action space) or action distributions (for continuous action space)
		from the current policy
		"""
		if self.discrete:
			return tf_sess.run([self.act_probs, self.v_preds], feed_dict={self.observations: observations})
		else:
			acts_mean, acts_logstd, v_preds = tf_sess.run(
				[self.act_probs_mean, self.act_probs_logstd, self.v_preds],
				feed_dict={self.observations: observations}
				)
			return acts_mean, acts_logstd, v_preds

	def get_variables(self):
		"""
		Get global variables from the associated scope
		"""
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

	def get_trainable_variables(self):
		"""
		Get trainable variables from the associated scope
		"""
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

