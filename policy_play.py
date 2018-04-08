import json
import numpy as np
import tensorflow as tf

import common_config
from actorcritic_net_fc import ActorCriticNetFC
from utils.running_stat import RunningMeanStd

def load_policy(policy_name, load_model, obs_space_shape, act_space_shape, fc_sizes_policy, fc_sizes_value, action_space_discrete):
	policy_graph = tf.Graph()
	with policy_graph.as_default():
		policy = ActorCriticNetFC(
			name=policy_name,
			obs_space_shape=obs_space_shape,
			act_space_shape=act_space_shape,
			fc_sizes_policy=fc_sizes_policy,
			fc_sizes_value=fc_sizes_value,
			discrete=action_space_discrete
			)

	tf_sess = tf.Session(graph=policy_graph)
	with tf_sess.as_default():
		with policy_graph.as_default():
			saver = tf.train.Saver()
			tf_sess.run(tf.global_variables_initializer())
			saver.restore(tf_sess, load_model + common_config.CHECKPOINT_FILENAME)

	return policy, tf_sess

def unload_policy(sess):
	sess.close()

def step_policy(policy, tf_sess, observations, obs_stat, normalize_observations, action_space_discrete):
	# Normalize observations
	if normalize_observations:
		obs_stat.update(observations)
		observations = (observations - obs_stat.mean) / (obs_stat.std + 1e-15)
		observations = np.clip(observations, -5.0, 5.0)

	if action_space_discrete:
		observations = np.stack([observations]).astype(dtype=np.float32)
		act, pred_val = policy.act(tf_sess=tf_sess, observations=observations)
		return act, pred_val
	else:
		act_mean, act_logstd, pred_val = policy.act(
			tf_sess=tf_sess,
			observations=[observations]
			)
		return act_mean, act_logstd, pred_val


if __name__ == '__main__':

	from utils.dummy_struct import DummyStruct

	import presets
	import env_openai_gym
	import env_tictactoe

	load_model_name = common_config.MODEL_FOLDER + '/autosave/3000'

	preset_cfg = DummyStruct()

	print("%s" % (load_model_name))
	with open(load_model_name + common_config.CONFIG_JSON_FILENAME, 'r') as cfg_f:
		config_json_str = cfg_f.read()
	# Yes, I know about `json.load`
	preset_cfg.deserialize(json.loads(config_json_str))

	if preset_cfg.env_type == 'openai.gym':
		env = env_openai_gym.EnvGym(preset_cfg.env_name)
	elif preset_cfg.env_type == 'tictactoe':
		env = env_tictactoe.EnvTicTacToe(preset_cfg.env_name)

	obs_space_shape = env.get_shape_observation()
	act_space_shape = env.get_shape_action()

	act_policy, sess = load_policy(
		policy_name='policy',
		load_model=load_model_name,
		obs_space_shape=obs_space_shape,
		act_space_shape=act_space_shape,
		fc_sizes_policy=preset_cfg.fc_sizes_policy,
		fc_sizes_value=preset_cfg.fc_sizes_value,
		action_space_discrete=preset_cfg.action_space_discrete
		)

	env.init()
	observation = env.reset()
	reward = 0

	num_wins = 0

	obs_stat = RunningMeanStd(obs_space_shape)

	MAX_ITERATIONS = 50000

	for iteration in range(MAX_ITERATIONS):
		episode_reward = 0
		while True:

			if preset_cfg.action_space_discrete:
				act, pred_val = step_policy(act_policy, sess, observation, obs_stat, preset_cfg.normalize_observations, preset_cfg.action_space_discrete)
				act = np.argmax(act[0])
				next_obs, reward, done, info = env.step(act)
			else:
				act_mean, act_logstd, pred_val = step_policy(act_policy, sess, observation, obs_stat, preset_cfg.normalize_observations, preset_cfg.action_space_discrete)
				if True:
					act = act_mean
				else:
					act = act_mean + np.exp(act_logstd) * np.random.normal(np.zeros(act_mean.shape), np.ones(act_mean.shape))

				act = act.flatten()
				next_obs, reward, done, info = env.step(act)
				if not isinstance(reward, float):
					reward = np.asscalar(reward)

			episode_reward += reward

			env.render()

			if done:
				observation = env.reset()
				if episode_reward > preset_cfg.success_reward:
					num_wins += 1
				break
			else:
				observation = next_obs

		print("[%4d] (%d) Run reward: %d" % (iteration, num_wins, episode_reward))

	unload_policy(sess)
