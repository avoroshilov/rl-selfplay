import sys
import os
import random
import argparse
import json
import datetime
import numpy as np
import tensorflow as tf

import common_config
from actorcritic_net_fc import ActorCriticNetFC
from ppo import AgentPPO
import presets
import env_openai_gym
import env_tictactoe

from advantage import calc_advantage
from utils.dummy_struct import DummyStruct
from utils.running_stat import RunningMeanStd


SUCCESS_RUNS_SEQ = 100

MAX_ITERATIONS = 500000

def main(render_mode=0, save_model_path=None, autosave_freq=0, load_model=None, no_train=False):
	autosave_model_path = common_config.MODEL_FOLDER + '/autosave'

	random.seed()
	np.random.seed(0)

	preset_cfg = DummyStruct()
	presets.fill_preset_cfg_default(preset_cfg)

	#########################################################################################
	# File operations
	if load_model is None:
		#TODO: implement actor-specific tweaks (like PPO clipping eps etc.)
		#TODO: implement env-specific states
		PRESET_IDX = 7
		presets.fill_preset_cfg(preset_cfg, PRESET_IDX)

		# Derive required preset properties
		preset_cfg.explore_cur = preset_cfg.explore_initial
		preset_cfg.iterations_total = 0
	else:
		print("Loading model %s" % (load_model))
		with open(load_model + common_config.CONFIG_JSON_FILENAME, 'r') as cfg_f:
			config_json_str = cfg_f.read()
		# Yes, I know about `json.load`
		preset_cfg.deserialize(json.loads(config_json_str))

	def save_state(tf_sesion, save_path):
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		saver.save(tf_sesion, save_path + common_config.CHECKPOINT_FILENAME)
		# Yes, I know about `json.dump`
		config_json_str = json.dumps(preset_cfg.serialize())
		with open(save_path + common_config.CONFIG_JSON_FILENAME, 'w') as cfg_f:
			cfg_f.write(config_json_str)
		print('Checkpoint+config saved as %s' % (save_path))
	#########################################################################################

	if preset_cfg.env_type == 'openai.gym':
		env = env_openai_gym.EnvGym(preset_cfg.env_name)
	elif preset_cfg.env_type == 'tictactoe':
		env = env_tictactoe.EnvTicTacToe(preset_cfg.env_name)

	obs_space_shape = env.get_shape_observation()
	act_space_shape = env.get_shape_action()

	policy = ActorCriticNetFC(
			name='policy',
			obs_space_shape=obs_space_shape,
			act_space_shape=act_space_shape,
			fc_sizes_policy=preset_cfg.fc_sizes_policy,
			fc_sizes_value=preset_cfg.fc_sizes_value,
			discrete=preset_cfg.action_space_discrete
			)
	old_policy = ActorCriticNetFC(
			name='old_policy',
			obs_space_shape=obs_space_shape,
			act_space_shape=act_space_shape,
			fc_sizes_policy=preset_cfg.fc_sizes_policy,
			fc_sizes_value=preset_cfg.fc_sizes_value,
			discrete=preset_cfg.action_space_discrete
			)
	agent = AgentPPO(
			policy=policy,
			old_policy=old_policy,
			discount_gamma=preset_cfg.discount_gamma,
			learning_rate=preset_cfg.learning_rate,
			clip_value=preset_cfg.ppo_clip_value
			)
	saver = tf.train.Saver()

	obs_stat = RunningMeanStd(obs_space_shape)

	with tf.Session() as tf_sess_policytrain:
		writer = tf.summary.FileWriter('./_tf-summary', tf_sess_policytrain.graph)
		tf_sess_policytrain.run(tf.global_variables_initializer())
		# Set agent's TF session
		agent.tf_session = tf_sess_policytrain

		env.init()
		save_model_path = env.need_save_policy()
		if save_model_path is not None:
			save_state(tf_sess_policytrain, save_model_path)
			env.add_policy(save_model_path)

		num_wins = 0

		if load_model is not None:
			saver.restore(tf_sess_policytrain, load_model + common_config.CHECKPOINT_FILENAME)

		for iteration in range(MAX_ITERATIONS):
			preset_cfg.iterations_total += 1

			observations = []
			actions = []
			rewards = []
			pred_vals = []

			episodes_num = preset_cfg.episodes_per_iteration
			episodes_lens = 0
			episodes_reward = 0
			for episode_idx in range(episodes_num):
				obs = env.reset()
				episode_reward = 0
				while True:
					episodes_lens += 1

					# Normalize observations (if requested)
					if preset_cfg.normalize_observations:
						obs_stat.update(obs)
						obs = (obs - obs_stat.mean) / (obs_stat.std + 1e-15)
						obs = np.clip(obs, -5.0, 5.0)

					# Calculate action distribution and predicted value of the state
					if preset_cfg.action_space_discrete:
						# Categorical distribution
						act, pred_val = policy.act(
								tf_sess=tf_sess_policytrain,
								observations=[obs]
								)

						if no_train:
							act = np.argmax(act[0])
						else:
							act_dist = act[0]
							act_choices = range(len(act_dist))
							act = np.random.choice(act_choices, p=act_dist)

						act = np.asscalar(act)
						if not no_train and (random.randint(0, 100) < preset_cfg.explore_cur):
							act = random.randint(0, act_space_shape[0]-1)

					else:
						# Gaussian distribution
						act_mean, act_logstd, pred_val = policy.act(
								tf_sess=tf_sess_policytrain,
								observations=[obs]
								)
						if no_train:
							act = act_mean
						else:
							act = act_mean + np.exp(act_logstd) * np.random.normal(np.zeros(act_mean.shape), np.ones(act_mean.shape))

						if not no_train and (random.randint(0, 100) < preset_cfg.explore_cur):
							act = np.random.normal(np.zeros(act_mean.shape), 1.0*np.ones(act_mean.shape))
						act = act.flatten()

					pred_val = np.asscalar(pred_val)

					if preset_cfg.action_space_discrete:
						next_obs, reward, done, info = env.step(act)
					else:
						next_obs, reward, done, info = env.step(act)
						if not isinstance(reward, float):
							reward = np.asscalar(reward)

					observations.append(obs)
					actions.append(act)
					rewards.append(reward)
					pred_vals.append(pred_val)

					episode_reward += reward

					request_rendering = False
					if render_mode == 1 or (render_mode == 2 and (iteration % 1000) > 990):
						request_rendering = (episode_idx == 0)

					if request_rendering:
						env.render()

					if done:
    					# Convenience: create array of values of next states, terminal state has 0 value
						pred_vals_t1 = pred_vals[1:] + [0]
						if episode_reward > preset_cfg.success_reward:
							num_wins += 1
						break
					else:
						obs = next_obs.flatten()

				episodes_reward += episode_reward

			episodes_reward_avg = sum(rewards) // episodes_num

			if preset_cfg.explore_cur > 1e-5:
				preset_cfg.explore_cur *= preset_cfg.explore_coeff
			else:
				preset_cfg.explore_cur = 0.0

			cur_datetime = datetime.datetime.now()
			timestamp = '%02d:%02d:%02d.%02d'%(cur_datetime.hour, cur_datetime.minute, cur_datetime.second, cur_datetime.microsecond // 10000)
			print("%s: [%4d] (%d) Avg. reward: %d, expl: %d" %
				(
					timestamp,
					iteration,
					num_wins,
					episodes_reward // episodes_num,
					int(preset_cfg.explore_cur)
				)
				)

			writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episodes_lens // episodes_num)]), iteration)
			writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=episodes_reward_avg)]), iteration)

			if autosave_freq != 0 and (iteration % autosave_freq) == autosave_freq-1:
				autosave_fullpath = "%s/%d" % (autosave_model_path, iteration+1)
				save_state(tf_sess_policytrain, autosave_fullpath)
				with open(autosave_fullpath + common_config.LAST_RESULTS_FILENAME, 'w') as result_f:
					result_f.write('%d' % (episodes_reward_avg))

			# Calculate Advantage function that helpes estimate policy performance
			advantage = calc_advantage(
					rewards=rewards,
					v_t_pred=pred_vals,
					v_t1_pred=pred_vals_t1,
					discount_gamma=preset_cfg.discount_gamma,
					gae_discount_lambda=preset_cfg.GAE_lambda
					)

			# Convert lists to numpy arrays
			observations = np.reshape(observations, newshape=[-1] + list(obs_space_shape))
			if preset_cfg.action_space_discrete:
				actions = np.array(actions).astype(dtype=np.int32)
			else:
				actions = np.array(actions).astype(dtype=np.float32)
			rewards = np.array(rewards).astype(dtype=np.float32)
			advantage = np.array(advantage).astype(dtype=np.float32)
			pred_vals_t1 = np.array(pred_vals_t1).astype(dtype=np.float32)

			# Standardize returns/advantage, so that there is always something to encourage/discourage
			advantage -= advantage.mean()
			advantage_std = advantage.std()
			if np.sum(advantage_std) != 0.0:
				advantage /= advantage_std

			if not no_train:
				agent.update_old_policy()

				for split in range(preset_cfg.train_splits):
					sample_indices = np.arange(0, observations.shape[0])
					np.random.shuffle(sample_indices)
					train_samples = sample_indices.shape[0] // preset_cfg.train_splits
					agent.train(
						observations=np.take(a=observations, indices=sample_indices[:train_samples], axis=0),
						actions=np.take(a=actions, indices=sample_indices[:train_samples], axis=0),
						rewards=np.take(a=rewards, indices=sample_indices[:train_samples], axis=0),
						advantage=np.take(a=advantage, indices=sample_indices[:train_samples], axis=0),
						v_t1_preds=np.take(a=pred_vals_t1, indices=sample_indices[:train_samples], axis=0),
						w_entropy=preset_cfg.w_entropy
						)

				save_model_path = env.need_save_policy()
				if save_model_path is not None:
					save_state(tf_sess_policytrain, save_model_path)
					env.add_policy(save_model_path)

			summary = agent.get_tf_summary(
					observations=observations,
					actions=actions,
					rewards=rewards,
					advantage=advantage,
					v_t1_preds=pred_vals_t1,
					w_entropy=preset_cfg.w_entropy
					)

			writer.add_summary(summary[0], iteration)

		if save_model_path and not no_train:
			save_state(tf_sess_policytrain, save_model_path)
			print('Training is done')

		writer.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--render',		dest='render', default=0, type=int, help='render mode: 0 - disabled, 1 - always, 2 - each 1000th iter')
	parser.add_argument('--load',		dest='load_model')
	parser.add_argument('--save',		dest='save_model_path', default=common_config.MODEL_FOLDER+'/savedmodel')
	parser.add_argument('--autosave',	dest='autosave_freq', type=int, default=1000)
	parser.add_argument('--notrain',	dest='notrain', action='store_true')
	parser.set_defaults(notrain=False)
	options = parser.parse_args()

	main(render_mode=options.render, save_model_path=options.save_model_path, autosave_freq=options.autosave_freq, load_model=options.load_model, no_train=options.notrain)
