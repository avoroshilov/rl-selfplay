import sys
import numpy as np
import time
import random
import json

import common_config
import env_wrapper
import render2d
import policy_play
from utils.dummy_struct import DummyStruct

PLAYER_X = 1
PLAYER_O = 2

class EnvTicTacToe(env_wrapper.EnvWrapper):

	class EnemyPolicy:
		""" Simple storage class to identify certain policy """
		def __init__(self):
			self.filename = None
			self.policy = None
			self.tf_session = None

	def __init__(self, env_name):
		env_wrapper.EnvWrapper.__init__(self)
		self.type = 'tictactoe'
		self.name = env_name

		self.self_play = False

		random.seed()

		self.screen_width = 800
		self.screen_height = 600
		self.renderer = None
		self.state_wrong = None
		self.state_win = None

		self.state_history_size = 3

		self.board_size = 3
		self.resetTicTac()

		self.action_space_size = self.num_cells
		self.action_space_abs_range = 0

		self.done = False

		if env_name == 'selfplay':
			self.self_play_stochastic = True
			self.self_play_iteration = 0
			self.self_play = True
			self.self_play_policies = []
			self.self_play_policies_max = 5

	def set_self_play_stochasticity(self, is_stochastic):
		self.self_play_stochastic = is_stochastic

	def load_selfplay_policy(self, policy_filename):
		self.self_play_preset_cfg = DummyStruct()

		with open(policy_filename + common_config.CONFIG_JSON_FILENAME, 'r') as cfg_f:
			config_json_str = cfg_f.read()
		# Yes, I know about `json.load`
		self.self_play_preset_cfg.deserialize(json.loads(config_json_str))

		self_play_policy, self_play_sess = policy_play.load_policy(
			policy_name='policy',
			load_model=policy_filename,
			obs_space_shape=self.get_shape_observation(),
			act_space_shape=self.get_shape_action(),
			fc_sizes_policy=self.self_play_preset_cfg.fc_sizes_policy,
			fc_sizes_value=self.self_play_preset_cfg.fc_sizes_value,
			action_space_discrete=self.self_play_preset_cfg.action_space_discrete
			)

		return self_play_policy, self_play_sess

	def init(self):
		pass

	def update_observations(self):
		self.observation = np.append(
			np.concatenate(tuple(self.state_history)),
			self.player_idx
			)

	def finalize_state(self):
		"""
		This function should be executed after each state change,
		for example this function could build state history
		"""
		del self.state_history[0]
		self.state_history.append(self.state)

	def resetTicTac(self):
		self.reward = 0
		self.num_cells = self.board_size*self.board_size
		# 0 - onoccupied, 1 - X, 2 - O
		player_turn = np.random.randint(0, 100)
		if player_turn < 50:
			self.player_idx = PLAYER_X
			self.enemy_idx = PLAYER_O
		else:
			self.player_idx = PLAYER_O
			self.enemy_idx = PLAYER_X

		self.state = np.zeros(self.num_cells)
		self.state_history = []
		# Fill state history
		for hist_idx in range(self.state_history_size):
			self.state_history.append(np.zeros(self.num_cells))
		if self.state_win is not None:
			del self.state_win
			self.state_win = None
		if self.state_wrong is not None:
			del self.state_wrong[:]
			self.state_wrong = None
		self.update_observations()
		self.done = False

		if self.player_idx == PLAYER_O:
			#print("Enemy step first!")
			self.step_enemy()
			self.update_observations()

	def mark_wrong_action(self, idx, player):
		self.state_wrong = [idx, player]

	def check_win(self, board, board_sidesize, state_idx, state_val):
		idx_y = state_idx // board_sidesize
		idx_x = state_idx - idx_y*board_sidesize

		#print("board: %s" % (board))

		is_win_x = True
		is_win_y = True
		for it in range(0, board_sidesize):
			# Check horizontal
			#print("hor (%d,%d)=%d" % (it, idx_y,board[it + idx_y*board_sidesize]))
			if is_win_x and board[it + idx_y*board_sidesize] != state_val:
				is_win_x = False
			# Check vertical
			#print("ver (%d,%d)=%d" % (idx_x, it,board[idx_x + it*board_sidesize]))
			if is_win_y and board[idx_x + it*board_sidesize] != state_val:
				is_win_y = False

		if is_win_x or is_win_y:
			self.state_win = []
			self.state_win.append(state_val)
			if is_win_x:
				self.state_win.append([0, idx_y])
				self.state_win.append([board_sidesize-1, idx_y])
			if is_win_y:
				self.state_win.append([idx_x, 0])
				self.state_win.append([idx_x, board_sidesize-1])
			#print("win_horver (%d,%d)" % (idx_x, idx_y))
			return True

		# Should we check diagonals?
		if idx_x != idx_y and idx_x != board_sidesize-1 - idx_y:
			return False

		# Check diagonals
		is_win_diagonal = True
		is_win_antidiag = True
		for it in range(0, board_sidesize):
			# Check diagonal
			if is_win_diagonal and board[it + it*board_sidesize] != state_val:
				is_win_diagonal = False
			# Check antidiagonal
			if is_win_antidiag and board[it + (board_sidesize-1 - it)*board_sidesize] != state_val:
				is_win_antidiag = False

		if is_win_diagonal or is_win_antidiag:
			self.state_win = []
			self.state_win.append(state_val)
			if is_win_diagonal:
				self.state_win.append([0, 0])
				self.state_win.append([board_sidesize-1, board_sidesize-1])
			if is_win_antidiag:
				self.state_win.append([0, board_sidesize-1])
				self.state_win.append([board_sidesize-1, 0])
			#print("win_diagonal(%d,%d)" % (idx_x, idx_y))
			return True

		return False

	def step_enemy(self):
		empty_cells = []
		it = np.nditer(self.state, flags=['f_index'])
		while not it.finished:
			if it[0] == 0:
				empty_cells.append(it.index)
			it.iternext()
		if not empty_cells:
			self.reward += 10
			self.done = True
			#print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>> win!\n")
		else:
			if not self.self_play or random.randint(0, 100) < 10:
			#if not self.self_play:
				# Perform random step
				act = empty_cells[random.randint(0, len(empty_cells) - 1)]
				#print("empty cells: %s" % (empty_cells,))
				#print("ai move: %d (%d)" % (ai_move, len(empty_cells) - 1))
				self.state[act] = self.enemy_idx
				self.finalize_state()
				# Check if opponent AI won
			else:
				enemy_observations = np.append(
					np.concatenate(tuple(self.state_history)),
					self.enemy_idx
					)

				if self.self_play_stochastic:
					num_self_play_policies = len(self.self_play_policies)
					action_distribution = np.zeros(self.get_shape_action(), dtype=np.float32)
					# Obtain all the probabilities from self-play policies
					for policy_idx in range(num_self_play_policies):
						act, v_pred = policy_play.step_policy(
							self.self_play_policies[-1].policy,
							self.self_play_policies[-1].tf_session,
							enemy_observations,
							None,
							False,
							True
							)
						action_distribution += act[0]
					# Feed random if there is not enough policies loaded
					if (self.self_play_policies_max > num_self_play_policies):
						for random_idx in range(self.self_play_policies_max - num_self_play_policies):
							act = np.random.random_sample(self.get_shape_action())
							action_distribution += act[0]
					# Normalize so that probabilities sum up to 1
					total_probability = 0.0
					for act_prob in action_distribution:
						total_probability += act_prob
					action_distribution /= total_probability

					act_dist = action_distribution
					act_choices = range(len(act_dist))
					act = np.random.choice(act_choices, p=act_dist)
				else:
					act, v_pred = policy_play.step_policy(
						self.self_play_policies[-1].policy,
						self.self_play_policies[-1].tf_session,
						enemy_observations,
						None,
						False,
						True
						)
					act = np.argmax(act[0])

				if act in empty_cells:
					self.state[act] = self.enemy_idx
					self.finalize_state()

					if len(empty_cells) == 1:
						# Tie
						self.done = False
				else:
					# AI poicy made a mistake
					self.reward -= 500
					self.mark_wrong_action(act, self.enemy_idx)
					self.done = True
					#print("AI policy mistake, finishing episode")
					return

			if self.check_win(self.state, self.board_size, act, self.enemy_idx):
				self.reward -= 100
				self.done = True
				#print("<<<<<<<<<<<<<<<<<<<<<<<< TRUE loss!")

	def step(self, action_idx):
		"""
		An environment dependent function that sends an action to the simulator.
		:param action_idx: the action to perform on the environment
		:return: None
		"""
		#print("act: %s"%(action_idx))
		cur_reward = self.reward
		if action_idx >= 0 and action_idx < self.num_cells and self.state[action_idx] == 0:
			self.state[action_idx] = self.player_idx
			self.finalize_state()
			self.reward += 1

			if self.check_win(self.state, self.board_size, action_idx, self.player_idx):
			#if False:
				self.reward += 100
				self.done = True
				#print(">>>>>>>>>>>>>>>>>>>>>>>>>>> TRUE win!")
			else:
				self.step_enemy()

		else:
			self.mark_wrong_action(action_idx, self.player_idx)
			self.reward -= 500
			self.done = True
		#print("reward: %d" % (self.reward))
		self.update_observations()
		return self.observation, self.reward - cur_reward, self.done, {}

	def draw_grid(self):
		aspect = self.screen_width / self.screen_height

		# Renderer coordinates are [-aspect, -1.0] - [aspect, 1.0]
		screen_w_div3 = 2.0 / 3.0
		screen_h_div3 = 2.0 / 3.0

		self.renderer.geoms.append(render2d.GeomLine(
			[0.0, 0.0, 0.0], [-1.0, -1.0 + screen_h_div3, 0.0],
			[0.0, 0.0, 0.0], [ 1.0, -1.0 + screen_h_div3, 0.0]
			))
		self.renderer.geoms.append(render2d.GeomLine(
			[0.0, 0.0, 0.0], [-1.0, -1.0 + 2*screen_h_div3, 0.0],
			[0.0, 0.0, 0.0], [ 1.0, -1.0 + 2*screen_h_div3, 0.0]
			))

		self.renderer.geoms.append(render2d.GeomLine(
			[0.0, 0.0, 0.0], [-1.0 + screen_w_div3, -1.0, 0.0],
			[0.0, 0.0, 0.0], [-1.0 + screen_w_div3,  1.0, 0.0]
			))
		self.renderer.geoms.append(render2d.GeomLine(
			[0.0, 0.0, 0.0], [-1.0 + 2*screen_w_div3, -1.0, 0.0],
			[0.0, 0.0, 0.0], [-1.0 + 2*screen_w_div3,  1.0, 0.0]
			))

	def draw_X(self, color, idx_x, idx_y, width=1.0):
		shape_size = 2.0 / 3.0
		margin = shape_size * 0.9
		self.renderer.geoms.append(render2d.GeomLine(
			color, [-1.0 + idx_x * shape_size + margin, -1.0 + idx_y * shape_size + margin, 0.0],
			color, [-1.0 + idx_x * shape_size + shape_size - margin, -1.0 + idx_y * shape_size + shape_size - margin, 0.0],
			width
			))
		self.renderer.geoms.append(render2d.GeomLine(
			color, [-1.0 + idx_x * shape_size + shape_size - margin, -1.0 + idx_y * shape_size + margin, 0.0],
			color, [-1.0 + idx_x * shape_size + margin, -1.0 + idx_y * shape_size + shape_size - margin, 0.0],
			width
			))

	def draw_O(self, color, idx_x, idx_y, width=1.0):
		shape_size = 2.0 / 3.0
		shape_pos_x = -1.0 + idx_x * shape_size + 0.5 * shape_size
		shape_pos_y = -1.0 + idx_y * shape_size + 0.5 * shape_size
		self.renderer.geoms.append(render2d.GeomCircle(
			[shape_pos_x, shape_pos_y, 0.0],
			(shape_size / 2.0) * 0.9,
			color,
			False,
			24,
			width
			))

	def reset(self):
		"""
		:param force_environment_reset: Force the environment to reset even if the episode is not done yet.
		:return:
		"""
		if self.renderer is not None:
			self.renderer.reset_geoms()
			self.draw_grid()

		self.resetTicTac()
		return self.observation

	def render(self):
		if self.renderer is None:
			self.renderer = render2d.Renderer(self.screen_width, self.screen_height)
			self.renderer.clear_color = [1.0, 1.0, 1.0, 0.0]
			self.draw_grid()

		shape_color = [0.0, 0.5, 1.0]
		enemy_shape_color = [0.0, 0.0, 0.0]
		wrong_shape_color = [1.0, 0.0, 0.0]

		it = np.nditer(self.state, flags=['f_index'])
		while not it.finished:
			if it[0] != 0:
				pos_y = it.index // self.board_size
				pos_x = it.index - pos_y*self.board_size
				cur_color = shape_color if it[0] == self.player_idx else enemy_shape_color
				if it[0] == PLAYER_X:
					self.draw_X(cur_color, pos_x, pos_y)
				if it[0] == PLAYER_O:
					self.draw_O(cur_color, pos_x, pos_y)
			it.iternext()

		if self.state_wrong is not None:
			wrong_index = self.state_wrong[0]
			wrong_player = self.state_wrong[1]
			pos_y = wrong_index // self.board_size
			pos_x = wrong_index - pos_y*self.board_size
			if wrong_player == PLAYER_X:
				self.draw_X(wrong_shape_color, pos_x, pos_y, 4.0)
			elif wrong_player == PLAYER_O:
				self.draw_O(wrong_shape_color, pos_x, pos_y, 4.0)

		if self.state_win is not None:
			win_player = self.state_win[0]
			win_cell0 = self.state_win[1]
			win_cell1 = self.state_win[2]
			if self.player_idx == win_player:
				win_color = [0.0, 1.0, 0.0]
			else:
				win_color = [1.0, 0.0, 0.0]
			shape_size = 2.0 / 3.0
			margin = shape_size * 0.5
			self.renderer.geoms.append(render2d.GeomLine(
				win_color, [-1.0 + win_cell0[0] * shape_size + margin, -1.0 + win_cell0[1] * shape_size + margin, 0.0],
				win_color, [-1.0 + win_cell1[0] * shape_size + shape_size - margin, -1.0 + win_cell1[1] * shape_size + shape_size - margin, 0.0],
				4.0
				))

		self.renderer.render()
		time.sleep(0.5)

	def get_shape_observation(self):
		""" Returns observable state shape """
		return self.observation.shape

	def get_shape_action(self):
		""" Returns observable state shape """
		return (self.num_cells,)

	def need_save_policy(self):
		"""
		In case of self-play, return expected saved model path,
		None otherwise
		"""
		if self.self_play:
			iteration = self.self_play_iteration
			model_name = common_config.MODEL_FOLDER+'/env_tictactoe_sp/%04d'%(self.self_play_iteration)
			self.self_play_iteration += 1
			if iteration % 100 == 0:
				return model_name
			else:
				return None
		else:
			return None

	def add_policy(self, policy_model_name):
		"""
		Add policy to the list, free/delete unneeded policies
		"""
		if self.self_play:
			new_policy = self.EnemyPolicy()
			new_policy.filename = policy_model_name
			new_policy.policy, new_policy.tf_session = self.load_selfplay_policy(new_policy.filename)
			self.self_play_policies.append(new_policy)

			if len(self.self_play_policies) > self.self_play_policies_max:
				policy_play.unload_policy(self.self_play_policies[0].tf_session)
				del self.self_play_policies[0]
