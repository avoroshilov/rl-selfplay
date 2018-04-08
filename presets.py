def fill_preset_cfg_default(preset_cfg):
	""" Default preset configuration """
	preset_cfg.action_space_discrete = True
	preset_cfg.normalize_observations = True
	# Affects long-term reward calculation: R = r0 + gamma*r1 + (gamma^2)*r2 + ... + (gamma^n)*rn
	# Effective horizon: num_lookup_steps = 1/(1-gamma), i.e. for gamma=0.99, EH=100 steps lookahead
	preset_cfg.discount_gamma = 0.99
	preset_cfg.GAE_lambda = 0.99
	preset_cfg.learning_rate = 1e-4
	preset_cfg.explore_initial = 0
	preset_cfg.explore_coeff = 0
	preset_cfg.w_entropy = 0.01
	preset_cfg.success_reward = 0
	preset_cfg.fc_sizes_policy = [32, 32]
	preset_cfg.fc_sizes_value = [32, 32]
	preset_cfg.train_splits = 1
	preset_cfg.ppo_clip_value = 0.2
	preset_cfg.episodes_per_iteration = 10

	# Defaults that shouldn't be modified by any preset (derived defaults)
	preset_cfg.explore_cur = 0.0
	preset_cfg.iterations_total = 0


def fill_preset_cfg(preset_cfg, preset_id):
	""" Specific preset configurations """
	if preset_id == 0:
		# OpenAI.Gym : Cartpole [discrete]
		preset_cfg.action_space_discrete = True
		preset_cfg.discount_gamma = 0.95
		preset_cfg.GAE_lambda = 0.99
		preset_cfg.env_type = 'openai.gym'
		preset_cfg.env_name = 'CartPole-v0'
		preset_cfg.learning_rate = 1e-4
		preset_cfg.explore_initial = 30
		preset_cfg.explore_coeff = 0.99
		preset_cfg.w_entropy = 0.01
		preset_cfg.normalize_observations = False
		preset_cfg.success_reward = 195
		preset_cfg.fc_sizes_policy = [20, 20]
		preset_cfg.fc_sizes_value = [20, 20]
		preset_cfg.train_splits = 1
		preset_cfg.ppo_clip_value = 0.2
		preset_cfg.episodes_per_iteration = 10
	elif preset_id == 1:
		# OpenAI.Gym : Mountaincar [discrete]
		preset_cfg.action_space_discrete = True
		preset_cfg.discount_gamma = 0.995
		preset_cfg.GAE_lambda = 0.97
		preset_cfg.env_type = 'openai.gym'
		preset_cfg.env_name = 'MountainCar-v0'
		preset_cfg.learning_rate = 1e-4
		preset_cfg.explore_initial = 50
		preset_cfg.explore_coeff = 0.995
		preset_cfg.w_entropy = 0.01
		preset_cfg.normalize_observations = True
		preset_cfg.success_reward = -190
		preset_cfg.fc_sizes_policy = [64, 64]
		preset_cfg.fc_sizes_value = [64, 64]
		preset_cfg.train_splits = 1
		preset_cfg.ppo_clip_value = 0.2
		preset_cfg.episodes_per_iteration = 50
	elif preset_id == 2:
		# OpenAI.Gym : Acrobot [discrete]
		preset_cfg.action_space_discrete = True
		preset_cfg.discount_gamma = 0.99
		preset_cfg.GAE_lambda = 0.99
		preset_cfg.env_type = 'openai.gym'
		preset_cfg.env_name = 'Acrobot-v1'
		preset_cfg.learning_rate = 2e-4
		preset_cfg.explore_initial = 0
		preset_cfg.explore_coeff = 0
		preset_cfg.w_entropy = 0.05
		preset_cfg.normalize_observations = False
		preset_cfg.success_reward = -100
		preset_cfg.fc_sizes_policy = [64, 64]
		preset_cfg.fc_sizes_value = [64, 64]
		preset_cfg.train_splits = 1
		preset_cfg.ppo_clip_value = 0.2
		preset_cfg.episodes_per_iteration = 10
	elif preset_id == 3:
		# OpenAI.Gym : Pendulum [continuous]
		preset_cfg.action_space_discrete = False
		preset_cfg.discount_gamma = 0.99
		preset_cfg.GAE_lambda = 0.99
		preset_cfg.env_type = 'openai.gym'
		preset_cfg.env_name = 'Pendulum-v0'
		preset_cfg.learning_rate = 2e-4
		preset_cfg.explore_initial = 0
		preset_cfg.explore_coeff = 0
		preset_cfg.w_entropy = 0.05
		preset_cfg.normalize_observations = False
		preset_cfg.success_reward = -100
		preset_cfg.fc_sizes_policy = [64, 64]
		preset_cfg.fc_sizes_value = [64, 64]
		preset_cfg.train_splits = 1
		preset_cfg.ppo_clip_value = 0.2
		preset_cfg.episodes_per_iteration = 10
	if preset_id == 4:
		# Tic Tac Toe, default preset
		preset_cfg.action_space_discrete = True
		preset_cfg.discount_gamma = 0.95
		preset_cfg.GAE_lambda = 0.99
		preset_cfg.env_type = 'tictactoe'
		preset_cfg.env_name = 'whatever'
		preset_cfg.learning_rate = 1e-4
		preset_cfg.explore_initial = 30
		preset_cfg.explore_coeff = 0.99
		preset_cfg.w_entropy = 0.005
		preset_cfg.normalize_observations = False
		preset_cfg.success_reward = 0
		preset_cfg.fc_sizes_policy = [64, 64]
		preset_cfg.fc_sizes_value = [64, 64]
		preset_cfg.train_splits = 1
		preset_cfg.ppo_clip_value = 0.2
		preset_cfg.episodes_per_iteration = 10
	if preset_id == 5:
		# Tic Tac Toe, self-play  preset
		preset_cfg.action_space_discrete = True
		preset_cfg.discount_gamma = 0.95
		preset_cfg.GAE_lambda = 0.99
		preset_cfg.env_type = 'tictactoe'
		preset_cfg.env_name = 'selfplay'
		preset_cfg.learning_rate = 1e-4
		preset_cfg.explore_initial = 50
		preset_cfg.explore_coeff = 0.99
		preset_cfg.w_entropy = 0.05
		preset_cfg.normalize_observations = False
		preset_cfg.success_reward = 0
		preset_cfg.fc_sizes_policy = [32, 32]
		preset_cfg.fc_sizes_value = [32, 32]
		preset_cfg.train_splits = 1
		preset_cfg.ppo_clip_value = 0.2
		preset_cfg.episodes_per_iteration = 50
