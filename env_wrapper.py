class EnvWrapper:
	def __init__(self):
		self.type = 'undefined'
		self.name = 'undefined'

	def init(self):
		""" Initializes the environment, returns nothuing """
		pass

	def reset(self):
		""" Resets the environment and returns the initial observation """
		pass

	def step(self, actions):
		"""
		Performs single step with specified action(s) and return:
			next observable state, reward, whether the episode is finished, and custom info
		"""
		pass

	def render(self):
		""" Render the environment's current state """
		pass

	def get_shape_observation(self):
		""" Returns observable state shape """
		pass

	def get_shape_action(self):
		""" Returns observable state shape """
		pass

	def need_save_policy(self):
		"""
		Returns expected filename if env expects saved policy (e.g. self-play),
		or None - otherwise.
		This function will be first called right before init, and then - after each
		training iteration.
		"""
		return None

	def add_policy(self, policy_model_name):
		"""
		In case policy was saved sucessfully, this function will be called
		"""
		pass