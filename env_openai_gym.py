import numpy as np
import gym
try:
	import pybullet
	import pybullet_envs
except ImportError:
	print("No PyBullet module, certain envs will be unavailable")

import env_wrapper

class EnvGym(env_wrapper.EnvWrapper):
	def __init__(self, env_name):
		env_wrapper.EnvWrapper.__init__(self)
		self.type = 'openai.gym'
		self.name = env_name

		self.screen = None

		self.gym_env = gym.make(self.name)
        
		self.is_state_discr = False
		if hasattr(self.gym_env.action_space, 'n'):
			self.is_state_discr = True

	def init(self):
		""" Initializes the environment, returns nothuing """
		self.gym_env.seed(0)

	def reset(self):
		""" Resets the environment and returns the initial observation """
		obs = self.gym_env.reset()
		return obs

	def step(self, actions):
		"""
		Performs single step with specified action(s) and return:
			next observable state, reward, whether the episode is finished, and custom info
		"""
		obs, reward, is_done, info = self.gym_env.step(actions)
		return obs, reward, is_done, info

	def render(self):
		""" Render the environment's current state """
		if True:
			# Default gym environments supply whole rendering chain
			self.gym_env.render()
		else:
			# Some environments produce pure RGB arrays, and require
			# separate window setup, which is done with pygame module
			try:
				import pygame
				img_data = self.gym_env.render(mode="rgb_array")
				if self.screen == None:
					w = 800
					h = 600
					self.screen = pygame.display.set_mode((w, h))
				img_surf = pygame.surfarray.make_surface(img_data)
				self.screen.blit(pygame.transform.rotate(img_surf,-90),(0,0))
				pygame.display.flip()
			except ImportError:
				print("Module pygame is not installed - no rendering for this environment")

	def get_shape_observation(self):
		""" Returns observable state shape """
		if hasattr(self.gym_env.observation_space, 'n'):
			return (1,)
		return self.gym_env.observation_space.shape

	def get_shape_action(self):
		""" Returns observable state shape """
		if hasattr(self.gym_env.action_space, 'n'):
			return (self.gym_env.action_space.n,)
		else:
			return self.gym_env.action_space.shape
