import numpy as np

class RunningMeanStd(object):
	def __init__(self, shape):
		self._shape = shape
		self._num_samples = 0
		self._mean = np.zeros(shape)
		self._std = np.zeros(shape)

	def reset(self):
		self._num_samples = 0
		self._mean = np.zeros(self._shape)
		self._std = np.zeros(self._shape)

	def update(self, sample):
		sample = np.asarray(sample)
		assert sample.shape == self._mean.shape, 'Input shape mismatch'
		self._num_samples += 1
		if self._num_samples == 1:
			self._mean[...] = sample
		else:
			old_mean = self._mean.copy()
			self._mean[...] = old_mean + (sample - old_mean) / self._num_samples
			self._std[...] = self._std + (sample - old_mean) * (sample - self._mean)

	@property
	def n(self):
		return self._num_samples

	@property
	def mean(self):
		return self._mean

	@property
	def var(self):
		return self._std / (self._num_samples - 1) if self._num_samples > 1 else np.square(self._mean)

	@property
	def std(self):
		return np.sqrt(self.var)

	@property
	def shape(self):
		return self._mean.shape