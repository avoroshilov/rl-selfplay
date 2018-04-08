class DummyStruct:
	def serialize(self):
		return self.__dict__
	def deserialize(self, dict):
		self.__dict__ = dict
