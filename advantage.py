import copy

def calc_advantage(rewards, v_t_pred, v_t1_pred, discount_gamma=0.995, gae_discount_lambda=0.97):
	"""
	Calculate GAE (Generalized Advantage Estimator):
		delta_t = r_t + discount*V(s_(t+1)) - V(s_t)
		A = delta_t + (lambda*discount)*delta_(t+1) + ...
	See PPO eqn. (11)
	:param rewards:
	:param v_t_pred: predicted state value for s_t
	:param v_t1_pred: predicted state value for s_(t+1)
	:param gae_discount_lambda: additional GAE discount lambda
	"""
	deltas = [r_t + discount_gamma * v_t1 - v_t for r_t, v_t, v_t1 in zip(rewards, v_t_pred, v_t1_pred)]
	advantage = copy.deepcopy(deltas)
	for t in reversed(range(len(advantage) - 1)):
		advantage[t] = advantage[t] + (gae_discount_lambda*discount_gamma) * advantage[t + 1]
	return advantage
