from section_4_1_gridworld import *
from utils.tabular_action_util import *

policy = np.ones((4, 4, 4)) / 4
V, pi = action_policy_iteration(policy=policy, nrow=4, ncol=4, γ=.9, p=p, state_space=state_space, action_space=action_space, action_name=action_name, max_iterations=10)
