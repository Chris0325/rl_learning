from utils.util import *
from section_4_1_gridworld import *

nrow = ncol = 4

V = analytical_state_value(nrow=nrow, ncol=ncol, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space)
print(q_expected_update_by_v(index_to_coordinate(11, ncol=4), (1, 0), nrow=nrow, ncol=ncol, γ=1, p=p, V=V, acc_prob=1, prob_threshold=0))

print(q_expected_update_by_v(index_to_coordinate(7, ncol=4), (1, 0), nrow=nrow, ncol=ncol, γ=1, p=p, V=V, acc_prob=1, prob_threshold=0))
