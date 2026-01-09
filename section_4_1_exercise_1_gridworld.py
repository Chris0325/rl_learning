from utils.util import *
from section_4_1_gridworld import *


V = analytical_state_value(nrow=4, ncol=4, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space)
# print_matrix(V)
# q(11, down) = -1
s_next, r = p(index_to_coordinate(7, ncol=4), (1, 0), nrow=4, ncol=4)
print(r + V[*s_next])
