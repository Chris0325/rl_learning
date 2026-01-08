from utils.util import *
from section_4_1_gridworld import *

V = analytical_state_value(nrow=4, ncol=4, γ=1, p=p, s_space=range(1, 15))
print_matrix(V, nrow=4, ncol=4)
# q(11, down) = -1
s_next, r = p(7, (1, 0), nrow=4, ncol=4)
print(r + V[s_next])
