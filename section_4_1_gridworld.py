import numpy as np
from utils.util import *
from utils.gridworld_util import *

s_space = [index_to_coordinate(n, ncol=4) for n in range(1, 15)]


def p(s, a, *, nrow, ncol):
    row, col = s[0] + a[0], s[1] + a[1]
    return (int(np.clip(row, 0, nrow-1)), int(np.clip(col, 0, ncol-1))), -1


if __name__ == "__main__":
    # V = analytical_state_value(nrow=4, ncol=4, γ=1, p=p, pi=random_pi, s_space=s_space)
    V = iterative_state_value(nrow=4, ncol=4, γ=1, p=p, pi=random_pi, s_space=s_space, max_iterations=1000)

    print_matrix(V)
    policy = value_policy(V, nrow=4, ncol=4, γ=1, p=p, s_space=s_space)
    print_policy(policy)