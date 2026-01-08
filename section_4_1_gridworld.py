import numpy as np
from utils.util import *
from utils.gridworld_util import *


def p(s, a, *, nrow, ncol):
    row, col = divmod(s, ncol)
    row += a[0]
    col += a[1]
    return int(np.clip(row, 0, nrow-1) * ncol + np.clip(col, 0, ncol-1)), -1


if __name__ == "__main__":
    # V = analytical_state_value(nrow=4, ncol=4, γ=1, p=p, s_space=range(1, 15))
    V = iterative_state_value(nrow=4, ncol=4, γ=1, p=p, s_space=range(1, 15), max_iterations=1000)

    print_matrix(V, nrow=4, ncol=4)
    print(policy(V, nrow=4, ncol=4, γ=1, p=p, s_space=range(1, 15)))
