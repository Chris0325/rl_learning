from utils.util import *
from utils.gridworld_util import *
from section_4_1_gridworld import p as old_p


def p_1(s, a, *, nrow, ncol):
    row, col = divmod(s, ncol)
    if row < nrow - 1:
        return old_p(s, a, nrow=nrow-1, ncol=ncol)
    elif s == 17:
        if a == (0, -1):
            return 12, -1
        elif a == (-1, 0):
            return 13, -1
        elif a == (0, 1):
            return 14, -1
        elif a == (1, 0):
            return 17, -1


def p_2(s, a, *, nrow, ncol):
    s_next, r = p_1(s, a, nrow=nrow, ncol=ncol)
    if s == 13 and a == (1, 0):
        s_next = 17
    return s_next, r


if __name__ == "__main__":
    V = analytical_state_value(nrow=5, ncol=4, γ=1, p=p_1, s_space=list(range(1, 15))+[17], round=2)
    print_matrix(V, nrow=5, ncol=4, round=2)

    V = iterative_state_value(nrow=5, ncol=4, γ=1, p=p_2, s_space=list(range(1, 15))+[17], round=2)
    print_matrix(V, nrow=5, ncol=4, round=2)