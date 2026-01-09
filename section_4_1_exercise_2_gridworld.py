from utils.util import *
from utils.tabular_util import *
from section_4_1_gridworld import *


def p_1(s, a, *, nrow, ncol):
    if s[0] < nrow - 1:
        return p(s, a, nrow=nrow-1, ncol=ncol)
    elif s == (4, 1):
        if a == (0, -1):
            return index_to_coordinate(12, ncol=4), -1
        elif a == (-1, 0):
            return index_to_coordinate(13, ncol=4), -1
        elif a == (0, 1):
            return index_to_coordinate(14, ncol=4), -1
        elif a == (1, 0):
            return s, -1


def p_2(s, a, *, nrow, ncol):
    s_next, r = p_1(s, a, nrow=nrow, ncol=ncol)
    if s == (4, 1) and a == (1, 0):
        s_next = index_to_coordinate(17, ncol=4)
    return s_next, r


if __name__ == "__main__":
    V = analytical_state_value(nrow=5, ncol=4, γ=1, p=p_1, pi=random_pi, s_space=s_space + [(4, 1)], round=2)
    print_matrix(V)

    V = analytical_state_value(nrow=5, ncol=4, γ=1, p=p_2, pi=random_pi, s_space=s_space + [(4, 1)], round=2)
    print_matrix(V)
