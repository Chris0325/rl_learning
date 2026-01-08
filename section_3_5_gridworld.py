from utils.gridworld_util import *


def p(s, a, *, nrow, ncol):
    old_row, old_col = row, col = divmod(s, ncol)
    if row == 0 and col == 1:
        return 4 * ncol + 1, 10
    elif row == 0 and col == 3:
        return 2 * ncol + 3, 5
    else:
        row += a[0]
        col += a[1]
        if row < 0 or row >= nrow or col < 0 or col >= ncol:
            return old_row * ncol + old_col, -1
        else:
            return row * ncol + col, 0


print(analytical_state_value(nrow=5, ncol=5, γ=.9, p=p, s_space=range(25)).reshape((5, 5)))
