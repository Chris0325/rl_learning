from section_6_4_sarsa_windy_gridworld import *

logging.getLogger().setLevel(logging.INFO)


def p(s, a, *, nrow, ncol):
    ts = []
    for row_stochasticity in [-1, 0, 1]:
        row, col = s[0] + a[0] + wind[s[1]] + row_stochasticity, s[1] + a[1]
        s_next = (int(np.clip(row, 0, nrow-1)), int(np.clip(col, 0, ncol-1)))
        ts.append(Transition(s_next, -1, 1/3))
    return ts


Q = np.zeros((nrow, ncol, 8))
run(Q, action_space=action_space + [(i, j) for i in [-1, 1] for j in [-1, 1]], p=p)
