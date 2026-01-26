from section_6_4_windy_gridworld import *

# exercise 6.4
V = np.zeros((nrow, ncol, 8))
action_space = action_space + [(i, j) for i in [-1, 1] for j in [-1, 1]]
run(V, action_space)


# exercise 6.5
def row_stochastic(s):
    return wind[s[1]] + np.random.choice([-1, 0, 1])

V = np.zeros((nrow, ncol, 8))
run(V, action_space, row_stochastic=row_stochastic)

# exercise 6.4
V = np.zeros((nrow, ncol, 9))
action_space = action_space + [(0, 0)]
run(V, action_space)
