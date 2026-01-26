from section_6_4_sarsa_windy_gridworld import *

logging.getLogger().setLevel(logging.INFO)

Q = np.zeros((nrow, ncol, 8))
action_space = action_space + [(i, j) for i in [-1, 1] for j in [-1, 1]]
run(Q, action_space)


Q = np.zeros((nrow, ncol, 9))
action_space = action_space + [(0, 0)]
run(Q, action_space)
