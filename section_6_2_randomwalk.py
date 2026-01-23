from utils.tabular_state_util import *

state_space = [(0, i) for i in range(1, 6)]
action_space = [(0, 1), (0, -1)]

V = np.zeros((1, 7))

def p(s, a, nrow, ncol):
    r = 1 if s[1] + a[1] == 6 else 0
    return [((0, s[1] + a[1]), r, 1)]


def random_pi(s):
    return np.array([.5, .5])


# V = analytical_state_value(nrow=1, ncol=7, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space, round=5)
V = iterative_state_value(nrow=1, ncol=7, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space, max_iterations=1000, round=5)

print_matrix(V)
