from section_4_1_gridworld import *
from utils.tabular_state_util import *

n = 101

action_space = range(n-1)
action_name = {a: a for a in action_space}
state_space = [(0, i) for i in range(1, n-1)]

V = np.zeros((1, n))
V[0][n-1] = 1


def valid_action(s, a):
    return a > 0 and a <= s[1] and a <= n - 1 - s[1]


def p(s, a, *, nrow, ncol, h_prob=.4):
    return [((0, s[1] - a), 0, 1 - h_prob), ((0, s[1] + a), (1 if s[1] + a == n - 1 else 0), h_prob)]


state_value_iteration(V, nrow=1, ncol=n, γ=1, p=p, state_space=state_space, action_space=action_space, action_name=action_name, valid_action=valid_action, θ=1e-7, plot_policy=functools.partial(print_policy, type='table'))
