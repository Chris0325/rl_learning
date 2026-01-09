import numpy as np
from scipy import stats

from utils.util import *
from utils.tabular_util import *

n = 21
state_space = tabular_states(nrow=n, ncol=n)

action_space = [(i, -i) for i in range(-5, 6)]
action_name = {a: a[0] for a in action_space}


def p(s, a, *, nrow, ncol):
    row, col = s
    row += a[0]
    col += a[1]

    row = int(np.clip(row, 0, nrow-1))
    col = int(np.clip(col, 0, ncol-1))

    return (row, col), -2 * abs(a[0])


def possion(lam, n):
    poisson_dist = stats.poisson(mu=lam)
    probs = poisson_dist.pmf(np.arange(0, n))
    probs[-1] = 1 - probs[:-1].sum()
    return probs


request_probs = np.outer(possion(3, n), possion(4, n))
return_probs = np.outer(possion(3, n), possion(2, n))


def valid_action(s, a):
    return s[0] >= a[0] if a[0] >= 0 else s[1] >= a[1]


def stochastic_state(s, *, nrow, ncol, prob_threshold):
    state_rewards = []
    for (request_row, request_col), request_prob in np.ndenumerate(request_probs):
        if request_prob > prob_threshold:
            r = 10 * (min(s[0], request_row) + min(s[1], request_col))
            s_after_request = (max(s[0] - request_row, 0), max(s[1] - request_col, 0))
            for (return_row, return_col), return_prob in np.ndenumerate(return_probs):
                s_next = (min(s_after_request[0] + return_row, nrow-1), min(s_after_request[1] + return_col, ncol-1))
                if request_prob * return_prob > prob_threshold:
                    state_rewards.append((s_next, request_prob * return_prob, r))
    return state_rewards

policy = np.outer(np.ones(n**2), np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])).reshape((n, n, 11))
V, pi = policy_iteration(policy=policy, nrow=n, ncol=n, γ=0.9, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action, stochastic_state=stochastic_state, max_iterations=1)

print_policy(pi, action_space=action_space, action_name=action_name)
