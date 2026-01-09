import numpy as np
from tqdm import tqdm

from utils.util import *

action_name = {(-1, 0): '↑', (1, 0): '↓', (0, -1): '←', (0, 1): '→'}
action_space = list(action_name.keys())


def analytical_state_value(*, nrow, ncol, γ, p, pi, s_space=None, round=1):
    A = np.eye(nrow * ncol)
    b = np.zeros(nrow * ncol)
    for s in (s_space if s_space is not None else [(i, j) for i in range(nrow) for j in range(ncol)]):
        for a, prob in pi(s):
            s_next, r = p(s, a, nrow=nrow, ncol=ncol)
            A[coordinate_to_index(s, ncol=ncol), coordinate_to_index(s_next, ncol=ncol)] -= prob * γ
            b[coordinate_to_index(s, ncol=ncol)] += prob * r

    # print_matrix(A, nrow=nrow*ncol, ncol=nrow*ncol, round=2)

    return np.linalg.solve(A, b).round(round).reshape((nrow, ncol))


def iterative_state_value(*, nrow, ncol, γ, p, pi, s_space, round=1, θ=1e-5, max_iterations=1000):
    V = np.zeros((nrow, ncol))

    for _ in tqdm(range(max_iterations)):
        Δ = 0
        for s in s_space:
            v, V[*s] = V[*s], 0
            for a, prob in pi(s):
                s_next, r = p(s, a, nrow=nrow, ncol=ncol)
                V[*s] += prob * (r + γ * V[*s_next])
            Δ = max(Δ, abs(V[s] - v))
        if Δ < θ:
            break
    return V.round(round)


def value_policy(V, *, nrow, ncol, γ, p, s_space):
    policy = [[[] for _ in range(ncol)] for _ in range(nrow)]

    for s in s_space:
        q_values = []
        for a in action_space:
            s_next, r = p(s, a, nrow=nrow, ncol=ncol)
            q_values.append(r + γ * V[*s_next] if s_next != s else -float('inf'))
        q_values = np.array(q_values)

        a_indices = np.where(q_values == q_values.max())[0]

        policy[s[0]][s[1]].extend([(action_space[a_index], 1/len(a_indices)) for a_index in a_indices])

    return policy


def print_policy(policy):
    nrow = len(policy)
    ncol = len(policy[0])
    
    string_policy = np.zeros((nrow, ncol), dtype=str)

    for i in range(nrow):
        for j in range(ncol):
            actions = policy[i][j]
            string_policy[i, j] = ''.join([action_name[a] for a, _ in actions])
    print_matrix(string_policy)
