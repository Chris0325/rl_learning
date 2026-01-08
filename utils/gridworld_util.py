import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.util import *

action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_symbols = ['↑', '↓', '←', '→']


def analytical_state_value(*, nrow, ncol, γ, p, s_space, round=1):
    A = np.eye(nrow * ncol)
    b = np.zeros(nrow * ncol)
    for s in s_space:
        for a in action_space:
            prob = 1 / len(action_space)
            s_next, r = p(s, a, nrow=nrow, ncol=ncol)
            A[s, s_next] -= prob * γ
            b[s] += prob * r

    # print_matrix(A, nrow=nrow*ncol, ncol=nrow*ncol, round=2)

    return np.linalg.solve(A, b).round(round)


def iterative_state_value(*, nrow, ncol, γ, p, s_space, round=1, θ=1e-5, max_iterations=1000):
    V = np.zeros(nrow * ncol)

    for _ in tqdm(range(max_iterations)):
        Δ = 0
        for s in s_space:
            v, V[s] = V[s], 0
            for a in action_space:
                prob = 1 / len(action_space)
                s_next, r = p(s, a, nrow=nrow, ncol=ncol)
                V[s] += prob * (r + γ * V[s_next])
            Δ = max(Δ, abs(V[s] - v))
        if Δ < θ:
            break
    return V.flatten().round(round)


def policy(V, *, nrow, ncol, γ, p, s_space):
    policy = [['' for _ in range(ncol)] for _ in range(nrow)]

    for s in s_space:
        q_values = []
        for a in action_space:
            s_next, r = p(s, a, nrow=nrow, ncol=ncol)
            q_values.append(r + γ * V[s_next] if s_next != s else -float('inf'))
        q_values = np.array(q_values)

        row, col = divmod(s, ncol)
        for a_index in np.where(q_values == q_values.max())[0]:
            policy[row][col] += action_symbols[a_index]

    plt.axis('off')
    table = plt.table(cellText=policy, loc='center', cellLoc='center')
    for i in range(nrow):
        for j in range(ncol):
            cell = table[(i, j)]
            cell.set_height(1.0/nrow)
            cell.set_width(1.0/ncol)
    plt.show()
