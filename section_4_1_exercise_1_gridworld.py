import numpy as np

n = 4
γ = 1

action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]
action_symbols = ['↑', '↓', '←', '→']


def step(s, a):
    row, col = divmod(s, n)
    row += a[0]
    col += a[1]
    return int(np.clip(row, 0, n-1) * n + np.clip(col, 0, n-1)), -1


def state_value():
    A = np.eye(n**2)
    b = np.zeros(n**2)
    for s in range(1, n**2-1):
        for a in action_space:
            prob = 1 / len(action_space)
            s_next, r = step(s, a)
            A[s, s_next] -= prob * γ
            b[s] += prob * r
    # import pandas as pd
    # print(pd.DataFrame(A))
    # print(b)
    V = np.linalg.solve(A, b).round(1)
    return V.reshape((n, n))

if __name__ == "__main__":
    V = state_value()

    # q(11, down) = -1
    s_next, r = step(7, (1, 0))
    print(r + γ * V.flatten()[s_next])
