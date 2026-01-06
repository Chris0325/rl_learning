import numpy as np

n = 5
γ = 0.9

action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]
action_symbols = ['↑', '↓', '←', '→']


def step(s, a):
    old_row, old_col = row, col = divmod(s, n)
    if row == 0 and col == 1:
        return 4 * n + 1, 10
    elif row == 0 and col == 3:
        return 2 * n + 3, 5
    else:
        row += a[0]
        col += a[1]
        if row < 0 or row >= n or col < 0 or col >= n:
            return old_row * n + old_col, -1
        else:
            return row * n + col, 0


def state_value():
    A = np.eye(n**2)
    b = np.zeros(n**2)
    for s in range(n**2):
        for a in action_space:
            prob = 1 / len(action_space)
            s_next, r = step(s, a)
            A[s, s_next] -= prob * γ
            b[s] += prob * r

    V = np.linalg.solve(A, b).round(1)
    return V.reshape((n, n))


if __name__ == "__main__":
    print(state_value())
