from utils.state_value import *

state_space = [(0, i) for i in range(1, 20)]
action_space = [(0, 1), (0, -1)]

V = np.zeros((1, 21))


def p(s, a, *, nrow, ncol):
    col = s[1] + a[1]
    r = -1 if not col else (1 if col == 20 else 0)
    return [Transition((0, col), r, 1)]


def random_pi(s):
    return np.array([.5, .5])


if __name__ == '__main__':
    V = analytical_state_value(nrow=1, ncol=21, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space, round=5)
    # V = iterative_state_value(nrow=1, ncol=21, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space, max_iterations=1000, round=5)

    print_matrix(V)
