from utils.tabular_state_util import *

n = 1000
state_space = [(0, i) for i in range(1, n+1)]
action_space = [(0, i) for i in range(1, 101)] + [(0, -i) for i in range(1, 101)]

V = np.zeros((1, n+2))


def p(s, a, nrow, ncol):
    row, col = s[0] + a[0], s[1] + a[1]
    s_next = (int(np.clip(row, 0, nrow-1)), int(np.clip(col, 0, ncol-1)))
    r = 1 if s_next[1] == n+1 else (-1 if s_next[1] == 0 else 0)
    return [Transition(s_next, r, 1)]


def random_pi(s):
    return np.ones(200) / 200


V = analytical_state_value(nrow=1, ncol=n+2, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space, round=5)
# V = iterative_state_value(nrow=1, ncol=n+2, γ=1, p=p, pi=random_pi, state_space=state_space, action_space=action_space, max_iterations=1000, round=5)
# print_matrix(V)


def sample_episode(nrow, ncol, action_space, pi, p, s_begin, s_ends):
    s = s_begin
    trajectory = [s]
    while s not in s_ends:
        a_index = np.random.choice(len(action_space), p=pi(s))
        a = action_space[a_index]
        trs = p(s, a, nrow, ncol)
        tr = np.random.choice(trs, p=[tr.prob for tr in trs])
        trajectory.extend([tr.r, tr.s_next])
        s = tr.s_next
    return trajectory


def aggregate_state(s):
    return (s[0], (s[1] - 1) // 100 + 1)


def gradient_mc(episodes, alpha, V, *, nrow, ncol, action_space, p, s_begin, s_ends, γ=1):
    for _ in tqdm(range(episodes)):
        trajectory = sample_episode(nrow, ncol, action_space, random_pi, p, s_begin, s_ends)
        G = 0
        for t in range(len(trajectory)-3, -1, -2):
            s = trajectory[t]
            r = trajectory[t + 1]
            G = r + γ * G
            s_agg = aggregate_state(s)
            V[*s_agg] += alpha * (G - V[*s_agg])


VV = np.zeros((1, 12))
gradient_mc(episodes=100000, alpha=2e-5, V=VV, nrow=1, ncol=n+2, action_space=action_space, p=p, s_begin=(0, 500), s_ends=[(0, 0), (0, n+1)], γ=1)
print_matrix(VV)


plt.plot(range(1, n+1), V[0][1:-1], label='Analytical')
plt.plot(range(1, n+1), [VV[*aggregate_state(s)] for s in state_space], label='Gradient MC with State Aggregation')
plt.legend()
plt.show()
