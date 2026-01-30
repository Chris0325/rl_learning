from utils.gridworld_util import *
from utils.tabular_util import *


nrow, ncol = 6, 9

obstacles = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]

state_space = [s for s in tabular_states(nrow=nrow, ncol=ncol) if s not in obstacles]
s_begin, s_end = (2, 0), (0, 8)


def p(s, a, *, nrow, ncol, t):
    row, col = s[0] + a[0], s[1] + a[1]
    s_next = (int(np.clip(row, 0, nrow-1)), int(np.clip(col, 0, ncol-1)))
    if s_next in obstacles:
        return [Transition(s, 0, 1)]
    return [Transition(s_next, int(s_next == s_end), 1)]


def dyna_q(episodes, n, alpha, epsilon, *, Q, action_space, nrow, ncol, p, s_begin, s_ends, T=None, γ=1):
    model, episode_steps, rewards, t = dict(), [], [], 0
    for _ in range(episodes):
        s, steps = s_begin, 0
        while s not in s_ends:
            a_index = greedy(q=Q[*s], epsilon=epsilon, action_space=action_space)[0]
            trs = p(s, action_space[a_index], nrow=nrow, ncol=ncol, t=t)
            tr = np.random.choice(trs, p=[tr.prob for tr in trs])
            model[(s, a_index)] = (tr.s_next, tr.r)
            rewards.append(tr.r)

            Q[*s][a_index] += alpha * (tr.r + γ * Q[*tr.s_next].max() - Q[*s][a_index])

            qs_hist = list(model)
            for q_index in np.random.choice(len(qs_hist), size=(n if qs_hist else 0)):
                s_hist, a_hist = qs_hist[q_index]
                s_hist_next, r_hist = model[qs_hist[q_index]]
                Q[*s_hist][a_hist] += alpha * (r_hist + γ * Q[*s_hist_next].max() - Q[*s_hist][a_hist])

            s = tr.s_next
            steps += 1
            t += 1
            if T is not None and t >= T:
                return episode_steps, rewards
        episode_steps.append(steps)
    return episode_steps, rewards


if __name__ == '__main__':
    runs, episodes, s_ends = 30, 50, [s_end]
    for n in [0, 5, 50]:
        stats = np.zeros((runs, episodes))
        for i in tqdm(range(runs)):
            # np.random.seed(0)
            Q = np.zeros((nrow, ncol, len(action_space)))
            stats[i] = np.array(dyna_q(episodes=episodes, n=n, alpha=.1, epsilon=.1, Q=Q, action_space=action_space, nrow=nrow, ncol=ncol, p=p, s_begin=s_begin, s_ends=s_ends, γ=.95)[0])
        plt.plot(range(2, episodes+1), stats.mean(axis=0)[1:], label=f'n={n}')

    plt.legend()
    plt.show()
