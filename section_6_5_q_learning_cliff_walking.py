from utils.gridworld_util import *
from section_6_4_sarsa_windy_gridworld import *

nrow, ncol = 4, 12


def p(s, a, *, nrow, ncol):
    row, col = s[0] + a[0], s[1] + a[1]
    s_next = (int(np.clip(row, 0, nrow-1)), int(np.clip(col, 0, ncol-1)))
    s_next, r = [(0, 0), -100] if s_next[0] == 3 and 0 < s_next[1] < ncol - 1 else [s_next, -1]
    return [Transition(s_next, r, 1)]


def q_learning(n, alpha, epsilon, *, Q, action_space, nrow, ncol, p, T, s_begin, s_ends, γ=1, QQ=None):
    episode_time, rewards, action_history = [], [], []
    j = 0
    for i in range(n):
        action_counter = defaultdict(Counter)
        episode_reward = 0

        s = s_begin
        while s not in s_ends:
            a_index = greedy(q=Q[*s], epsilon=epsilon, action_space=action_space)[0]
            action_counter[s][a_index] += 1

            ts = p(s, action_space[a_index], nrow=nrow, ncol=ncol)
            t = np.random.choice(ts, p=[t.prob for t in ts])
            Q[*s][a_index] += alpha * (t.r + γ * Q[*t.s_next].max() - Q[*s][a_index])
            s = t.s_next

            episode_reward += t.r
            j += 1
            episode_time.append([i, j])
            if len(episode_time) ==  T:
                return episode_time, rewards

        rewards.append(episode_reward)
        action_history.append(action_counter)
        # print(i, j, episode_reward)
    return episode_time, rewards, action_history


def run(runs, episodes, s_begin=(3, 0), s_ends=[(3, ncol-1)]):
    for method in [sarsa, q_learning]:
        stats = np.zeros((runs, episodes))
        for i in tqdm(range(runs)):
            Q = np.zeros((nrow, ncol, 4))
            stats[i] = np.array(method(n=episodes, alpha=.1, epsilon=.1, Q=Q, action_space=action_space, nrow=nrow, ncol=ncol, p=p, T=episodes*1000, s_begin=s_begin, s_ends=s_ends)[1])
        plt.plot(range(1, episodes+1), stats.mean(axis=0), label=method.__name__)
        plt.ylim((-100, 0))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(100, 500)
