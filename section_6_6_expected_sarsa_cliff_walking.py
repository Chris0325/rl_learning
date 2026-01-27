from section_6_5_q_learning_cliff_walking import *


def expected_sarsa(n, alpha, epsilon, *, Q, action_space, nrow, ncol, p, T, s_begin, s_end, QQ=None):
    episode_time, rewards = [], []
    j = 0
    for i in range(n):
        episode_reward = 0

        s = s_begin
        a_index = greedy(q=Q[*s], epsilon=epsilon, action_space=action_space)[0]
        while s != s_end:
            ts = p(s, action_space[a_index], nrow=nrow, ncol=ncol)
            t = np.random.choice(ts, p=[t.prob for t in ts])
            a_next_index, a_next_dist = greedy(q=Q[*t.s_next], epsilon=epsilon, action_space=action_space)
            Q[*s][a_index] += alpha * (t.r + np.dot(Q[*t.s_next], a_next_dist) - Q[*s][a_index])
            s, a_index = t.s_next, a_next_index

            episode_reward += t.r
            j += 1
            episode_time.append([i, j])
            if len(episode_time) ==  T:
                return episode_time, rewards

        rewards.append(episode_reward)
    return episode_time, rewards


def run(runs, episodes, s_begin=(3, 0), s_end=(3, ncol-1)):
    for method in [sarsa, expected_sarsa]:
        stats = np.zeros((runs, episodes))
        for i in tqdm(range(runs)):
            Q = np.zeros((nrow, ncol, 4))
            stats[i] = np.array(method(n=episodes, alpha=.1, epsilon=.1, Q=Q, action_space=action_space, nrow=nrow, ncol=ncol, p=p, T=episodes*1000, s_begin=s_begin, s_end=s_end)[1])
        plt.plot(range(1, episodes+1), stats.mean(axis=0), label=method.__name__)
        plt.ylim((-2000, 0))
    plt.legend()
    plt.show()


run(10, 500)
