from section_6_5_q_learning_cliff_walking import *


def expected_sarsa(n, alpha, epsilon, *, Q, action_space, nrow, ncol, p, T, s_begin, s_ends, γ=1, QQ=None):
    episode_time, rewards = [], []
    j = 0
    for i in range(n):
        episode_reward = 0

        s = s_begin
        a_index = greedy(q=Q[*s], epsilon=epsilon, action_space=action_space)[0]
        while s not in s_ends:
            trs = p(s, action_space[a_index], nrow=nrow, ncol=ncol)
            tr = np.random.choice(trs, p=[tr.prob for tr in trs])
            a_next_index, a_next_dist = greedy(q=Q[*tr.s_next], epsilon=epsilon, action_space=action_space)
            Q[*s][a_index] += alpha * (tr.r + γ * np.dot(Q[*tr.s_next], a_next_dist) - Q[*s][a_index])
            s, a_index = tr.s_next, a_next_index

            episode_reward += tr.r
            j += 1
            episode_time.append([i, j])
            if len(episode_time) ==  T:
                return episode_time, rewards

        rewards.append(episode_reward)
    return episode_time, rewards


def run(runs, episodes, s_begin=(3, 0), s_ends=[(3, ncol-1)]):
    for method in [sarsa, expected_sarsa]:
        stats = np.zeros((runs, episodes))
        for i in tqdm(range(runs)):
            Q = np.zeros((nrow, ncol, 4))
            stats[i] = np.array(method(n=episodes, alpha=.1, epsilon=.1, Q=Q, action_space=action_space, nrow=nrow, ncol=ncol, p=p, T=episodes*1000, s_begin=s_begin, s_ends=s_ends)[1])
        plt.plot(range(1, episodes+1), stats.mean(axis=0), label=method.__name__)
        plt.ylim((-2000, 0))
    plt.legend()
    plt.show()


run(10, 500)
