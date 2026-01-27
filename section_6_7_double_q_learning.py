from section_6_5_q_learning_cliff_walking import *

nrow, ncol = 1, 3
action_space = [(0, -1), (0, 1)]


def p(s, a, *, nrow, ncol):
    row, col = s[0] + a[0], s[1] + a[1]
    col = 0 if col == 3 else col

    if s == (0, 1):
        return [Transition((row, col), np.random.normal(loc=-.1, scale=1), 1)]
    else:
        return [Transition((row, col), 0, 1)]


def double_q_learning(n, alpha, epsilon, *, Q, action_space, nrow, ncol, p, T, s_begin, s_end, QQ=None):
    episode_time, rewards, action_history = [], [], []
    j = 0
    for i in range(n):
        action_counter = defaultdict(Counter)
        episode_reward = 0

        s = s_begin
        while s != s_end:
            a_index = greedy(q=Q[*s]+QQ[*s], epsilon=epsilon, action_space=action_space)[0]
            action_counter[s][a_index] += 1

            ts = p(s, action_space[a_index], nrow=nrow, ncol=ncol)
            t = np.random.choice(ts, p=[t.prob for t in ts])

            if np.random.random() < .5:
                Q[*s][a_index] += alpha * (t.r + QQ[*t.s_next][np.argmax(Q[*t.s_next])] - Q[*s][a_index])
            else:
                QQ[*s][a_index] += alpha * (t.r + Q[*t.s_next][np.argmax(QQ[*t.s_next])] - QQ[*s][a_index])
            s = t.s_next

            episode_reward += t.r
            j += 1
            episode_time.append([i, j])
            if len(episode_time) ==  T:
                return episode_time, rewards

        rewards.append(episode_reward)
        action_history.append(action_counter)
    return episode_time, rewards, action_history


def run(runs, episodes, s_begin=(0, 2), s_end=(0, 0)):
    stats = np.zeros((runs, episodes))
    for method in [q_learning, double_q_learning]:
        for i in tqdm(range(runs)):
            Q, QQ = np.zeros((nrow, ncol, len(action_space))), np.zeros((nrow, ncol, len(action_space)))
            action_history = method(n=episodes, alpha=.1, epsilon=.1, Q=Q, action_space=action_space, nrow=nrow, ncol=ncol, p=p, T=episodes*1000, s_begin=s_begin, s_end=s_end, QQ=QQ)[2]

            left_cnt = np.array([action_history[j][(0, 2)][0] for j in range(episodes)]).cumsum()
            right_cnt = np.array([action_history[j][(0, 2)][1] for j in range(episodes)]).cumsum()
            stats[i] = left_cnt / (left_cnt + right_cnt)
                
        plt.plot(range(1, episodes+1),  stats.mean(axis=0), label=method.__name__)
    plt.legend()
    plt.show()


run(10000, 300)
