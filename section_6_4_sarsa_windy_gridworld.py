from utils.gridworld_util import *
from utils.tabular_util import *

nrow, ncol = 7, 10
wind = [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]


def p(s, a, *, nrow, ncol):
    row, col = s[0] + a[0] + wind[s[1]], s[1] + a[1]
    s_next = (int(np.clip(row, 0, nrow-1)), int(np.clip(col, 0, ncol-1)))
    return [Transition(s_next, -1, 1)]


def sarsa(n, alpha, epsilon, *, Q, action_space, nrow, ncol, p, T, s_begin, s_ends, γ=1, QQ=None):
    episode_time, rewards = [], []
    j = 0
    for i in range(n):
        episode_reward = 0

        s = s_begin
        a_index = greedy(q=Q[*s], epsilon=epsilon, action_space=action_space)[0]
        while s not in s_ends:
            ts = p(s, action_space[a_index], nrow=nrow, ncol=ncol)
            t = np.random.choice(ts, p=[t.prob for t in ts])
            a_next_index = greedy(q=Q[*t.s_next], epsilon=epsilon, action_space=action_space)[0]
            Q[*s][a_index] += alpha * (t.r + γ * Q[*t.s_next][a_next_index] - Q[*s][a_index])
            s, a_index = t.s_next, a_next_index

            episode_reward += t.r
            j += 1
            episode_time.append([i, j])
            if len(episode_time) ==  T:
                return episode_time, rewards

        rewards.append(episode_reward)
        logging.info(f'episode {i}, time {j}, reward {episode_reward}')
    return episode_time, rewards


def run(Q, action_space, p=p, s_begin=(3, 0), s_ends=[(3, 7)]):
    episode_time = sarsa(200, alpha=.5, epsilon=.1, Q=Q, action_space=action_space, nrow=nrow, ncol=ncol, p=p, T=10000, s_begin=s_begin, s_ends=s_ends)[0]

    plt.plot(np.array(episode_time)[:,1], np.array(episode_time)[:,0])
    plt.show()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    Q = np.zeros((nrow, ncol, 4))
    run(Q, action_space)
