from utils.gridworld_util import *
from utils.tabular_util import *

nrow, ncol = 7, 10
wind = [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]

def default_row_stochastic():
    return 0


def sarsa(n, alpha, epsilon, *, V, action_space, nrow, ncol, T, row_stochastic):
    episode_time = []
    t = 0
    for i in range(n):
        s = (3, 0)
        a = greedy(s, epsilon=epsilon, V=V, action_space=action_space)
        while s != (3, 7):
            r = -1
            row, col = s[0] + action_space[a][0] + wind[s[1]] + row_stochastic(), s[1] + action_space[a][1]
            s_next = (int(np.clip(row, 0, nrow-1)), int(np.clip(col, 0, ncol-1)))
            a_next = greedy(s_next, epsilon=epsilon, V=V, action_space=action_space)

            V[*s][a] += alpha * (r + V[*s_next][a_next] - V[*s][a])

            t += 1
            episode_time.append([i, t])
            if len(episode_time) ==  T:
                return episode_time

            s, a = s_next, a_next
        
        print(i, t)
    return episode_time


def run(V, action_space, row_stochastic=default_row_stochastic):
    episode_time = sarsa(200, alpha=.5, epsilon=.1, V=V, action_space=action_space, nrow=nrow, ncol=ncol, T=8000, row_stochastic=row_stochastic)

    # print(episode_time)
    plt.plot(np.array(episode_time)[:,1], np.array(episode_time)[:,0])
    plt.show()


if __name__ == '__main__':
    V = np.zeros((nrow, ncol, 4))
    run(V, action_space)
