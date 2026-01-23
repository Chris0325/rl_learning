from utils.util import *

state_space = ['A', 'B', 'C', 'D', 'E']


def sample_episode():
    trajectory = [3]
    while trajectory[-1] not in [0, 6]:
        s_next = trajectory[-1] + np.random.choice([-1, 1])
        trajectory.extend([(1 if s_next == 6 else 0), s_next])
    return trajectory


def sample_method(n, a, error):
    Vs = np.zeros((n, 7))
    V = np.ones(7) / 2
    V[0] = V[6] = 0

    for i in range(0, n):
        Vs[i] = V
        episode = sample_episode()
        for s, r, s_next in windowed(episode, 3, step=2):
            V[s] += a * error(V, s, r, s_next, episode)
    return Vs


def td_error(V, s, r, s_next, episode):
    return r + V[s_next] - V[s]


def mc_error(V, s, r, s_next, episode):
    return episode[-2] - V[s]


def rms(V):
    return np.sqrt(np.mean(np.square(V - np.arange(1, 6) / 6)))


def run1():
    Vs = sample_method(n=1001, a=.1, error=td_error)
    for i in [0, 1, 10, 100, 1000]:
        plt.plot(state_space, Vs[i][1:6], 'o-', label=i)
    plt.plot(state_space, np.arange(1, 6) / 6, 'o-', label='true')
    plt.legend()
    plt.show()


def run2():
    for type, a, error in [
        ('mc', .01, mc_error),
        ('mc', .02, mc_error),
        ('mc', .03, mc_error),
        ('mc', .04, mc_error),
        ('td0', .05, td_error),
        ('td0', .1, td_error),
        ('td0', .15, td_error),
    ]:
        Vs = np.zeros((100, 100))
        for i in range(100):
            Vs[i] = np.apply_along_axis(rms, axis=1, arr=sample_method(n=101, a=a, error=error)[1:,1:6])
        plt.plot(range(1, 101), np.mean(Vs, axis=0), label=f'{type}{a}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run1()
    run2()
