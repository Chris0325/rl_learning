from utils.util import *

state_space = ['A', 'B', 'C', 'D', 'E']


def sample_episode():
    trajectory = [3]
    while trajectory[-1] not in [0, 6]:
        s_next = trajectory[-1] + np.random.choice([-1, 1])
        trajectory.extend([(1 if s_next == 6 else 0), s_next])
    return trajectory


def td0(n, a):
    Vs = np.zeros((n, 7))
    V = np.ones(7) / 2
    V[0] = V[6] = 0

    for i in range(0, n):
        for s, r, s_next in windowed(sample_episode(), 3, step=2):
            V[s] += a *(r + V[s_next] - V[s])
        
        Vs[i] = V

    return Vs


def mc(n, a):
    Vs = np.zeros((n, 7))
    V = np.ones(7) / 2
    V[0] = V[6] = 0

    for i in range(0, n):
        episode = sample_episode()
        for s in episode[:-1:2]:
            V[s] += a * (episode[-2] - V[s])
        Vs[i] = V

    return Vs


def rms(V):
    return np.sqrt(np.mean(np.square(V - np.arange(1, 6) / 6)))


def run1():
    Vs = td0(n=1001, a=.1)
    for i in [0, 1, 10, 100, 1000]:
        plt.plot(state_space, Vs[i][1:6], 'o-', label=i)
    plt.plot(state_space, np.arange(1, 6) / 6, 'o-', label='true')
    plt.legend()
    plt.show()


def run2():
    for type, a, func in [
        ('mc', .01, mc),
        ('mc', .02, mc),
        ('mc', .03, mc),
        ('mc', .04, mc),
        ('td0', .05, td0),
        ('td0', .1, td0),
        ('td0', .15, td0),
    ]:
        Vs = np.zeros((100, 100))
        for i in range(100):
            Vs[i] = np.apply_along_axis(rms, axis=1, arr=func(n=101, a=a)[1:,1:6])
        plt.plot(range(1, 101), np.mean(Vs, axis=0), label=f'{type}{a}')
    plt.legend()
    plt.show()

run1()
run2()
