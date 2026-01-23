from section_6_2_td0_randomwalk import *


def batch_sample_method(n, a, error):
    Vs = np.zeros((n, 7))
    V = np.ones(7) / 2
    V[0] = V[6] = 0

    history_episodes = []
    for i in range(0, n):
        Vs[i] = V

        history_episodes.append(sample_episode())

        for _ in range(1000):
            increment = np.zeros_like(V)
            for episode in history_episodes:
                for s, r, s_next in windowed(episode, 3, step=2):
                    increment[s] = a * error(V, s, r, s_next, episode)

            V += increment
            if np.abs(increment.max()) < 1e-4:
                break

    return Vs


for type, a, error in [('mc', .001, mc_error), ('td0', .001, td_error)]:
    Vs = np.zeros((100, 100))
    for i in tqdm(range(100)):
        Vs[i] = np.apply_along_axis(rms, axis=1, arr=batch_sample_method(n=101, a=a, error=error)[1:,1:6])
    plt.plot(range(1, 101), np.mean(Vs, axis=0), label=f'{type}{a}')

plt.legend()
plt.show()
