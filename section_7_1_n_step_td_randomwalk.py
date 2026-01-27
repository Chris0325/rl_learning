from section_7_1_randomwalk import *


def random_pi(s, *, action_space):
    return np.random.choice(len(action_space))


def rms(V):
    return np.sqrt(np.mean(np.square(V - np.arange(-9, 10) / 10)))


def n_step_td(n, V, alpha, pi, nrow, ncol, action_space, episodes, s_begin, s_ends, γ=1):
    states, rewards, scales = [1] * (n+1), np.zeros(n+1), γ ** np.arange(n+1)
    for _ in range(episodes):
        T, t = float('inf'), 0
        states[t] = s_begin
        while True:
            if t < T:
                a_index = pi(states[t % (n+1)], action_space=action_space)
                trs = p(states[t % (n+1)], action_space[a_index], nrow=nrow, ncol=ncol)
                tr = np.random.choice(trs, p=[tr.prob for tr in trs])
                if tr.s_next in s_ends:
                    T = t + 1

                states[(t+1) % (n+1)] = tr.s_next
                rewards[(t+1) % (n+1)] = tr.r
            
            tau = t - n + 1
            if tau >= 0:
                G = sum(scales[i] * rewards[j % (n+1)] for i, j in enumerate(range(tau+1, min(tau+n, T)+1)))
                if tau + n < T:
                    G += scales[n] * V[*states[(tau+n) % (n+1)]]

                V[*states[tau % (n+1)]] += alpha * (G - V[*states[tau % (n+1)]])
            
            t += 1
            if tau == T - 1:
                break


def run():
    for n in tqdm(2 ** np.arange(10)):
        errors = []
        for alpha in np.linspace(0, 1, 6).round(1):
            V = np.zeros((100, 1, 21))
            for i in range(100):
                n_step_td(n=n, V=V[i], alpha=alpha, pi=random_pi, nrow=1, ncol=21, action_space=action_space, episodes=10, s_begin=(0, 10), s_ends=[(0, 0), (0, 20)])

            errors.append(np.apply_along_axis(rms, axis=2, arr=V[..., 1:20]).mean())
        
        plt.plot(np.linspace(0, 1, 6).round(1), errors, label=f'n={n}')

    plt.legend()
    plt.show()


run()
