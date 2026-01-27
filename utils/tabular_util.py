from utils.util import *


def default_stochastic_state_rewards(s, *, nrow, ncol, prob_threshold=1e-3):
    return [Transition(s, 0, 1)]


def default_valid_action(s, a, *, nrow, ncol):
    return True


def print_policy(policy, *, state_space, action_space, action_name, type='dataframe'):
    nrow = len(policy)
    ncol = len(policy[0])
    
    if nrow > 1:
        string_policy = [['' for _ in range(ncol)] for _ in range(nrow)]
        for i in range(nrow):
            for j in range(ncol):
                string_policy[i][j] = ''.join([str(action_name[action_space[index]]) for index, prob in enumerate(policy[i][j]) if prob > 0])
        # print(string_policy)
        print_matrix(np.array(string_policy), type=type)
    else:
        for s in state_space:
            print(s, np.where(policy[0][s[1]] == policy[0][s[1]].max())[0])

        plt.plot([s[1] for s in state_space], [np.random.choice(np.where(policy[0][s[1]] == policy[0][s[1]].max())[0]) for s in state_space])
        plt.show()


def policy_countour(policy, *, action_space, action_name):
    nrow = len(policy)
    ncol = len(policy[0])
    X, Y = np.meshgrid(np.arange(nrow), np.arange(ncol))
    Z = np.array([np.random.choice([action_name[action_space[index]] for index, prob in enumerate(policy[i][j]) if prob > 0]) for i in range(nrow) for j in range(ncol)]).reshape((nrow, ncol))
    plt.contour(X, Y, Z, levels=len(np.unique(Z)))
    plt.show()


def policy_surf(policy, *, action_space, action_name):
    nrow = len(policy)
    ncol = len(policy[0])
    X, Y = np.meshgrid(np.arange(nrow), np.arange(ncol))
    Z = np.array([np.random.choice([action_name[action_space[index]] for index, prob in enumerate(policy[i][j]) if prob > 0]) for i in range(nrow) for j in range(ncol)]).reshape((nrow, ncol))
    plt.figure().add_subplot(111, projection='3d').plot_surface(X, Y, Z)
    plt.show()


def v_expected_update(s, *, nrow, ncol, γ, p, pi, action_space, V, acc_prob, prob_threshold):
    return sum([a_prob * q_expected_update_by_v(s, action_space[a_index], nrow=nrow, ncol=ncol, γ=γ, p=p, V=V, acc_prob=acc_prob*a_prob, prob_threshold=prob_threshold) for a_index, a_prob in enumerate(pi(s)) if a_prob > prob_threshold])


def v_optimal_update(s, *, nrow, ncol, γ, p, pi, action_space, V, acc_prob, prob_threshold):
    return max([q_expected_update_by_v(s, action_space[a_index], nrow=nrow, ncol=ncol, γ=γ, p=p, V=V, acc_prob=acc_prob*a_prob, prob_threshold=prob_threshold) for a_index, a_prob in enumerate(pi(s)) if a_prob > prob_threshold])


def q_expected_update_by_v(s, a, *, nrow, ncol, γ, p, V, acc_prob, prob_threshold):
    return sum([t.prob * (t.r + γ * V[*t.s_next]) for t in p(s, a, nrow=nrow, ncol=ncol) if acc_prob * t.prob > prob_threshold])


def greedy(*, q, epsilon, action_space):
    a_max_index = np.random.choice(np.where(q == q.max())[0])
    a_dist = np.ones(len(action_space)) * epsilon / len(action_space)
    a_dist[a_max_index] += 1 - epsilon
    a_index = np.random.choice(range(len(action_space))) if np.random.random() < epsilon else a_max_index
    return a_index, a_dist
