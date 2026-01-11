from utils.util import *


def default_stochastic_state_rewards(s, *, nrow, ncol, prob_threshold=1e-3):
    # state, reward, prob
    return [(s, 0, 1)]


def default_valid_action(s, a):
    return True


def print_policy(policy, *, action_space, action_name):
    nrow = len(policy)
    ncol = len(policy[0])
    
    string_policy = [['' for _ in range(ncol)] for _ in range(nrow)]

    for i in range(nrow):
        for j in range(ncol):
            string_policy[i] [j] = ''.join([str(action_name[action_space[index]]) for index, prob in enumerate(policy[i][j]) if prob > 0])

    # print(string_policy)
    print_matrix(np.array(string_policy))


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


def v_update(s, *, nrow, ncol, γ, p, pi, action_space, V, acc_prob, prob_threshold):
    return sum([a_prob * q_expected_update_by_v(s, action_space[a_index], nrow=nrow, ncol=ncol, γ=γ, p=p, V=V, acc_prob=acc_prob*a_prob, prob_threshold=prob_threshold) for a_index, a_prob in enumerate(pi(s)) if a_prob > prob_threshold])


def q_expected_update_by_v(s, a, *, nrow, ncol, γ, p, V, acc_prob, prob_threshold):
    return sum([prob * (r + γ * V[*s_next]) for s_next, r, prob in p(s, a, nrow=nrow, ncol=ncol) if acc_prob * prob > prob_threshold])
