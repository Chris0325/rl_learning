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
    print(policy)
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
