from section_4_3_example_jack_car_rental import *


def p(s, a, *, nrow, ncol):
    row, col = s
    row -= a[0]
    col -= a[1]

    row = int(np.clip(row, 0, nrow-1))
    col = int(np.clip(col, 0, ncol-1))

    # A -> B one car free
    transfer_reward = (2 * a[0] if a[0] <=0 else -2 * (a[0] - 1))
    # parking fee
    park_reward = -4 * ((row > 10) + (col > 10))

    return [((row, col), transfer_reward + park_reward, 1)]


policy = np.outer(np.ones(n**2), np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])).reshape((n, n, 11))

V, policy = state_policy_iteration(policy=policy, nrow=n, ncol=n, γ=0.9, p=p, state_space=state_space, action_space=action_space, action_name=action_name, valid_action=valid_action, stochastic_state_rewards=stochastic_state_rewards, max_iterations=10, prob_threshold=1e-4)

policy_countour(policy, action_space=action_space, action_name=action_name)
policy_surf(policy, action_space=action_space, action_name=action_name)
