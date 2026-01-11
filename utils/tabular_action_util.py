from utils.tabular_util import *


def iterative_action_value(*, nrow, ncol, γ, p, pi, state_space, action_space, valid_ation=default_valid_action, stochastic_state_rewards=default_stochastic_state_rewards, round=1, θ=1e-5, max_iterations=1000, Q=None, prob_threshold=1e-3):
    if Q is None:
        Q = np.zeros((nrow, ncol, len(action_space)))

    for _ in tqdm(range(max_iterations), desc='Action Policy Evaluation'):
        Δ = 0
        for s in state_space:
            for a_index in range(len(action_space)):
                a = action_space[a_index]
                if valid_ation(s, a):
                    q, Q[*s, a_index] = Q[*s, a_index], 0
                    s_next, Q[*s, a_index] = p(s, a, nrow=nrow, ncol=ncol)
                    for s_s, s_r, s_prob in stochastic_state_rewards(s_next, nrow=nrow, ncol=ncol, prob_threshold=prob_threshold):
                        Q[*s, a_index] += s_prob * (s_r + γ * np.dot(Q[*s_s], pi(s_s)))
                    Δ = max(Δ, abs(Q[*s, a_index] - q))
        if Δ < θ:
            break
    return Q.round(round)


def action_policy_iteration(*, policy, nrow, ncol, γ, p, state_space, action_space, action_name, valid_action=default_valid_action, stochastic_state_rewards=default_stochastic_state_rewards, θ=1e-3, max_iterations=100, max_evaluation_iterations=100, prob_threshold=1e-3):
    Q = np.zeros((nrow, ncol, len(action_space)))
    print_policy(policy, action_space=action_space, action_name=action_name)

    for _ in tqdm(range(max_iterations), desc='Action Policy Iteration'):
        Q = iterative_action_value(nrow=nrow, ncol=ncol, γ=γ, p=p, pi=lambda s: policy[s[0]][s[1]], state_space=state_space, action_space=action_space, stochastic_state_rewards=stochastic_state_rewards, θ=θ, Q=Q, max_iterations=max_evaluation_iterations, prob_threshold=prob_threshold)
        new_policy = action_policy(Q, nrow=nrow, ncol=ncol, state_space=state_space, action_space=action_space, valid_action=valid_action)
        if np.allclose(new_policy, policy):
            break
        policy = new_policy
        print_policy(policy, action_space=action_space, action_name=action_name)

    return Q, policy


def action_policy(Q, *, nrow, ncol, state_space, action_space, valid_action=default_valid_action):
    policy = np.zeros((nrow, ncol, len(action_space)))

    for s in state_space:
        q_values = []
        for a_index in range(len(action_space)):
            if valid_action(s, action_space[a_index]):
                q_values.append(Q[*s, a_index])
            else:
                q_values.append(-1e5)
        q_values = np.array(q_values)
        policy[s[0]][s[1]] = np.where(q_values == q_values.max(), 1, 0) / len(q_values)

    return policy
