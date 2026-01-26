from utils.tabular_util import *


def analytical_state_value(*, nrow, ncol, γ, p, pi, state_space, action_space, stochastic_state_rewards=default_stochastic_state_rewards, round=1, prob_threshold=1e-3):
    A = np.eye(nrow * ncol)
    b = np.zeros(nrow * ncol)
    for s in state_space:
        for st in stochastic_state_rewards(s, nrow=nrow, ncol=ncol, prob_threshold=prob_threshold):
            if st.prob > prob_threshold:
                b[coordinate_to_index(s, ncol=ncol)] += st.prob * st.r
                for a_index, a_prob in enumerate(pi(st.s)):
                    if st.prob * a_prob > prob_threshold:
                        for t in p(st.s, action_space[a_index], nrow=nrow, ncol=ncol):
                            if st.prob * a_prob * t.prob > prob_threshold:
                                A[coordinate_to_index(s, ncol=ncol), coordinate_to_index(t.s, ncol=ncol)] -= st.prob * a_prob * t.prob * γ
                                b[coordinate_to_index(s, ncol=ncol)] += st.prob * a_prob * t.prob * t.r
    # print_matrix(A, nrow=nrow*ncol, ncol=nrow*ncol, round=2)

    return np.linalg.solve(A, b).round(round).reshape((nrow, ncol))


def iterative_state_value(*, nrow, ncol, γ, p, pi, state_space, action_space, stochastic_state_rewards=default_stochastic_state_rewards, round=3, θ=1e-5, max_iterations=1000, V=None, prob_threshold=1e-3):
    if V is None:
        V = np.zeros((nrow, ncol))

    for _ in tqdm(range(max_iterations), desc='State Policy Evaluation'):
        Δ = 0
        for s in state_space:
            v, V[*s] = V[*s], 0
            for st in stochastic_state_rewards(s, nrow=nrow, ncol=ncol, prob_threshold=prob_threshold):
                if st.prob > prob_threshold:
                    V[*s] += st.prob * (st.r + v_expected_update(st.s, nrow=nrow, ncol=ncol, γ=γ, p=p, pi=pi, action_space=action_space, V=V, acc_prob=st.prob, prob_threshold=prob_threshold))
            Δ = max(Δ, abs(V[s] - v))
        if Δ < θ:
            break

    return V.round(round)


def value_policy(V, *, nrow, ncol, γ, p, state_space, action_space, valid_action=default_valid_action):
    policy = np.zeros((nrow, ncol, len(action_space)))

    for s in state_space:
        q_values = []
        for a in action_space:
            if valid_action(s, a, nrow=nrow, ncol=ncol):
                q_values.append(q_expected_update_by_v(s, a, nrow=nrow, ncol=ncol, γ=γ, p=p, V=V, acc_prob=1, prob_threshold=0))
            else:
                q_values.append(-1e10)
        q_values = np.array(q_values)
        policy[s[0]][s[1]] = np.where(q_values == q_values.max(), 1, 0) / len(q_values)

    return policy


def state_policy_iteration(policy, V=None, *, nrow, ncol, γ, p, state_space, action_space, action_name, valid_action=default_valid_action, stochastic_state_rewards=default_stochastic_state_rewards, θ=1e-3, max_iterations=100, max_evaluation_iterations=100, prob_threshold=1e-3, plot_policy=print_policy, round=3):
    if V is None:
        V = np.zeros((nrow, ncol))
    plot_policy(policy, state_space=state_space, action_space=action_space, action_name=action_name)

    for _ in tqdm(range(max_iterations), desc='State Policy Iteration'):
        V = iterative_state_value(nrow=nrow, ncol=ncol, γ=γ, p=p, pi=lambda s: policy[s[0]][s[1]], state_space=state_space, action_space=action_space, stochastic_state_rewards=stochastic_state_rewards, θ=θ, V=V, max_iterations=max_evaluation_iterations, prob_threshold=prob_threshold, round=round)
        new_policy = value_policy(V, nrow=nrow, ncol=ncol, γ=γ, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action)
        if np.allclose(new_policy, policy):
            break
        policy = new_policy
        plot_policy(policy, state_space=state_space, action_space=action_space, action_name=action_name)

    return V, policy


def state_value_iteration(V, *, nrow, ncol, γ, p, state_space, action_space, action_name, valid_action=default_valid_action, stochastic_state_rewards=default_stochastic_state_rewards, θ=1e-3, max_iterations=100, max_evaluation_iterations=100, prob_threshold=1e-3, plot_policy=print_policy):
    policy = value_policy(V, nrow=nrow, ncol=ncol, γ=γ, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action)

    for _ in tqdm(range(max_iterations), desc='State Value Iteration'):
        Δ = 0
        for s in state_space:
            v = V[*s]
            V[*s] = v_optimal_update(s, nrow=nrow, ncol=ncol, γ=γ, p=p, pi=lambda s: policy[s[0]][s[1]], action_space=action_space, V=V, acc_prob=1, prob_threshold=prob_threshold)
            Δ = max(Δ, abs(V[*s] - v))
        
        if Δ < θ:
            break
    
    policy = value_policy(V, nrow=nrow, ncol=ncol, γ=γ, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action)
    plot_policy(policy, state_space=state_space, action_space=action_space, action_name=action_name)

    return V, policy
