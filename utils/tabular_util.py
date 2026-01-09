import numpy as np
from tqdm import tqdm

from utils.util import *


def default_stochastic_state(s, *, nrow, ncol, prob_threshold=1e-3):
    # state, prob, reward
    return [(s, 1, 0)]


def default_valid_action(s, a):
    return True


def analytical_state_value(*, nrow, ncol, γ, p, pi, state_space, action_space, stochastic_state=default_stochastic_state, round=1, prob_threshold=1e-3):
    A = np.eye(nrow * ncol)
    b = np.zeros(nrow * ncol)
    for s in state_space:
        for s_s, s_prob, s_r in stochastic_state(s, nrow=nrow, ncol=ncol, prob_threshold=prob_threshold):
            if s_prob > prob_threshold:
                b[coordinate_to_index(s, ncol=ncol)] += s_prob * s_r
                for a_index, prob in enumerate(pi(s_s)):
                    if prob > prob_threshold:
                        s_next, r = p(s_s, action_space[a_index], nrow=nrow, ncol=ncol)
                        A[coordinate_to_index(s, ncol=ncol), coordinate_to_index(s_next, ncol=ncol)] -= s_prob * prob * γ
                        b[coordinate_to_index(s, ncol=ncol)] += s_prob * prob * r
    # print_matrix(A, nrow=nrow*ncol, ncol=nrow*ncol, round=2)

    return np.linalg.solve(A, b).round(round).reshape((nrow, ncol))


def iterative_state_value(*, nrow, ncol, γ, p, pi, state_space, action_space, stochastic_state=default_stochastic_state, round=1, θ=1e-5, max_iterations=1000, V=None, prob_threshold=1e-3):
    if V is None:
        V = np.zeros((nrow, ncol))

    for _ in tqdm(range(max_iterations), desc='Policy Evaluation'):
        Δ = 0
        for s in state_space:
            v, V[*s] = V[*s], 0
            for s_s, s_prob, s_r in stochastic_state(s, nrow=nrow, ncol=ncol, prob_threshold=prob_threshold):
                if s_prob > prob_threshold:
                    V[*s] += s_prob * s_r
                    for a_index, prob in enumerate(pi(s_s)):
                        if prob > prob_threshold:
                            s_next, r = p(s_s, action_space[a_index], nrow=nrow, ncol=ncol)
                            V[*s] += s_prob * prob * (r + γ * V[*s_next])
            Δ = max(Δ, abs(V[s] - v))
        if Δ < θ:
            break
    return V.round(round)


def value_policy(V, *, nrow, ncol, γ, p, state_space, action_space, valid_action):
    policy = np.zeros((nrow, ncol, len(action_space)))

    for s in state_space:
        q_values = []
        for a in action_space:
            if valid_action(s, a):
                s_next, r = p(s, a, nrow=nrow, ncol=ncol)
                q_values.append(r + γ * V[*s_next])
            else:
                q_values.append( -float('inf'))
        q_values =  np.array(q_values)
        policy[s[0]][s[1]] = np.where(q_values == q_values.max(), 1, 0)

    return policy


def policy_iteration(*, policy, nrow, ncol, γ, p, state_space, action_space, valid_action, stochastic_state=default_stochastic_state, θ=1e-5, max_iterations=100, max_evaluation_iterations=100):
    V = np.zeros((nrow, ncol))

    for _ in tqdm(range(max_iterations), desc='Policy Iteration'):
        V = iterative_state_value(nrow=nrow, ncol=ncol, γ=γ, p=p, pi=lambda s: policy[s[0]][s[1]], state_space=state_space, action_space=action_space, stochastic_state=stochastic_state, θ=θ, V=V, max_iterations=max_evaluation_iterations)
        new_policy = value_policy(V, nrow=nrow, ncol=ncol, γ=γ, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action)
        if np.allclose(new_policy, policy):
            break
        policy = new_policy

    return V, policy


def print_policy(policy, *, action_space, action_name):
    nrow = len(policy)
    ncol = len(policy[0])
    
    string_policy = [['' for _ in range(ncol)] for _ in range(nrow)]

    for i in range(nrow):
        for j in range(ncol):
            string_policy[i] [j] = ''.join([str(action_name[action_space[index]]) for index, prob in enumerate(policy[i][j]) if prob > 0])

    print(string_policy)
    print_matrix(np.array(string_policy))
