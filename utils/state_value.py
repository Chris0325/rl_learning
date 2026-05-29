from math import tau

from utils.tabular_util import *


class StateValue:

    def __init__(self, *, size, tau, state_space, action_space, action_name=None):
        self.size, self.tau = size, tau
        self.state_space, self.action_space, self.action_name = state_space, action_space, action_name
        self.ignore_prob = 1e-3

    def state_evolve(self, s):
        return [Transition(s, 0, 1)]

    def p(self, s, a):
        ...

    def pi(self, s):
        return np.ones(len(self.action_space)) / len(self.action_space)

    def q_update_by_v(self, s, a, *, V, base_prob=1):
        return sum([tr.prob * (tr.r + self.tau * V[*tr.s]) for tr in self.p(s, a) if base_prob * tr.prob > self.ignore_prob])

    def expected_update(self, s, *, V, base_prob=1):
        return sum([a_prob * self.q_update_by_v(s, self.action_space[a_index], V=V, base_prob=base_prob*a_prob) for a_index, a_prob in enumerate(self.pi(s)) if a_prob > self.ignore_prob])

    def s_has_a(self, s, a):
        return True

    def value_policy(self, V, tol=.01):
        policy = np.zeros((self.size[0], self.size[1], len(self.action_space)))

        for s in self.state_space:
            q_values = []
            for a in self.action_space:
                if self.s_has_a(s, a):
                    q_values.append(self.q_update_by_v(s, a, V=V))
                else:
                    q_values.append(-1e10)
            q_values = np.array(q_values)
            policy[s[0]][s[1]] = np.where(q_values > q_values.max()-tol, 1, 0) / len(q_values)

        return policy

    def print_policy(self, policy, type='dataframe'):
        nrow = len(policy)
        ncol = len(policy[0])
        
        if nrow > 1:
            string_policy = [['' for _ in range(ncol)] for _ in range(nrow)]
            for i in range(nrow):
                for j in range(ncol):
                    string_policy[i][j] = ''.join([str(self.action_name[self.action_space[index]]) for index, prob in enumerate(policy[i][j]) if prob > 0])
            # print(string_policy)
            print_matrix(np.array(string_policy), type=type)
        else:
            for s in self.state_space:
                print(s, np.where(policy[0][s[1]] == policy[0][s[1]].max())[0])

            plt.plot([s[1] for s in self.state_space], [np.random.choice(np.where(policy[0][s[1]] == policy[0][s[1]].max())[0]) for s in self.state_space])
            plt.show()

class AnalyticalStateValue(StateValue):

    def __call__(self, round=2):
        A = np.eye(self.size[0] * self.size[1])
        b = np.zeros(self.size[0] * self.size[1])
        for s in self.state_space:
            s_row = to_index(s, size=self.size)

            for s_tr in self.state_evolve(s):
                if s_tr.prob > self.ignore_prob:
                    b[s_row] += s_tr.prob * s_tr.r

                    for a_index, a_prob in enumerate(self.pi(s_tr.s)):
                        if s_tr.prob * a_prob > self.ignore_prob:

                            for a_tr in self.p(s_tr.s, self.action_space[a_index]):
                                if s_tr.prob * a_prob * a_tr.prob > self.ignore_prob:

                                    A[s_row, to_index(a_tr.s, size=self.size)] -= s_tr.prob * a_prob * a_tr.prob * self.tau
                                    b[s_row] += s_tr.prob * a_prob * a_tr.prob * a_tr.r

        return la.solve(A, b).reshape(self.size).round(round)


class IterativeStateValue(StateValue):
    def __call__(self, theta=1e-3, max_iterations=1000, round=4, V=None):
        if V is None:
            V = np.zeros(self.size)

        for _ in tqdm(range(max_iterations), desc='State Policy Evaluation'):
            delta = 0
            for s in self.state_space:
                v = V[*s]
                V[*s] = sum([s_tr.prob * (s_tr.r + self.expected_update(s, V=V, base_prob=s_tr.prob)) for s_tr in self.state_evolve(s) if s_tr.prob > self.ignore_prob])
                delta = max(delta, abs(V[*s] - v))
            if delta < theta:
                break

        return V.round(round)


# def state_policy_iteration(policy, V=None, *, nrow, ncol, tau, p, state_space, action_space, action_name, valid_action=default_valid_action, stochastic_state_rewards=default_stochastic_state_rewards, theta=1e-3, max_iterations=100, max_evaluation_iterations=100, prob_threshold=1e-3, plot_policy=print_policy, round=3):
#     if V is None:
#         V = np.zeros((nrow, ncol))
#     plot_policy(policy, state_space=state_space, action_space=action_space, action_name=action_name)

#     for _ in tqdm(range(max_iterations), desc='State Policy Iteration'):
#         V = iterative_state_value(nrow=nrow, ncol=ncol, tau=tau, p=p, pi=lambda s: policy[s[0]][s[1]], state_space=state_space, action_space=action_space, stochastic_state_rewards=stochastic_state_rewards, theta=theta, V=V, max_iterations=max_evaluation_iterations, prob_threshold=prob_threshold, round=round)
#         new_policy = value_policy(V, nrow=nrow, ncol=ncol, tau=tau, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action)
#         if np.allclose(new_policy, policy):
#             break
#         policy = new_policy
#         plot_policy(policy, state_space=state_space, action_space=action_space, action_name=action_name)

#     return V, policy


# def state_value_iteration(V, *, nrow, ncol, tau, p, state_space, action_space, action_name, valid_action=default_valid_action, stochastic_state_rewards=default_stochastic_state_rewards, theta=1e-3, max_iterations=100, max_evaluation_iterations=100, prob_threshold=1e-3, plot_policy=print_policy):
#     policy = value_policy(V, nrow=nrow, ncol=ncol, tau=tau, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action)

#     for _ in tqdm(range(max_iterations), desc='State Value Iteration'):
#         delta = 0
#         for s in state_space:
#             v = V[*s]
#             V[*s] = v_optimal_update(s, nrow=nrow, ncol=ncol, tau=tau, p=p, pi=lambda s: policy[s[0]][s[1]], action_space=action_space, V=V, acc_prob=1, prob_threshold=prob_threshold)
#             delta = max(delta, abs(V[*s] - v))
        
#         if delta < theta:
#             break
    
#     policy = value_policy(V, nrow=nrow, ncol=ncol, tau=tau, p=p, state_space=state_space, action_space=action_space, valid_action=valid_action)
#     plot_policy(policy, state_space=state_space, action_space=action_space, action_name=action_name)

#     return V, policy
