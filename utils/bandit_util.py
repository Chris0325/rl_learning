import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import softmax
from collections import defaultdict

np.random.seed(0)


class Bandit:

    def __init__(self, k, loc=0):
        self.k, self.means = k, np.random.normal(loc=loc, scale=1, size=k)
        self.variance = 1
    
    def evolve(self):
        ...

    def reward(self, i):
        reward = np.random.normal(loc=self.means[i], scale=1)
        self.evolve()
        return reward


class NonstationaryBandit(Bandit):

    def __init__(self, k, loc=0):
        super().__init__(k, loc)
        self.means = np.zeros(k)

    def evolve(self):
        self.means += np.random.normal(loc=0, scale=.01, size=self.k)


class EpsilonGreedyPolicy:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return f'e={self.epsilon}'

    def choose(self, action_value, action_count=None, t=None):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(action_value))
        else:
            return np.random.choice(np.where(action_value == action_value.max())[0])


class UpperConfidenceBoundPolicy:

    def __init__(self, c):
        self.c = c

    def __str__(self):
        return f'ucb,c={self.c}'

    def choose(self, action_value, action_count, t):
        ucb_value = action_value + self.c * np.sqrt(np.log(t + 1) / (action_count + 1e-8))
        return np.random.choice(np.where(ucb_value == ucb_value.max())[0])


class SlowSampleAverageValue:

    def __init__(self, horizon=1000, init_value=0):
        self.horizon, self.init_value = horizon, init_value

    def __str__(self):
        return 'ssa'

    def run(self, bandit, policy, need_best_ratio=True):
        best_actions = np.where(bandit.means == bandit.means.max())[0]
        action_history, reward_history, action_rewards, action_count = [], [], defaultdict(list), np.zeros(bandit.k)
        for t in range(self.horizon):
            action_value = np.array([np.array(action_rewards[i]).mean() if action_rewards[i] else self.init_value for i in range(bandit.k)])
            action = policy.choose(action_value, action_count, t)
            action_history.append(action)

            reward = bandit.reward(action)
            reward_history.append(reward)

            action_count[action] += 1
            action_rewards[action].append(reward)
        
        best_ratio = None
        if need_best_ratio:
            action_history = np.array(action_history)
            best_hits = np.isin(action_history, best_actions).cumsum()
            best_ratio = [count / (i + 1) for i, count in enumerate(best_hits)]
        return best_ratio, reward_history


class SampleAverageValue(SlowSampleAverageValue):

    def __str__(self):
        return 'sa'

    def run(self, bandit, policy, need_best_ratio=True):
        best_actions = np.where(bandit.means == bandit.means.max())[0]
        action_history, reward_history = [], []
        action_value, action_count = np.ones(bandit.k) * self.init_value, np.zeros(bandit.k)

        for t in range(self.horizon):
            action = policy.choose(action_value, action_count, t)
            action_history.append(action)

            reward = bandit.reward(action)
            reward_history.append(reward)

            action_count[action] += 1
            action_value[action] += 1 / action_count[action] * (reward - action_value[action])

        best_ratio = None
        if need_best_ratio:
            action_history = np.array(action_history)
            best_hits = np.isin(action_history, best_actions).cumsum()
            best_ratio = [count / (i + 1) for i, count in enumerate(best_hits)]
        return best_ratio, reward_history


class ConstantStepValue(SlowSampleAverageValue):

    def __init__(self, a, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a

    def __str__(self):
        return f'cs,a={self.a}'

    def run(self, bandit, policy, need_best_ratio=False):
        action_history, reward_history = [], []
        action_value, action_count = np.ones(bandit.k) * self.init_value, np.zeros(bandit.k)

        for t in range(self.horizon):
            action = policy.choose(action_value, action_count, t)
            action_history.append(action)

            reward = bandit.reward(action)
            reward_history.append(reward)

            action_count[action] += 1
            action_value[action] += self.a * (reward - action_value[action])

        return None, reward_history


class ConstantStepUnbiasValue(ConstantStepValue):

    def __str__(self):
        return f'csub,a={self.a}'

    def run(self, bandit, policy, need_best_ratio=False):
        action_history, reward_history = [], []
        action_value, action_count = np.ones(bandit.k) * self.init_value, np.zeros(bandit.k)

        demominator = 0
        for t in range(self.horizon):
            action = policy.choose(action_value, action_count, t)
            action_history.append(action)

            reward = bandit.reward(action)
            reward_history.append(reward)

            demominator += self.a * (1 - demominator)
            action_value[action] += self.a / demominator * (reward - action_value[action])

        return None, reward_history


class GradientValue(ConstantStepValue):

    def __str__(self):
        return f'g,a={self.a}'

    def run(self, bandit, policy='w_b', need_best_ratio=False):
        action_history, reward_history = [], []
        preferences = np.zeros(bandit.k)
        avg_reward = 0

        for t in range(self.horizon):
            action_prob = softmax(preferences)
            action = np.random.choice(bandit.k, p=action_prob)
            action_history.append(action)

            reward = bandit.reward(action)
            reward_history.append(reward)

            avg_reward += 1 / (t + 1) * (reward - avg_reward)
            if policy != 'w_b':
                avg_reward = 0

            for i in range(bandit.k):
                if i == action:
                    preferences[i] += self.a * (reward - avg_reward) * (1 - action_prob[i])
                else:
                    preferences[i] -= self.a * (reward - avg_reward) * action_prob[i]

        return None, reward_history


def plan_to_str(value, policy):
    return f'{str(value)},{str(policy)}'


def run_plans(k, bandit_class, plans, runs, bandit_mean=0, need_best_ratio=True):
    ratio_stats, reward_stats = defaultdict(list), defaultdict(list)
    for value, policy in plans:
        for _ in tqdm(range(runs), desc=plan_to_str(value, policy)):
            bandit = bandit_class(k, loc=bandit_mean)
            best_ratio, reward_history = value.run(bandit, policy, need_best_ratio)
            ratio_stats[plan_to_str(value, policy)].append(best_ratio)
            reward_stats[plan_to_str(value, policy)].append(reward_history)
    return ratio_stats, reward_stats


def experiment(k, bandit_class, plans, runs, bandit_mean=0, need_best_ratio=True):
    ratio_stats, reward_stats = run_plans(k, bandit_class, plans, runs, bandit_mean, need_best_ratio)

    plt.subplot(2, 1, 1)
    for value, policy in plans:
        plt.plot(np.array(reward_stats[plan_to_str(value, policy)]).mean(axis=0), label=plan_to_str(value, policy))
    plt.legend()

    if need_best_ratio:
        plt.subplot(2, 1, 2)
        for value, policy in plans:
            plt.plot(np.array(ratio_stats[plan_to_str(value, policy)]).mean(axis=0), label=plan_to_str(value, policy))

    plt.legend()
    plt.show()


def banchmark(k, bandit_class, plans, runs, bandit_mean=0, need_best_ratio=False, from_step=0):
    banchmark_dict = defaultdict(list)
    for plan_name, plan_settings in plans.items():
        for parameter, (value, policy) in plan_settings:
            ratio_stats, reward_stats = run_plans(k, bandit_class, [(value, policy)], runs, bandit_mean, need_best_ratio)
            banchmark_dict[plan_name].append([parameter, np.array(reward_stats[plan_to_str(value, policy)][from_step:]).mean()])

    for plan_name in plans:
        plt.plot(np.array(banchmark_dict[plan_name])[:, 0], np.array(banchmark_dict[plan_name])[:, 1], label=plan_name)

    plt.legend()
    plt.show()
