import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(0)


class Bandit:

    def __init__(self, k):
        self.k, self.means = k, np.random.randn(k)
        self.variance = 1
    
    def evolve(self):
        ...

    def reward(self, i):
        reward = np.random.normal(loc=self.means[i], scale=1)
        self.evolve()
        return reward


class NonstationaryBandit(Bandit):

    def __init__(self, k):
        super().__init__(k)
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


class ConstantStepUnbiasValue(SlowSampleAverageValue):

    def __init__(self, a, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a

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


def experiment(k, bandit_class, combinations, runs, need_best_ratio=True):
    def combination_to_str(value, policy):
        return f'{str(value)},{str(policy)}'

    ratio_stats, reward_stats = defaultdict(list), defaultdict(list)
    for value, policy in combinations:
        for _ in tqdm(range(runs), desc=str(policy)):
            bandit = bandit_class(k)
            best_ratio, reward_history = value.run(bandit, policy, need_best_ratio)
            ratio_stats[combination_to_str(value, policy)].append(best_ratio)
            reward_stats[combination_to_str(value, policy)].append(reward_history)

    plt.subplot(2, 1, 1)
    for value, policy in combinations:
        plt.plot(np.array(reward_stats[combination_to_str(value, policy)]).mean(axis=0), label=combination_to_str(value, policy))
    plt.legend()

    if need_best_ratio:
        plt.subplot(2, 1, 2)
        for value, policy in combinations:
            plt.plot(np.array(ratio_stats[combination_to_str(value, policy)]).mean(axis=0), label=combination_to_str(value, policy))

    plt.legend()
    plt.show()
