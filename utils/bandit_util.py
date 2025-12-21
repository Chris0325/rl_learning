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


class SlowEpsilonGreedySampleAverageBandit:

    def __init__(self, epsilon, horizon=1000, init_value=0):
        self.epsilon, self.horizon, self.init_value = epsilon, horizon, init_value

    def __str__(self):
        return f'e={self.epsilon}'

    def choose(self, action_value):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(action_value))
        else:
            return np.random.choice(np.where(action_value == action_value.max())[0])

    def run(self, bandit, need_best_ratio=True):
        best_actions = np.where(bandit.means == bandit.means.max())[0]
        logging.debug(f'epsilon: {self.epsilon}, best_actions: {best_actions.tolist()}')

        action_history, reward_history, action_rewards = [], [], defaultdict(list)

        for _ in range(self.horizon):
            action_value = np.array([np.array(action_rewards[i]).mean() if action_rewards[i] else self.init_value for i in range(bandit.k)])
            action = self.choose(action_value)
            action_history.append(action)

            reward = bandit.reward(action)
            reward_history.append(reward)

            action_rewards[action].append(reward)
        
        best_ratio = None
        if need_best_ratio:
            action_history = np.array(action_history)
            best_hits = np.isin(action_history, best_actions).cumsum()
            best_ratio = [count / (i + 1) for i, count in enumerate(best_hits)]
        return best_ratio, reward_history


class EpsilonGreedySampleAverageBandit(SlowEpsilonGreedySampleAverageBandit):

    def run(self, bandit, need_best_ratio=True):
        best_actions = np.where(bandit.means == bandit.means.max())[0]
        logging.debug(f'epsilon: {self.epsilon}, best_actions: {best_actions.tolist()}')

        action_history, reward_history = [], []
        action_value, action_count = np.ones(bandit.k) * self.init_value, np.zeros(bandit.k)

        for _ in range(self.horizon):

            action = self.choose(action_value)
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


class EpsilonGreedyConstantStepBandit(SlowEpsilonGreedySampleAverageBandit):

    def __init__(self, a, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a

    def __str__(self):
        return f'cs,e={self.epsilon}'

    def run(self, bandit, need_best_ratio=False):
        action_history, reward_history = [], []
        action_value = np.ones(bandit.k) * self.init_value

        for _ in range(self.horizon):

            action = self.choose(action_value)
            action_history.append(action)

            reward = bandit.reward(action)
            reward_history.append(reward)

            action_value[action] += self.a * (reward - action_value[action])

        return None, reward_history


def experiment(k, bandit_class, policies, runs, need_best_ratio=True):
    ratio_stats, reward_stats = defaultdict(list), defaultdict(list)
    for policy in policies:
        for _ in tqdm(range(runs), desc=str(policy)):
            bandit = bandit_class(k)
            best_ratio, reward_history = policy.run(bandit, need_best_ratio)
            ratio_stats[str(policy)].append(best_ratio)
            reward_stats[str(policy)].append(reward_history)

    plt.subplot(2, 1, 1)
    for policy in policies:
        plt.plot(np.array(reward_stats[str(policy)]).mean(axis=0), label=str(policy))
    plt.legend()

    if need_best_ratio:
        plt.subplot(2, 1, 2)
        for policy in policies:
            plt.plot(np.array(ratio_stats[str(policy)]).mean(axis=0), label=str(policy))

    plt.legend()
    plt.show()
