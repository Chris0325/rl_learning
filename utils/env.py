from utils.util import *


class Bandit:

    def __init__(self, k, initial_mean=0, initial_variance=1, action_variance=1):
        self.k, self.means, self.action_variance = k, np.random.normal(loc=initial_mean, scale=np.sqrt(initial_variance), size=k), action_variance
        self.best_actions_ = np.where(self.means == self.means.max())[0]

    def evolve(self):
        ...

    def step(self, i):
        reward = np.random.normal(loc=self.means[i], scale=np.sqrt(self.action_variance))
        self.evolve()
        return reward

    def best_actions(self):
        return self.best_actions_


class NonstationaryBandit(Bandit):

    def __init__(self, k, initial_mean=0, evolve_mean=0, evolve_variance=1, action_variance=1):
        self.k, self.means, self.evolve_mean, self.evolve_variance, self.action_variance = k, np.ones(k) * initial_mean, evolve_mean, evolve_variance, action_variance

    def evolve(self):
        self.means += np.random.normal(loc=self.evolve_mean, scale=np.sqrt(self.evolve_variance), size=self.k)

    def best_actions(self):
        return np.where(self.means == self.means.max())[0]
