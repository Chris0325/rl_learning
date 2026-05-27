from utils.util import *

class Policy:

    def __init__(self, value, **kwargs):
        self.value, self.kwargs = value, kwargs

    def run(self, env, horizon):
        for t in range(horizon):
            action = self.choose(t)
            reward = env.step(action)
            self.value.update(action, reward)


class EpsilonGreedyPolicy(Policy):

    def __str__(self):
        return f'e={self.kwargs["epsilon"]},{str(self.value)}'

    def choose(self, t):
        if np.random.random() < self.kwargs["epsilon"]:
            return np.random.choice(len(self.value.action_value))
        else:
            return argmax(self.value.action_value)


class UpperConfidenceBoundPolicy(Policy):

    def __str__(self):
        return f'ucb,c={self.kwargs["c"]},{str(self.value)}'

    def choose(self, t):
        return argmax(self.value.action_value + self.kwargs["c"] * np.sqrt(np.log(t + 1) / (self.value.action_count + 1e-8)))


class GradientPolicy(Policy):

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.preferences = np.zeros(self.kwargs["k"])

    def __str__(self):
        return f'g,a={self.kwargs["a"]},b={int(self.kwargs["with_baseline"])}'

    def choose(self, t):
        action_probs = softmax(self.preferences)
        return np.random.choice(self.kwargs["k"], p=action_probs), action_probs

    def update(self, action, reward, action_probs):
        average_reward = self.value.average_reward if self.kwargs["with_baseline"] else 0
        for i in range(self.kwargs["k"]):
            if i == action:
                self.preferences[i] += self.kwargs["a"] * (reward - average_reward) * (1 - action_probs[i])
            else:
                self.preferences[i] -= self.kwargs["a"] * (reward - average_reward) * action_probs[i]

    def run(self, env, horizon):
        for t in range(horizon):
            action, action_probs = self.choose(t)
            reward = env.step(action)
            self.update(action, reward, action_probs)
            self.value.update(action, reward)
