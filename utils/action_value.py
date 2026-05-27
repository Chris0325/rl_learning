from utils.util import *


class Value:

    def __init__(self, k, initial_value=0):
        self.action_history, self.reward_history, self.average_reward = [], [], initial_value
        self.action_value, self.action_count = np.ones(k) * initial_value, np.zeros(k)


class SampleAverageValue(Value):

    def __str__(self):
        return 'sa'

    def update(self, action, reward):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_count[action] += 1
    
        self.action_value[action] += 1 / self.action_count[action] * (reward - self.action_value[action])
        self.average_reward += 1 / len(self.reward_history) * (reward - self.average_reward)


class ConstantStepValue(Value):
    
    def __init__(self, k, a, initial_value=0):
        super().__init__(k, initial_value)
        self.a = a

    def __str__(self):
        return f'cs,a={self.a}'

    def update(self, action, reward):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_count[action] += 1
    
        self.action_value[action] += self.a * (reward - self.action_value[action])
        self.average_reward += self.a * (reward - self.average_reward)


class ConstantStepUnbiasValue(Value):

    def __init__(self, k, a, initial_value=0):
        super().__init__(k, initial_value)
        self.a, self.demominator = a, 0

    def __str__(self):
        return f'ubcs,a={self.a}'

    def update(self, action, reward):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_count[action] += 1

        self.demominator += self.a * (1 - self.demominator)
        self.action_value[action] += self.a / self.demominator * (reward - self.action_value[action])
        self.average_reward += self.a / self.demominator * (reward - self.average_reward)
