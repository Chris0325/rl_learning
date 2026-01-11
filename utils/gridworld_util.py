from utils.util import *

action_name = {(-1, 0): '↑', (1, 0): '↓', (0, -1): '←', (0, 1): '→'}
action_space = sorted(list(action_name.keys()))


def random_pi(s):
    return np.ones(len(action_space)) / len(action_space)
