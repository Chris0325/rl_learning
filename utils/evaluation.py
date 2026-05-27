from utils.util import *


def best_action_ratio(action_history, best_actions):
    return np.isin(np.array(action_history), best_actions).cumsum() / np.arange(1, len(action_history)+1)
