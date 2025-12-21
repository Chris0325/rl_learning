from utils.bandit_util import *

experiment(k=10, bandit_class=Bandit, policies=[EpsilonGreedyConstantStepBandit(a=.1, epsilon=0, init_value=5), EpsilonGreedyConstantStepBandit(a=.1, epsilon=.1)], runs=2000, need_best_ratio=False)
