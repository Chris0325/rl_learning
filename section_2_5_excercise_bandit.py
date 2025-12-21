from utils.bandit_util import *

experiment(k=10, bandit_class=NonstationaryBandit, policies=[EpsilonGreedySampleAverageBandit(epsilon=.1, horizon=10_000), EpsilonGreedyConstantStepBandit(a=.1, epsilon=.1, horizon=10_000)], runs=2000, need_best_ratio=False)
