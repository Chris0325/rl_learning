from utils.bandit_util import *

# experiment(k=10, bandit_class=Bandit, policies=[SlowEpsilonGreedySampleAverageBandit(epsilon=epsilon) for epsilon in [0, .01, .1]], runs=2000)
experiment(k=10, bandit_class=Bandit, policies=[EpsilonGreedySampleAverageBandit(epsilon=epsilon) for epsilon in [0, .01, .1]], runs=2000)
