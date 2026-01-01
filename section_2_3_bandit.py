from utils.bandit_util import *

# experiment(k=10, bandit_class=Bandit, combinations=[[SlowSampleAverageValue(), EpsilonGreedyPolicy(epsilon=epsilon)] for epsilon in [0, .01, .1]], runs=2000)
experiment(k=10, bandit_class=Bandit, combinations=[[SampleAverageValue(), EpsilonGreedyPolicy(epsilon=epsilon)] for epsilon in [0, .01, .1]], runs=2000)
