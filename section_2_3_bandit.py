from utils.bandit_util import *

# experiment(k=10, bandit_class=Bandit, plans=[[SlowSampleAverageValue(), EpsilonGreedyPolicy(epsilon=epsilon)] for epsilon in [0, .01, .1]], runs=2000)
experiment(k=10, bandit_class=Bandit, plans=[[SampleAverageValue(), EpsilonGreedyPolicy(epsilon=epsilon)] for epsilon in [0, .01, .1]], runs=2000)
