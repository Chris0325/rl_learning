from utils.bandit_util import *

experiment(
    k=10,
    bandit_class=Bandit,
    combinations=[
        [SampleAverageValue(), EpsilonGreedyPolicy(epsilon=.1)],
        [SampleAverageValue(), UpperConfidenceBoundPolicy(c=2)]
    ],
    runs=2000,
    need_best_ratio=False,
)
