from utils.bandit_util import *

experiment(
    k=10,
    bandit_class=NonstationaryBandit,
    plans=[
            [SampleAverageValue(horizon=10_000), EpsilonGreedyPolicy(epsilon=.1)],
            [ConstantStepValue(a=.1, horizon=10_000), EpsilonGreedyPolicy(epsilon=.1),
        ]
    ],
    runs=2000,
    need_best_ratio=False
)
