from utils.bandit_util import *

experiment(
    k=10,
    bandit_class=NonstationaryBandit,
    combinations=[
        [ConstantStepValue(a=.1), EpsilonGreedyPolicy(epsilon=.1)],
        [ConstantStepUnbiasValue(a=.1), EpsilonGreedyPolicy(epsilon=.1)],
    ],
    runs=2000,
    need_best_ratio=False,
)
