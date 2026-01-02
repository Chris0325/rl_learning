from utils.bandit_util import *

experiment(
    k=10,
    bandit_class=Bandit,
    plans=[
        [ConstantStepValue(a=.1, init_value=5), EpsilonGreedyPolicy(epsilon=0)],
        [ConstantStepValue(a=.1), EpsilonGreedyPolicy(epsilon=.1)],
    ],
    runs=2000,
    need_best_ratio=False
)