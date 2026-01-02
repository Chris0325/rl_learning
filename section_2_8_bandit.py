from utils.bandit_util import *

experiment(
    k=10,
    bandit_class=Bandit,
    plans=[
        [GradientValue(a=.1), 'w_b'],
        [GradientValue(a=.4), 'w_b'],
        [GradientValue(a=.1), 'w/t_b'],
        [GradientValue(a=.4), 'w/t_b'],
    ],
    runs=2000,
    need_best_ratio=False,
    bandit_mean=4,
)
