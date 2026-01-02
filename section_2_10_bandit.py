from utils.bandit_util import *

banchmark(
    k=10,
    bandit_class=Bandit,
    plans={
        'gradient': [[exponent, (GradientValue(a=2**exponent), 'w_b')] for  exponent in range(-5, 3)],
        'e-greedy': [[exponent, (SampleAverageValue(), EpsilonGreedyPolicy(epsilon=2**exponent))] for exponent in range(-7, -1)],
        'op-greedy': [[exponent, (ConstantStepValue(a=.1, init_value=2**exponent), EpsilonGreedyPolicy(epsilon=0))] for exponent in range(-2, 3)],
        'ucb': [[exponent, (SampleAverageValue(), UpperConfidenceBoundPolicy(c=2**exponent))] for exponent in range(-4, 3)],
    },
    runs=2000,
)
