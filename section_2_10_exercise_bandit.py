from utils.bandit_util import *

steps=2000

banchmark(
    k=10,
    bandit_class=NonstationaryBandit,
    plans={
        'gradient': [[exponent, (GradientValue(a=2**exponent, horizon=steps), 'w_b')] for  exponent in range(-5, 3)],
        'e-greedy': [[exponent, (SampleAverageValue(horizon=steps), EpsilonGreedyPolicy(epsilon=2**exponent))] for exponent in range(-7, -1)],
        'cse-greedy': [[exponent, (ConstantStepValue(a=.1,horizon=steps), EpsilonGreedyPolicy(epsilon=2**exponent))] for exponent in range(-7, -1)],
        'op-greedy': [[exponent, (ConstantStepValue(a=.1, init_value=2**exponent, horizon=steps), EpsilonGreedyPolicy(epsilon=0))] for exponent in range(-2, 3)],
        'ucb': [[exponent, (SampleAverageValue(horizon=steps), UpperConfidenceBoundPolicy(c=2**exponent))] for exponent in range(-4, 3)],
    },
    runs=2000,
    from_step=-steps//2,
)
