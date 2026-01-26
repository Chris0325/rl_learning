import logging
import functools
import numpy as np
import pandas as pd
from enum import Enum
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import softmax
from collections import defaultdict
from more_itertools import windowed

np.random.seed(0)
logging.getLogger().setLevel(logging.WARN)


def print_matrix(V, type='dataframe'):
    nrow, ncol = V.shape
    if type == 'dataframe':
        df = pd.DataFrame(V)
        print(df)
    else:
        plt.axis('off')
        table = plt.table(cellText=V, loc='center', cellLoc='center')
        for i in range(nrow):
            for j in range(ncol):
                cell = table[(i, j)]
                cell.set_height(1.0/nrow)
                cell.set_width(1.0/ncol)
        plt.show()


def coordinate_to_index(s, *, ncol):
    return s[0] * ncol + s[1]


def index_to_coordinate(index, *, ncol):
    return divmod(index, ncol)


def tabular_states(nrow, ncol):
    return [(i, j) for i in range(nrow) for j in range(ncol)]


class Transition:
    def __init__(self, s, r, prob):
        self.s, self.r, self.prob = s, r, prob
