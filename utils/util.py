import random
import logging
import functools
import numpy as np
import pandas as pd
from enum import Enum
from tqdm import tqdm
from scipy import stats
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import softmax
from collections import defaultdict, Counter
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


def argmax(array):
    return np.random.choice(np.where(array == array.max())[0])


def to_index(s, *, size):
    return s[0] * size[1] + s[1]


def to_coordinate(index, *, size):
    return divmod(index, size[1])


def tabular_states(size):
    return [(i, j) for i in range(size[0]) for j in range(size[1])]


class Transition:
    def __init__(self, s, r, prob):
        self.s, self.r, self.prob = s, r, prob
