import pandas as pd


def print_matrix(v, *, nrow, ncol, round=1):
    print(pd.DataFrame(v.reshape((nrow, ncol))).round(round))
