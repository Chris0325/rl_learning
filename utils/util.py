import matplotlib.pyplot as plt


def print_matrix(V):
    nrow, ncol = V.shape
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
