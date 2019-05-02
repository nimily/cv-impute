import numpy as np
import numpy.random as npr

from impute import Dataset
from impute import SoftImpute
from impute import EntryTraceLinearOp as LinOp

from utils import CvProblem
from utils import cv_impute
from utils import relative_error
from utils import split_and_add_to_subproblems


def init_problem(shape):
    imputer = SoftImpute(shape)
    train = Dataset(LinOp(shape), [])
    test = Dataset(LinOp(shape), [])

    return CvProblem(imputer, train, test)


def __main__():
    npr.seed(4)

    sd = 1.0
    rank = 3
    n_row = 100
    n_col = 100
    n_fold = 10
    step = 100
    n_sample = 1000

    n_entry = n_row * n_col

    shape = n_row, n_col

    bl = npr.randn(n_row, rank)
    br = npr.randn(n_col, rank)
    b = bl @ br.T

    probs = [init_problem(shape) for _ in range(n_fold)]

    obs = npr.choice(n_entry, n_sample, replace=False)
    xs = [(o % n_col, o // n_col, 1) for o in obs]
    ys = [b[i, j] + npr.randn() * sd for i, j, _ in xs]

    head = 0
    while head < n_sample:
        xss = xs[head:head + step]
        yss = ys[head:head + step]

        split_and_add_to_subproblems(probs, xss, yss)

        bh, *_ = cv_impute(probs)
        error = relative_error(b, bh)

        print(f'relative error = {error}')

        head += step


if __name__ == '__main__':
    __main__()
